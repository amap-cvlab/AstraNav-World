"""Optional visual foresight for the existing navigation ActionFormer agents."""
import torch
from safetensors.torch import load_file
from types import SimpleNamespace
from wan.checkpoint import load_action_former_checkpoint


def add_future_frame_arguments(parser):
    parser.add_argument("--predict-future-frames", action="store_true", default=False,
                        help="Run Wan future-frame generation at each navigation decision (default: off).")
    parser.add_argument("--wan-model-path", default=None,
                        help="Diffusers-format Wan2.2 TI2V base directory/repository; required for future frames.")


def build_navigation_model(checkpoint_path, predict_future_frames=False, wan_model_path=None):
    if not predict_future_frames:
        # No Wan base, transformer blocks, VAE or scheduler are constructed or downloaded.
        from wan.models.action_former.action_former_policy import ActionFormer
        head = ActionFormer(hidden_size=2048, query_action_layer=4, waypoint_number=5)
        load_action_former_checkpoint(head, checkpoint_path)
        head.to(device="cuda", dtype=torch.bfloat16).eval()
        return SimpleNamespace(action_former=head), None
    if not wan_model_path:
        raise ValueError("Future-frame prediction requires wan_model_path")
    from wan.models.action_former.transformer import WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(
        wan_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.load_action_former(hidden_size=2048, query_action_layer=4, waypoint_number=5)
    load_action_former_checkpoint(transformer.action_former, checkpoint_path)
    predictor = FutureFramePredictor(transformer, wan_model_path, checkpoint_path)
    return transformer, predictor


class FutureFramePredictor:
    def __init__(self, transformer, wan_model_path, checkpoint_path):
        # Imported only when foresight is enabled: the default path never loads a VAE.
        from wan.models.lora import WanAttnProcessorLora
        from wan.pipelines.pipeline_i2v_action_former import WanPipeline

        if checkpoint_path.endswith(".safetensors"):
            state = load_file(checkpoint_path, device="cpu")
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Future-frame prediction requires a full transformer state dict")
        transformer.condition_embedder.load_vlm_embedder(dim=2048)
        rank_keys = [key for key in state if ".processor.lora_to_q.down.weight" in key]
        if not rank_keys:
            raise ValueError("Future-frame prediction requires Wan LoRA weights, not an action-head-only checkpoint")
        ranks = {state[key].shape[0] for key in rank_keys}
        if len(ranks) != 1:
            raise ValueError("Inconsistent Wan LoRA ranks in checkpoint")
        rank = ranks.pop()
        if rank != 128:
            raise ValueError("The infer_wan.py generation configuration requires rank=128, alpha=128")
        dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        transformer.set_attn_processor({
            key: WanAttnProcessorLora(dim, dim, rank=rank, network_alpha=rank,
                                     device=transformer.device, dtype=torch.bfloat16)
            for key in transformer.attn_processors
        })
        expected = transformer.state_dict()
        required = {key for key in expected if key.startswith(("condition_embedder.vlm_embedder.", "action_former."))
                    or ".processor.lora_" in key}
        missing = sorted(required - state.keys())
        mismatched = [key for key in state if key in expected
                      and (not torch.is_tensor(state[key]) or state[key].shape != expected[key].shape)]
        if missing or mismatched:
            raise ValueError(f"Invalid future-frame weights: missing={missing}, shape_mismatches={mismatched}")
        transformer.load_state_dict({key: value for key, value in state.items() if key in expected}, strict=False)
        transformer.set_pred_latent()
        transformer.to(device="cuda", dtype=torch.bfloat16).eval()
        self.pipeline = WanPipeline.from_pretrained(
            wan_model_path, transformer=transformer, text_encoder=None, tokenizer=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda", torch.bfloat16)
        self.pipeline.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def predict(self, images, condition_emb):
        # Agent image order: history, left, right, front. Pipeline: history, front, right, left.
        if len(images) < 3:
            raise ValueError("Future-frame prediction requires left, right and front images")
        frames = list(images[:-3]) + [images[-1], images[-2], images[-3]]
        # Match the eight images dumped for infer_wan.py: repeat the first frame, then take the last eight.
        while len(frames) < 8:
            frames.insert(0, frames[0].copy())
        frames = frames[-8:]
        video, _ = self.pipeline(
            image=frames, prompt_embeds=condition_emb, condition_embs=condition_emb,
            negative_prompt_embeds=condition_emb[:, -1:].repeat(1, condition_emb.shape[1], 1),
            height=576, width=640, num_frames=21, num_inference_steps=20,
            generator=torch.Generator(device=condition_emb.device).manual_seed(42),
            output_type="np", return_dict=False, stop_timestep=100,
        )
        # Pipeline decodes individual latent frames: 5 history + current + 5 future + 2 side views.
        return video[:, 6:-2]
