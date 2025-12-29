# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from src.models.action_former.policy import PolicyAutoEncoder
import copy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. If provided, enables
            two-stage denoising where `transformer` handles high-noise stages and `transformer_2` handles low-noise
            stages. If not provided, only `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # norm = [{"mean": [[-0.00029581971199498956, 0.36787371854635903, 0.014954494843272544, 0.9370392243315385, 0.11111164567436758], [-0.0021755785979208826, 0.49628562756622624, 0.00980841430248885, -0.09450662412121616, 0.12359155946000558], [-0.006537387456843426, 0.4946204086256411, 0.0027271122422390637, -0.07584896793281377, 0.1377745917807692], [-0.011895896263471145, 0.46674328577621704, -0.001475823814134049, -0.050122957131872724, 0.15255900776506587], [-0.01560200421017946, 0.4222075003864933, -0.0035487155837610903, -0.031046077048903088, 0.16974895840349477]], "std": [[0.06789973776063911, 0.41139151773138893, 0.3429346606197173, 0.06426253708659885, 0.31427034201448506], [0.25218543437889096, 0.37416442328924704, 0.2105894962514581, 0.14495055436599424, 0.32911500404911576], [0.35979320234775136, 0.3478510248831022, 0.14209455268732346, 0.16297993201356886, 0.34466324672121856], [0.4189104251465312, 0.3565332171939786, 0.1310259377759319, 0.13597135349889866, 0.35956189580488707], [0.44369571087279847, 0.3881790982497418, 0.13781612528520876, 0.10685504625084495, 0.37541210625714155]], "min": [[-0.4898879826068878, -3.371747652636259e-06, -0.500000110699574, 0.8660253398720004, 0.0], [-0.8396326303482086, -3.3974647521972656e-06, -0.5176382086374279, -0.3660254840604251, 0.0], [-0.9391180276870728, -0.15929907560349102, -0.5176382086374279, -0.500000059133653, 0.0], [-0.9291015863418579, -0.5872985124588013, -0.5176382086374279, -0.5176381337764339, 0.0], [-0.9319192171096802, -0.8579317331314087, -0.5176382086374279, -0.5176381337764339, 0.0]], "max": [[0.5872985124588013, 0.9557347893714905, 0.5000001061835708, 1.0, 1.0], [0.8579317331314087, 1.0390557050704985, 0.5176382086374279, 0.03407419667237266, 1.0], [0.9092906713485659, 0.9378378391265869, 0.5176382086374279, 0.25881910207964864, 1.0], [0.9166926145553589, 0.9238722324371338, 0.5176382086374279, 0.44828776491701233, 1.0], [0.9166924357414246, 0.9437556266784668, 0.5176382086374279, 0.5176381220243294, 1.0]], "cliped_min": [[-0.2156829982995987, -1.1842378646990687e-14, -0.5000000476197127, 0.8660253744433825, 0.0], [-0.5892562866210938, 0.0, -0.5000000231604275, -0.3660254456623787, 0.0], [-0.8049386143684387, 0.0, -0.5000000267263442, -0.5000000329932205, 0.0], [-0.8333336710929871, -0.2156828194856614, -0.5000000193662221, -0.5000000267695113, 0.0], [-0.8333336710929871, -0.5892557501792908, -0.5000000232517577, -0.4999999894331238, 0.0]], "cliped_max": [[0.2156829684972763, 0.8640206348896021, 0.5000000476197127, 1.0, 1.0], [0.5892562866210938, 0.8660314083099395, 0.5000000214928615, 0.03407417595960183, 1.0], [0.8049386143684387, 0.8333348035812378, 0.5000000244794888, 0.15891862945162905, 1.0], [0.8333336710929871, 0.8333350419998169, 0.5000000154900036, 0.24118088154568645, 1.0], [0.8333336710929871, 0.8333346843719482, 0.5000000084622189, 0.2588190533922409, 1.0]]}]
        self.policy_head = PolicyAutoEncoder(norm_type='mean_std')
        self.scheduler_waypoints = copy.deepcopy(scheduler)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    @torch.no_grad()
    def encode_video(self, video, sample_mode="argmax"):
        # 确保是 5D 张量 (B, C, T, H, W)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        video = video.to(device=self.vae.device, dtype=self.vae.dtype)

        latent = retrieve_latents(self.vae.encode(video), sample_mode=sample_mode)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(self.vae.config.latents_std)
            .view(1, -1, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latent = (latent - latents_mean) * latents_std
        return latent

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        shape_waypoints = (
            batch_size,
            5,
            5,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # import pdb; pdb.set_trace()
        right_frame, left_frame, history_frames = None, None, None
        if image.ndim == 4: # [F,C,H,W]
            if image.shape[0] == 1:
                # 单帧模式
                front_frame = image
            elif image.shape[0] == 3:
                # 三视图模式: [front, right, left]
                front_frame = image[0:1]
                right_frame = image[1:2]
                left_frame = image[2:3]
            else:
                # 带历史帧模式: [...history, front, right, left]
                history_frames = image[:-3]
                front_frame = image[-3:-2]
                right_frame = image[-2:-1]
                left_frame = image[-1:]

        front_frame = front_frame.unsqueeze(0).transpose(1,2)  # (B, C, 1, H, W)
        latent_front = self.encode_video(front_frame, sample_mode="argmax")
        latent_front = latent_front.to(dtype).repeat(batch_size, 1, 1, 1, 1)

        # ===== 编码左右视图（如果存在）=====
        additional_latents = []
        
        if right_frame is not None:
            right_frame = right_frame.unsqueeze(0).transpose(1,2)  # (B, C, 1, H, W)
            latent_right = self.encode_video(right_frame, sample_mode="argmax")
            latent_right = latent_right.to(dtype).repeat(batch_size, 1, 1, 1, 1)
            additional_latents.append(latent_right)
        
        if left_frame is not None:
            left_frame = left_frame.unsqueeze(0).transpose(1,2)  # (B, C, 1, H, W)
            latent_left = self.encode_video(left_frame, sample_mode="argmax")
            latent_left = latent_left.to(dtype).repeat(batch_size, 1, 1, 1, 1)
            additional_latents.append(latent_left)
        
        # ===== 编码历史帧（如果存在）=====
        if history_frames is not None:
            # history_frames shape: (N_history, C, H, W)
            # 需要转换为 (N_history, C, 1, H, W) 然后逐帧编码
            history_latents = []
            for i in range(history_frames.shape[0]):
                hist_frame = history_frames[i:i+1].unsqueeze(0).transpose(1,2)  # (B, C, F, H, W)
                latent_hist = self.encode_video(hist_frame, sample_mode="argmax")
                latent_hist = latent_hist.to(dtype).repeat(batch_size, 1, 1, 1, 1)
                history_latents.append(latent_hist)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # 用前视图替换第一帧
        latents[:, :, 0] = latent_front[:, :, 0]
        
        # 拼接额外的潜在变量（右、左、历史）
        if additional_latents:
            latents = torch.cat([latents] + additional_latents, dim=2)
        
        if history_frames is not None:
            latents = torch.cat(history_latents + [latents], dim=2)

        latents_waypoints = randn_tensor(shape_waypoints, generator=generator, device=device, dtype=dtype)

        return latents, latents_waypoints

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PipelineImageInput, List[PipelineImageInput]]=None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler_waypoints.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, latents_waypoints = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        history_len = latents.shape[2] - 8
        latent_history_first_frame = latents[:, :, :-7].clone()
        latent_left_right = latents[:, :, -2:].clone()
        # num_latents_frames = num_frames//self.vae_scale_factor_temporal+1
        num_latents_tokens = latents.shape[-1] * latents.shape[-2] * latents.shape[-3] // 4
        num_tokens_per_frame = latents.shape[-2] * latents.shape[-1] // 4
        mask_latents = torch.ones(latents.shape[0], num_latents_tokens, dtype=torch.float32, device=device)
        mask_latents[:, :num_tokens_per_frame*(history_len+1)] = 0  # history and front
        mask_latents[:, -2 * num_tokens_per_frame:] = 0 # left and right
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                latents_waypoints = latents_waypoints.to(transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                # import pdb; pdb.set_trace()
                timestep = timestep * mask_latents.to(timestep.device, timestep.dtype)
                with current_model.cache_context("cond"):
                    noise_pred, noise_pred_waypoints  = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        hidden_states_waypoints=latents_waypoints,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond, noise_uncond_waypoints = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            hidden_states_waypoints=latents_waypoints,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
                    noise_pred_waypoints = noise_uncond_waypoints + current_guidance_scale * (noise_pred_waypoints - noise_uncond_waypoints)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_waypoints = self.scheduler_waypoints.step(noise_pred_waypoints, t, latents_waypoints, return_dict=False)[0]
                # if i == len(timesteps) - 1:
                #     delta = 0 - t
                # else:
                #     delta = timesteps[i + 1] - t
                # latents = latents + noise_pred * (delta / 1000.)
                # latents_waypoints = latents_waypoints + noise_pred_waypoints * (delta / 1000.)

                latents[:, :, :latent_history_first_frame.shape[2]] = latent_history_first_frame
                latents[:, :, -latent_left_right.shape[2]:] = latent_left_right

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean # [b, c, f, h, w]
            latents = latents.transpose(1,2).flatten(0,1).unsqueeze(2) # [bf, c, 1, h, w]
            video = self.vae.decode(latents, return_dict=False)[0] # [bf, 3, 1, h, w]
            video = video.squeeze(2).unflatten(0, (latents_std.shape[0],-1)).transpose(1,2) # [b, 3, f, h, w]
            video = self.video_processor.postprocess_video(video, output_type=output_type) # [b, f, h, w, 3]

            waypoints = self.policy_head.decode(latents_waypoints)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video[0],waypoints.cpu())

        return WanPipelineOutput(frames=video)