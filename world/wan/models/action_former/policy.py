import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.models.qwen2_5_vl.cross_attention_dit import DiT
from transformers.feature_extraction_utils import BatchFeature
from torch.distributions import Beta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ActionDecoder, ActionEncoder, SinusoidalPositionalEncoding
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="bfloat16", metadata={"help": "Model data type."})
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding  dimension."}
    )
    decoder_hidden_size: int = field(default=1024, metadata={"help": "decoder hidden size."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=5, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=5, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=100, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=10,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class ActionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)

# Encoder with shared linear layers
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Shared layers for all categories
        self.W1 = nn.Linear(action_dim, hidden_size)  # (d -> w)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = nn.Linear(hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape

        # Expand timesteps across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # Forward pass
        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))

        x = self.W3(x)
        return x


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        #decoder_hidden_size为diffusion policy的输出维度，与action_decoder的输入维度一致
        self.decoder_hidden_size = config.decoder_hidden_size
        #input_embedding_dim为diffusion policy的输入维度与attention_head_dim*num_attention_heads对齐
        self.input_embedding_dim = config.input_embedding_dim
        diffusion_model_new_cfg={'attention_head_dim': config.input_embedding_dim//32,'dropout': 0.2,'final_dropout': True,'interleave_self_attention': True,'norm_type': 'ada_norm','num_attention_heads': 32,'num_layers': 16,'output_dim': self.decoder_hidden_size,'positional_embeddings': None}
        self.model = DiT(**diffusion_model_new_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,        
        )
        self.action_decoder = ActionDecoder(
            input_dim=self.decoder_hidden_size,
            hidden_dim=self.decoder_hidden_size,
            output_dim=self.action_dim,
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        alpha = torch.tensor(config.noise_beta_alpha, dtype=torch.float32)
        beta = torch.tensor(config.noise_beta_beta, dtype=torch.float32)

        # 初始化 Beta 分布
        self.beta_dist = Beta(alpha, beta)

        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size])
        sample = sample.to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature) -> BatchFeature:
        # Get vision and language embeddings.
        vl_embeds = backbone_output

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = action_features

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class PolicyAutoEncoder(ModelMixin, ConfigMixin):
    """
    归一化waypoint这些信息
    对waypoint、angle、arrive进行encode->token emb
    """
    @register_to_config
    def __init__(self, norm=None, norm_type='min_max'):
        super().__init__()

        self.norm = norm if norm is not None else [{"mean": [[-0.00029206654289737344, 0.3680045008659363, 0.014955667778849602, 0.937039315700531, 0.1111111119389534], [-0.002167902886867523, 0.4964025318622589, 0.024766579270362854, 0.8425331115722656, 0.12359096854925156], [-0.006535205524414778, 0.49472588300704956, 0.027493935078382492, 0.7669820785522461, 0.13777393102645874], [-0.011900688521564007, 0.4667980372905731, 0.02601882442831993, 0.7171341776847839, 0.15255826711654663], [-0.015607640147209167, 0.42227205634117126, 0.022465374320745468, 0.6865590810775757, 0.16974814236164093]], "std": [[0.06824818253517151, 0.4115539789199829, 0.34293508529663086, 0.06426256895065308, 0.3142704367637634], [0.2522902488708496, 0.37428611516952515, 0.5027111172676086, 0.1918523758649826, 0.3291151225566864], [0.35984712839126587, 0.3479742407798767, 0.5547965168952942, 0.32122474908828735, 0.3446633815765381], [0.4189571738243103, 0.35670235753059387, 0.5628238320350647, 0.4102112352848053, 0.3595620393753052], [0.44380590319633484, 0.38832738995552063, 0.5682132244110107, 0.45306530594825745, 0.3754122853279114]], "min": [[-0.4898879826068878, -3.371747652636259e-06, -0.5000001192092896, 0.8660253286361694, 0.0], [-0.8396326303482056, -3.3974647521972656e-06, -0.866025447845459, 0.49999988079071045, 0.0], [-0.9391180276870728, -0.1592990756034851, -1.0, -1.6292068494294654e-07, 0.0], [-0.9291015863418579, -0.5872985124588013, -1.0, -0.5000000596046448, 0.0], [-0.9319192171096802, -0.8579317331314087, -1.0, -0.8660253882408142, 0.0]], "max": [[0.5872985124588013, 1.0088157653808594, 0.5000001192092896, 1.0, 1.0], [0.8579317331314087, 1.0390557050704956, 0.866025447845459, 1.0, 1.0], [0.9092906713485718, 0.9378378391265869, 1.0, 1.0, 1.0], [0.9166926145553589, 0.9238722324371338, 1.0, 1.0, 1.0], [0.9166924357414246, 0.9437556266784668, 1.0, 1.0, 1.0]]}]
        self.norm_type = norm_type
        # self.pose_init() # TODO

    def encode(self, waypoints):
        norm = self.norm
        assert norm is not None # used for norm action
        angle = waypoints[:,:,-2] # [b, l]
        sin_angle = torch.sin(angle).unsqueeze(-1) # [b, l, 1]
        cos_angle = torch.cos(angle).unsqueeze(-1)
        waypoints = torch.cat([waypoints[:,:,:-2], sin_angle, cos_angle, waypoints[:,:,-1:]], dim=-1)
        dtype = waypoints.dtype
        device = waypoints.device
        # min_vals = torch.tensor(norm[0]['min']).mean(dim=0,keepdim=True).to(device,dtype)
        # max_vals = torch.tensor(norm[0]['max']).mean(dim=0,keepdim=True).to(device,dtype)
        # mean_vals = torch.tensor(norm[0]['mean']).mean(dim=0,keepdim=True).to(device,dtype)
        # std_vals = torch.tensor(norm[0]['std']).mean(dim=0,keepdim=True).to(device,dtype)
        min_vals = torch.tensor(norm[0]['min']).to(device,dtype)
        max_vals = torch.tensor(norm[0]['max']).to(device,dtype)
        mean_vals = torch.tensor(norm[0]['mean']).to(device,dtype)
        std_vals = torch.tensor(norm[0]['std']).to(device,dtype)
        delta_waypoints = torch.zeros_like(waypoints)
        delta_waypoints[:, 0] = waypoints[:, 0]
        delta_waypoints[:, 1:,:2] = waypoints[:, 1:,:2] - waypoints[:, :-1,:2]
        delta_waypoints[:, :,2:] = waypoints[:, :,2:]
        mean_vals = (mean_vals - min_vals) / (max_vals - min_vals + 1e-8)
        std_vals = std_vals / (max_vals - min_vals + 1e-8)
        delta_waypoints = (delta_waypoints - min_vals) / (max_vals - min_vals + 1e-8)
        if self.norm_type != 'min_max':
            delta_waypoints = (delta_waypoints - mean_vals) / (std_vals + 1e-8) 
        # waypoints = (waypoints - min_vals) / (max_vals - min_vals + 1e-8)
        # waypoints = (waypoints - mean_vals) / (std_vals + 1e-8)  

        return delta_waypoints

    def decode(self, pred_waypoints):
        norm = self.norm
        dtype = pred_waypoints.dtype
        device = pred_waypoints.device
        # min_vals = torch.tensor(norm[0]['min']).mean(dim=0,keepdim=True).to(device,dtype)
        # max_vals = torch.tensor(norm[0]['max']).mean(dim=0,keepdim=True).to(device,dtype)
        # mean_vals = torch.tensor(norm[0]['mean']).mean(dim=0,keepdim=True).to(device,dtype)
        # std_vals = torch.tensor(norm[0]['std']).mean(dim=0,keepdim=True).to(device,dtype)
        min_vals = torch.tensor(norm[0]['min']).to(device,dtype)
        max_vals = torch.tensor(norm[0]['max']).to(device,dtype)
        mean_vals = torch.tensor(norm[0]['mean']).to(device,dtype)
        std_vals = torch.tensor(norm[0]['std']).to(device,dtype)
        mean_vals = (mean_vals - min_vals) / (max_vals - min_vals + 1e-8)
        std_vals = std_vals / (max_vals - min_vals + 1e-8)
        # pred_waypoints = pred_waypoints * (std_vals + 1e-8) + mean_vals
        # pred_waypoints = pred_waypoints * (max_vals - min_vals + 1e-8) + min_vals
        if self.norm_type != 'min_max':
            pred_waypoints = pred_waypoints * std_vals + mean_vals
        pred_waypoints = pred_waypoints * (max_vals - min_vals) + min_vals
        arrive_pred = pred_waypoints[:,:,-1:]
        pred_waypoints[:,:,:2] = torch.cumsum(pred_waypoints[:,:,:2], dim=1)  
        sin_pred = pred_waypoints[..., 2:3]
        cos_pred = pred_waypoints[..., 3:4]
        pred_waypoints = torch.cat((pred_waypoints[:,:,:2],sin_pred,cos_pred,arrive_pred), dim=-1)
        return pred_waypoints