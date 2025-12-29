import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from .transformer import _get_qkv_projections, _get_added_kv_projections, dispatch_attention_fn

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Parameter validation
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
            
        self.down = nn.Linear(in_channels, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_channels, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights"""
        nn.init.normal_(self.down.weight, std=1 / self.rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure consistent device and dtype
        hidden_states = hidden_states.to(self.down.weight.device, self.down.weight.dtype)
        
        down_hidden_states = self.down(hidden_states)
        hidden_states = self.up(down_hidden_states)
        
        if self.network_alpha is not None:
            hidden_states *= self.network_alpha / self.rank
        
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1)

        return hidden_states

class WanAttnProcessorLora(nn.Module):
    _attention_backend = None
    _parallel_config = None

    def __init__(self, in_channels, out_channels, rank, network_alpha, device, dtype, lora_modules=['q', 'k', 'v', 'out']):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        if lora_modules is not None and 'q' in lora_modules:
            self.lora_to_q = LoRALinearLayer(in_channels, out_channels, rank, network_alpha, device, dtype)
        if lora_modules is not None and 'k' in lora_modules:
            self.lora_to_k = LoRALinearLayer(in_channels, out_channels, rank, network_alpha, device, dtype)
        if lora_modules is not None and 'v' in lora_modules:
            self.lora_to_v = LoRALinearLayer(in_channels, out_channels, rank, network_alpha, device, dtype)
        if lora_modules is not None and 'out' in lora_modules:
            self.lora_to_out = LoRALinearLayer(in_channels, out_channels, rank, network_alpha, device, dtype)

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if hasattr(self, 'lora_to_q'):
            query += self.lora_to_q(hidden_states)
        if hasattr(self, 'lora_to_k'):
            key += self.lora_to_k(encoder_hidden_states)
        if hasattr(self, 'lora_to_v'):
            value += self.lora_to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        out_hidden_states = attn.to_out[0](hidden_states)
        if hasattr(self, 'lora_to_out'):
            out_hidden_states += self.lora_to_out(hidden_states)
        hidden_states = attn.to_out[1](out_hidden_states)
        return hidden_states