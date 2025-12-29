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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ActionFormerHead, MultiLayerEmbedding
# from src.models.coupleflow.policy import FlowmatchingActionHeadConfig

class ActionFormer(nn.Module):
    def __init__(self, hidden_size=2048, query_action_layer=4, waypoint_number=5):
        super().__init__()  
        self.waypoint_number = 5
        self.predictor = nn.Linear(hidden_size,5)
        self.predict_angle = True
        self.use_arrive_list = True

        self.query_action_layer = query_action_layer
        if query_action_layer == 1:
            self.query_multihead_attn = nn.MultiheadAttention(
                embed_dim=hidden_size, 
                num_heads=4,
                batch_first=True)  
        else:
            self.query_multihead_multi_attn = ActionFormerHead(hidden_dim=hidden_size, num_heads=4, num_blocks=config.query_action_layer)
        self.query_action= nn.Parameter(torch.empty(1, waypoint_number, hidden_size))


    def forward(self, hidden_states, video_hidden_states=None,train=False,):
        """
        hidden_states: List[[b,l,2048],]
        """
        query_action = self.query_action.expand(hidden_states.shape[0], -1, -1)
        if self.query_action_layer==1: # go here
            action_features, _ = self.query_multihead_attn(query_action, hidden_states, hidden_states)
        else:
            action_features = self.query_multihead_multi_attn(query_action, hidden_states, hidden_states)

        action_features = action_features # [b, 5, c] 

        pred = self.predictor(action_features) # .view(-1, self.waypoint_number, 2)
        wp_pred = pred[:, :, :2]
        wp_angle_pred_raw = pred[:, :, 2:-1] # [b, 5,2]
        wp_angle_pred = torch.tanh(wp_angle_pred_raw)
        sin_pred = wp_angle_pred[:,:,0]
        cos_pred = wp_angle_pred[:,:,1]
        arrive_pred = pred[:, :, -1]
        wp_pred = torch.cumsum(wp_pred, dim=1)
        return wp_pred,arrive_pred,sin_pred,cos_pred

