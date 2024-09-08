# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import dropout_add

from ...quant_funcs import pseudo_quantize_tensor


class BloomAttention(nn.Module):
    def __init__(self, config: BloomConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.layer_idx = layer_idx

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _reshape(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) and reshapes to (bs, heads, len, dim) shape
        without making any copies, results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, num_heads, seq_length, head_dim]
            key: [batch_size, num_heads, seq_length, head_dim]
            value: [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query_layer = fused_qkv[..., 0, :].transpose(1, 2)
        key_layer = fused_qkv[..., 1, :].transpose(1, 2)
        value_layer = fused_qkv[..., 2, :].transpose(1, 2)
        return query_layer, key_layer, value_layer

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        # TODO: quantization
        if self.config.kv_bit < 16:
            key_layer = key_layer.transpose(1, 2)
            key_layer = pseudo_quantize_tensor(key_layer, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size, inplace=True)
            key_layer = key_layer.transpose(1, 2)
            value_layer = pseudo_quantize_tensor(value_layer, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size, inplace=True)
        
        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(1, 2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        kv_length = cache_position[-1] + 1  # cache position is 0-indexed while length should start from 1

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, :kv_length]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs