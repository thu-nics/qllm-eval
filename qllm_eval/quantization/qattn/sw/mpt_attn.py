"""Attention layers."""
import math
import warnings
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

from ...quant_funcs import pseudo_quantize_tensor

def is_flash_v2_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) >= version.parse('2.0.0')

def is_flash_v1_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) < version.parse('2.0.0')

def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool) -> bool:
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal

def repeat_kv_for_gqa(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Perform repeat of kv heads along a particular dimension.
    hidden.shape expected to be: (batch size, seq len, kv_n_heads, head_dim)
    n_rep: amount of repetitions of kv_n_heads
    Unlike torch.repeat_interleave, this function avoids allocating new memory.
    """
    if n_rep == 1:
        return hidden
    (b, s, kv_n_heads, d) = hidden.shape
    hidden = hidden[:, :, :, None, :].expand(b, s, kv_n_heads, n_rep, d)
    return hidden.reshape(b, s, kv_n_heads * n_rep, d)

def scaled_multihead_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: bool=False, kv_bit: int=4, kv_group_size: int=128) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)

    # TODO: quantization
    if kv_bit < 16:
        k_shape = k.shape
        k = k.transpose(2,3)
        k = pseudo_quantize_tensor(k, n_bits=kv_bit, q_group_size=kv_group_size)
        k = k.transpose(2,3)
        v = pseudo_quantize_tensor(v, n_bits=kv_bit, q_group_size=kv_group_size)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    if kv_n_heads > 1 and kv_n_heads < n_heads:
        k = repeat_kv_for_gqa(k.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
        v = repeat_kv_for_gqa(v.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)

def check_valid_inputs(*tensors: torch.Tensor, valid_dtypes: Optional[List[torch.dtype]]=None):
    if valid_dtypes is None:
        valid_dtypes = [torch.float16, torch.bfloat16]
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')

def flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: bool=False, kv_bit: int=4, kv_group_size: int=128) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError('Please install flash-attn==1.0.9 or flash-attn==2.3.2')
    check_valid_inputs(query, key, value)
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    
    # TODO: quantization
    if kv_bit < 16:
        key = pseudo_quantize_tensor(key, n_bits=kv_bit, q_group_size=kv_group_size)
        value = pseudo_quantize_tensor(value, n_bits=kv_bit, q_group_size=kv_group_size)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')
    (batch_size, seqlen) = query.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1):]
    (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
    (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(key, key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)
    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)
    if kv_n_heads == 1:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads, value_unpad.size(-1))
    elif kv_n_heads < n_heads:
        key_unpad = repeat_kv_for_gqa(key_unpad.view(batch_size, seqlen, kv_n_heads, -1), n_heads // kv_n_heads).view(batch_size * seqlen, n_heads, -1)
        value_unpad = repeat_kv_for_gqa(value_unpad.view(batch_size, seqlen, kv_n_heads, -1), n_heads // kv_n_heads).view(batch_size * seqlen, n_heads, -1)
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    if is_flash_v1_installed():
        output_unpad = flash_attn_interface.flash_attn_unpadded_func(q=query_unpad, k=key_unpad, v=value_unpad, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    elif is_flash_v2_installed():
        output_unpad = flash_attn_interface.flash_attn_varlen_func(q=query_unpad, k=key_unpad, v=value_unpad, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    else:
        raise RuntimeError('flash-attn==1.0.9 or flash-attn==2.3.2 is required.')
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return (output, None, past_key_value)

def triton_flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: bool=False, kv_bit: int=4, kv_group_size: int=128) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    try:
        from .flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            raise RuntimeError('Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU ' + 'and `pip install .[gpu]` if installing from llm-foundry source or ' + '`pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` ' + 'if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). ' + 'Note: (1) requires you have CMake and PyTorch already installed.')
    check_valid_inputs(query, key, value)
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    
    # TODO: quantization
    if kv_bit < 16:
        key = pseudo_quantize_tensor(key, n_bits=kv_bit, q_group_size=kv_group_size)
        value = pseudo_quantize_tensor(value, n_bits=kv_bit, q_group_size=kv_group_size)
    
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(f'Dropout not implemented for attn_impl: triton.')
    dropout_p = dropout_p if training else 0.0
    if needs_weights:
        raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
    if key_padding_mask is not None:
        warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=kv_n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=kv_n_heads)
    if kv_n_heads == 1:
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)
    elif kv_n_heads < n_heads:
        key = repeat_kv_for_gqa(key, n_heads // kv_n_heads)
        value = repeat_kv_for_gqa(value, n_heads // kv_n_heads)
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None, past_key_value)