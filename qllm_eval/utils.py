import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from fastchat.model import get_conversation_template

def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

# build model and tokenizer
def build_model_and_enc(model_path, use_flash_attn, kv_bit=16, kv_group_size=128):
    print(f"* Building model {model_path}")

    # weither trust remote code
    if 'chatglm' in model_path or 'mpt' in model_path or 'stable' in model_path:
        trust_remote_code = True
    else:
        trust_remote_code = False

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if use_flash_attn and 'chatglm' not in model_path and 'mpt' not in model_path:
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
    elif use_flash_attn and 'mpt' in model_path:
        config.attn_config['attn_impl'] = 'triton'
    else:
        config._flash_attn_2_enabled = False
        config._attn_implementation = None

    # add the kv quantization parameters
    config.kv_bit = kv_bit
    config.kv_group_size = kv_group_size

    # load tokenizer
    if 'mpt' in model_path or 'stable' in model_path:
        use_fast = True
    else:
        use_fast = False
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)

    # load model
    kwargs = {"torch_dtype": torch.float16, "device_map": "balanced"}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    return model, enc

def download_model(model_name, use_auth_token):
    # download tokenizer
    while True:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token, trust_remote_code=True)
        except:
            continue
        break

    # download model
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=use_auth_token, device_map="auto", resume_download=True, trust_remote_code=True)
        except:
            continue
        break


def format_chat_prompt(input, model_name):
    if "longchat" in model_name.lower():
        conv = get_conversation_template("vicuna")
    else:
        conv = get_conversation_template(model_name)

    # add system call
    if 'llama' in model_name.lower():
        conv.set_system_message("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


import transformers
import transformers.models.llama.modeling_llama
from torch.distributed import get_rank, is_initialized
from functools import partial

def rank0_print(*args):
    if is_initialized():
        if get_rank() == 0:
            print(*args)
    else:
        print(*args)

class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, ratio, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work.
        self.ratio = ratio
        max_position_embeddings *= ratio
        rank0_print(f"Condensing Positional embeddings from {max_position_embeddings} to {max_position_embeddings // ratio}")
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype) / ratio
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, ratio=ratio)
