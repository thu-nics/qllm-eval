import argparse
import torch

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from qllm_eval.methods.rep.apply_rep import apply_awq

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--rep_file", type=str, help="path to load the reparameterization factors")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
args = parser.parse_args()


def main():
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
    
    if args.rep_file:
        rep_results = torch.load(args.rep_file, map_location="cpu")
        apply_awq(model, rep_results)

    # quantize model
    model = quantize_model(model, args)

    # save the quantized model
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=False)
        enc.save_pretrained(args.output_path)

    # evaluation
    # TODO: add evaluation functions
    prompt = "Hello, my name is human, and I like drinking"
    input_ids = enc(prompt, return_tensors="pt")['input_ids'].to(next(model.parameters()).device)
    output = model.generate(input_ids, do_sample=True, max_length=50, top_p=0.95, top_k=60)
    print(enc.decode(output[0]))


if __name__ == "__main__":
    main()
