import argparse
import os
from tqdm import tqdm

import torch

from longeval.utils import maybe_monkey_patch, load_testcases, test_topics_one_sample, test_lines_one_sample

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from qllm_eval.methods.rep.apply_rep import apply_awq

def get_output_dir(args):
    path = args.model_name_or_path

    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = f"{args.test_dir}/{args.task}/predictions/{name}/w_{str(args.w_bit)}_a_{str(args.a_bit)}_kv_{str(args.kv_bit)}"
    if args.rep_file:
        output_dir += '_' + args.rep_file.split('/')[-2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"output to {output_dir}")
    return output_dir

def longeval_test(model, tokenizer, output_dir, args):
    if args.task == "topics":
        for num_topics in [5, 10, 15, 20, 25]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(args.test_dir, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
            
            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)

            print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************")
            if args.eval_shortest_only:
                break
            
    elif args.task == "lines":
        for num_lines in [200, 300, 400, 500, 600, 680]:
        # for num_lines in [600,680]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
            
            output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
            accuracy = num_correct / len(test_cases)

            with open(output_file, "a+") as f:
                f.write(f"Accuracy: {accuracy}")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
            if args.eval_shortest_only:
                break
    else:
        print(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default='/share/datasets/public_models/lmsys_vicuna-7b-v1.5-16k', help="model path")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--task", type=str, default='lines', help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--max_gpu_memory", type=int, default=80, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--framework", type=str, default=None, help="Framework for serving")
    parser.add_argument("--rep_file", type=str, help="path to load the reparameterization factors")
    # quantization config
    parser.add_argument("--w_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=16)
    parser.add_argument("--a_group_size", type=int, default=128)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_bit", type=int, default=16)
    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    model, tokenizer = build_model_and_enc(args.model_name_or_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    if args.rep_file:
        rep_results = torch.load(args.rep_file, map_location="cpu")
        apply_awq(model, rep_results)

    # quantize model
    model = quantize_model(model, args)
    
    longeval_test(model, tokenizer, output_dir, args)
