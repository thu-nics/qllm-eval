#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run longchat to get responses.

Currently, this script only supports `longchat-13b-16k`.

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import math
import pathlib
import random
from copy import deepcopy

import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.prompting import Document, get_closedbook_qa_prompt, get_qa_prompt
from qllm_eval.utils import build_model_and_enc
from qllm_eval.quantization.quant_wrapper import quantize_model


# Copied from https://github.com/DachengLi1/LongChat/blob/43d71f03d7711a2ab3b78ee8d1e38b65bb7fd22f/longeval/utils.py
def maybe_monkey_patch(model_name: str, longchat_ratio: int):
    if "longchat" in model_name:
        if args.longchat_ratio != 8:
            raise ValueError("--longchat-ratio=8 is the only value currently supported.")
        else:
            from qllm_eval.utils import replace_llama_with_condense
            replace_llama_with_condense(longchat_ratio)

    import transformers  # noqa: F401


def main(args):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    print(args.output_path)

    maybe_monkey_patch(model_name=args.model_name, longchat_ratio=args.longchat_ratio)

    examples = []
    prompts = []
    all_model_documents = []

    # Fetch all of the prompts
    with xopen(args.input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if args.closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if args.use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if args.closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=args.prompt_mention_random_ordering,
                    query_aware_contextualization=args.query_aware_contextualization,
                )
            if 'chatglm' not in args.model_name.lower():
                prompt = format_chat_prompt(prompt, model_name=args.model_name)
            # else:
            #     prompt = prompt.strip('Answer:')
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    # Get responses for all of the prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    # load model and tokenizer
    model, tokenizer = build_model_and_enc(args.model_name, args.use_flash_attn, args.kv_bit, args.kv_group_size)
    model = quantize_model(model, args)
    print(model.generation_config)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    do_sample = args.temperature > 0.0

    responses = []
    for batched_prompts in tqdm(chunks(prompts, 1), total=math.ceil(len(prompts) / 1)):
        if "chatglm" in args.model_name:
            try:
                outputs, _ = model.chat(tokenizer, batched_prompts[0], history=[], max_length=16384)
                outputs = [outputs]
            except:
                outputs = [" "]
        else:
            inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                use_cache=True,
                return_dict_in_generate=False,
            )
        for i, generated_sequence in enumerate(outputs):
            if "chatglm" in args.model_name:
                responses.append(generated_sequence)
            else:
                input_ids = inputs["input_ids"][i]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    )
                new_text = text[prompt_length:]
                responses.append(new_text)

    with xopen(args.output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = args.model_name
            output_example["model_temperature"] = args.temperature
            output_example["model_top_p"] = args.top_p
            output_example["model_prompt_mention_random_ordering"] = args.prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = args.use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Model to use in generating responses", required=True)
    parser.add_argument("--input_path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top_p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--output_path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents).")
    parser.add_argument("--prompt_mention_random_ordering", action="store_true", help="Mention that search results are ordered randomly in the prompt")
    parser.add_argument("--use_random_ordering", action="store_true", help="Randomize the ordering of the distractors, rather than sorting by relevance.")
    parser.add_argument("--query_aware_contextualization", action="store_true", help="Place the question both before and after the documents.")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    # quantization config
    parser.add_argument("--w_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=16)
    parser.add_argument("--a_group_size", type=int, default=128)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_bit", type=int, default=16)
    args = parser.parse_args()

    print("* Running QA responses for {} model".format(args.model_name))
    main(args)
    print("* Finished running")
