# Evaluation with LLM_judge
Basic instructions on evaluating quantized LLMs with [LLM_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). 

> Note that this repository only contains necessary code required by our experiments, including code for quantized LLMs' dialogue generation and "single" mode GPT-4 judgement. Also be aware that the scripts are adapted from the original ones provided by [LLM_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge), so the script arguments are not exactly the same, please follow THIS document's instructions if any conflict exists. If you want to learn more, please refer to the original repository for [LLM_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

## Installation
Change directory to `QLLM-Evaluation` and conduct the following command to install our repository's requirements:	
```
cd path/to/QLLM-Evaluation
pip install -e .
pip install -r requirements.txt
```
You don't need to download the source code of FastChat, for the package `fschat` has been included in our `requirements.txt`.

## Evaluate a quantized model on MT-bench
### Generate model answers to MT-bench questions
Change directory to `QLLM-Evaluation/qllm_eval/evaluation/q_dialogue` and run `gen_model_answer.py`:
```
python gen_model_answer.py \
--model-path [MODEL-PATH] \
--model-id [MODEL-ID] \
[--quant] \
[--w_bit [W-BIT]] \
[--w_group_size [W-GROUP-SIZE]] \
[--a_bit [A-BIT]] \
[--a_group_size [A-GROUP-SIZE]] \
[--kv_bit [KV-BIT]] \
[--kv_group_size [KV-GROUP-SIZE]] \
[--rep_file [REP-FILE]] \
[--use_flash_attn]
```
The following arguments are required:
- `[MODEL-PATH]` is the path to the weights, which can be a local folder or a HuggingFace repo ID.
- `[MODEL-ID]` is a name you give to the model.

The following arguments are optional:
- `--quant` indicates whether you want to generate dialogues using a quantized model.
- `[W-BIT]`, `[A-BIT]`, `[KV-BIT]` are the quantization bit-width for weight, activation and kv-cache, all default to 16.
- `[W-GROUP-SIZE]`, `[A-GROUP-SIZE]`, `[KV-GROUP-SIZE]` are the group size for group-wise quantiztion, all default to 128.
- `[REP-FILE]` is the path to the cache file if you want to use AWQ or SmoothQuant.
- `--use_flash_attn` indicates whether you want to use flash-attention to save memory and speed up decoding.

The answers will be saved to: `./fschat_dat/mt_bench/model_answer/[MODEL-ID].jsonl`.

> Also, please note that the `--model_id` argument should match one of [FastChat's supported models](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md) to get the proper prompt template. The matching rule of each supported model could be found at [model_adapter.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py). For example, `Llama2Adapter`'s matching rule is defined as:
> ```python
> def match(self, model_path: str):
>      return "llama-2" in model_path.lower()
> ```
> Therefore, when evaluating llama-2 models, you should specify a `--model_id` argument containing the pattern "llama-2".
> 
> For unsupported models, you could either support your model according to [model_support.md](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md), or just use the default model adapter without any modification.

e.g.
```
python gen_model_answer.py \
--model-path meta-llama/Llama-2-7b-chat-hf \
--model-id llama-2-7b-chat_quant_w_4_a_4 \
--quant \
--w_bit 4 \
--a_bit 4 \
--use_flash_attn
```

### Generate GPT-4 judgements
We only support single-answer grading here. This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison. For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns. Please note that while the original script support passing more than one model_ids to the `--model-list` argument, we recommend passing only one model_id each time for clarity.
```
python gen_judgment.py \
--model-list [MODEL-ID] \
--save_name [MODEL-SAVENAME] \
[--parallel [PARALLEL]] 
```
The evaluation results will be saved to: `./fschat_dat/mt_bench/model_judgement/[MODEL-SAVENAME].jsonl`. Here we recommend setting `[MODEL-SAVENAME]` the same as `[MODEL-ID]` for simplicity.

e.g.
```
python gen_judgment.py \
--model-list llama-2-7b-chat_quant_w_4_a_4 \
--parallel 4 \
--save_name llama-2-7b-chat_quant_w_4_a_4
```

### Show MT-bench scores
You can show all the available results by simply running:
```
python show_result.py
```
If you want to show results of model_ids with a specified pattern, you can pass `--only_show` argument. For example, if you only want to see the results of llama models:
```
python show_result.py --only_show llama
```

### Results
We open-source our results in [this directory](https://github.com/LSY-noya/QLLM-Evaluation/tree/main/qllm_eval/visualization/dialogue/assets), corresponding to the data we present in our paper.
