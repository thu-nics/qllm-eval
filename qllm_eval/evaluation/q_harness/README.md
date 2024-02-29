# Evaluation with lm_evaluation_harness
Basic instructions on evaluating quantized LLMs with [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness). 

## Installation
Install the lm_eval package:
```
pip install lm_eval==0.3.0
```
The package `lm_eval` has been included in our `requirements.txt`.


## Use lm_eval for evaluation
Use the main.py in `qllm_eval/evaluation/q_harness` as an example, you can use the `--tasks A,B,C` to select different tasks for evaluation, where the `A,B,C` represents three different tasks in `lm_eval`.

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path /Your/LLM/Path --tasks A,B,C \
--w_group_size w_group_size --w_bit w_bit \
--a_group_size a_group_size --a_bit a_bit \
--kv_group_size kv_group_size --kv_bit kv_bit
```

> The tasks supported by `lm_eval` can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).

> In our paper, we use `--tasks truthfulqa_mc` to evaluate quantized LLMs on the `TruthfulQA` dataset.

