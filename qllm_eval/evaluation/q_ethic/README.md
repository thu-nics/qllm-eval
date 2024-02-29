# Evaluation with ETHICS benchmark
Basic instructions on evaluating quantized LLMs with ETHICS benchmark. 

## Run ETHICS benchmark
We implement an end-to-end evaluation flow for `ETHICS` benchmark. You can use the following command below to evaluate four subdatasets, including `commonsense`, `deontology`, `justice`, and `virtue`.

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path /Your/LLM/Path --output_dir /Your/Path/to/Results \
--w_group_size w_group_size --w_bit w_bit \
--a_group_size a_group_size --a_bit a_bit \
--kv_group_size kv_group_size --kv_bit kv_bit
```

> The evaluation results can be found in `/Your/Path/to/Results`.

> Here, the main.py script is in `qllm_eval/evaluation/q_ethic`.
