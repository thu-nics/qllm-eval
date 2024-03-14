# Evaluation Long-Context Tasks
Basic instructions on evaluating quantized LLMs with [LongEval](https://github.com/DachengLi1/LongChat) and [Lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) benchmarks. You need to install the qllm_eval package first.

## For LongEval dataset

1. Generate new key-value retrieval test cases.
    ```
    cd qllm_eval/evaluation/q_long

    python longeval/generate_testcase.py longeval/generate_testcases_configs.yaml
    ```

2. Evaluation with `LongEval`.
    ```
    CUDA_VISIBLE_DEVICES=0 python main_longeval.py \
    --model-name-or-path /Your/LLM/Path --use_flash_attn \
    --task lines --test_dir new_cases \
    --w_group_size w_group_size --w_bit w_bit \
    --a_group_size a_group_size --a_bit a_bit \
    --kv_group_size kv_group_size --kv_bit kv_bit
    ```


## For Lost-in-the-middle dataset

1. Git clone [Lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) and install it locally in the qllm_eval conda environment. 
   ```
    conda activate qllm_eval
	git clone git@github.com:nelson-liu/lost-in-the-middle.git
	cd <lost-in-the-middle_path>
    Installation...
    ```

2. Evaluation with `lost-in-the-middle`.
    ```
    cd qllm_eval/evaluation/q_long

    CUDA_VISIBLE_DEVICES=0 python main_litm.py \
    --model_name /Your/LLM/Path --use_flash_attn \
    --w_group_size w_group_size --w_bit w_bit \
    --a_group_size a_group_size --a_bit a_bit \
    --kv_group_size kv_group_size --kv_bit kv_bit \
    --input_path <lost-in-the-middle_path>/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz \
    --max_new_tokens 100 --output_path /Your/Path/to/Results
    ```

    > The input data file is `<lost-in-the-middle_path>/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz`

    > The evaluation results can be found in `/Your/Path/to/Results`.
