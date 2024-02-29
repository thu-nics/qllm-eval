# Evaluation with OpenCompass
Basic instructions on evaluating quantized LLMs with [OpenCompass](https://github.com/open-compass/opencompass). You need to install the qllm_eval package first.


## Installation

1. Git clone [OpenCompass](https://github.com/open-compass/opencompass) and install it locally in the qllm_eval conda environment. See [requirements of OpenCompass](https://github.com/open-compass/opencompass/blob/main/requirements.txt).
   ```
    conda activate qllm
	git clone git@github.com:open-compass/opencompass.git
	cd <opencompass_path>
    ```  

2. Install the required packages from the source.

   ```
   pip install -e .
   ```

3. Note that LlaMA should be installed mannually. Take the following steps to ensure LlaMA works properly:

	```
    git clone https://github.com/facebookresearch/llama.git
	cd <llama_path>
	pip install -r requirements.txt
	pip install -e .
	```

## Evaluation

1. Prepare datasets. Change directory to `QLLM-Evaluation/qllm_eval/evaluation/q_opencompass/` and create a new folder:

	```
	cd qllm_eval/evaluation/q_opencompass
	mkdir data
	cd data
	```

	Run the following commands to download and place the datasets in the `./qllm_eval/evaluation/q_opencompass/data` directory can complete dataset preparation.

	```
	# Run in the OpenCompass directory
	wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
	unzip OpenCompassData-core-20231110.zip
	```

	You may also use the pre-downloaded zip file, which is located at `/share/datasets/public_datasets/`.  

2. Run the following demo command to evaluate `OPT-125m` with weights quantized to 8-bit on `SuperGLUE_BoolQ_ppl` dataset:

	```
	cd qllm_eval/evaluation/q_opencompass
	CUDA_VISIBLE_DEVICES=0 python main.py --models hf_opt_125m --datasets SuperGLUE_BoolQ_ppl --work-dir ./outputs/debug/api_test --w_bit 8
	```

3. If you want to evaluate models with different quantization settings, please modify `./qllm_eval/evaluation/q_opencompass/utils/build.py`. If you want to support new datasets and new models, please add their configs to `./qllm_eval/evaluation/q_opencompass/configs`, whose original configs may be found at opencompass repo.

	* Specially, if you want to evaluate the models with kv cache quantized, please modify the imported model class in the model configuration file. We provide class `HuggingFaceCausalLM_` for this specific need.

	```python
	from qllm_eval.evaluation.q_opencompass.utils.models import HuggingFaceCausalLM_
	```

## Reference Table for Evaluation Failure Cases

From time to time we get upset evalution results from opencompass. Hopefully this table can help you solve the problem quickly.

1. Evaluation failure due to unparsed model outputs.

	When you evaluate one quantized model with a generation task, the model might output paired curly brace characters, which will be loaded as a dict variable, causing errors in the following string processing. In this case, you could modify the local opencompass package to avoid this:

	```
	opencompass/opencompass/tasks/openicl_eval.py
	```

	Adding `try-except` for exception processing might be helpful.
