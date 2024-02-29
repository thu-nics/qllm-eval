# results for mt-bench
# the elements in each list are the results of round1 and round2
# the elements in each sub-list are the results of precision [[FP16, W8A8, W4A8, W4A4], [FP16, W8A8, W4A8, W4A4]]

results = {
    # llama2-chat series
    'llama-2_7b_chat' : [[5.31, 4.94, 5.12, 1.00], [4.14, 3.79, 4.22, 1.00]],
    'llama-2_13b_chat' : [[5.72, 5.83, 5.88, 1.00], [5.05, 5.12, 4.97, 1.00]],
    'llama-2_70b_chat' : [[6.26, 6.17, 6.11, 1.00], [5.99, 5.86, 5.86, 1.00]],

    # falcon-instruct series
    'falcon_7b_instruct' : [[3.79, 3.59, 3.15, 1.00], [2.30, 2.14, 2.05, 1.00]],
    'falcon_40b_instruct' : [[4.92, 4.71, 4.40, 1.00], [3.36, 3.55, 3.14, 1.00]],
    'falcon_180b_chat' : [[6.35, 6.35, 6.56, 1.00], [5.59, 5.43, 5.88, 1.00]],

    # mistral-instruct series
    'mistral_7b_instruct' : [[6.70, 6.74, 6.53, 1.00], [6.00, 6.08, 5.76, 1.00]],
    'mixtral_8x7b_instruct' : [[7.89, 7.22, 7.42, 1.00], [6.55, 6.61, 6.49, 1.00]],

    # chatglm3 series
    'chatglm3_6b' : [[5.13, 5.14, 4.85, 1.00], [3.68, 3.92, 3.50, 1.00]],

    # stablelm series
    'stablelm_zephyr_3b' : [[5.03, 5.29, 5.48, 1.14], [4.25, 4.10, 3.94, 1.00]],

    # gemma series
    'gemma_2b_it' : [[4.06, 3.91, 3.64, 1.00], [2.81, 2.95, 2.73, 1.00]],
    'gemma_7b_it' : [[5.25, 5.09, 5.14, 1.00], [3.62, 3.79, 3.28, 1.00]],

    # mamba series
    'mamba_2b8_chat' : [[1.95, 2.08, 1.83, 1.41], [1.40, 1.45, 1.12, 1.00]],
}

