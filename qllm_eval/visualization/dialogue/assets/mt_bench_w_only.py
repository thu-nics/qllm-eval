# results for mt-bench
# the elements in each list are the results of round1 and round2
# the elements in each sub-list are the results of precision [[FP16, W8, W4, W3], [FP16, W8, W4, W3]],

results = {
    # llama2-chat series
    'llama-2_7b_chat' : [[5.31, 5.16, 5.47, 4.94], [4.14, 4.04, 4.30, 3.52]],
    'llama-2_13b_chat' : [[5.72, 5.95, 5.74, 5.38], [5.05, 5.31, 4.65, 4.26]],
    'llama-2_70b_chat' : [[6.26, 6.49, 5.91, 5.86], [5.99, 5.83, 5.55, 5.12]],

    # falcon-instruct series
    'falcon_7b_instruct' : [[3.79, 3.71, 3.39, 3.06], [2.30, 2.19, 2.27, 1.89]],
    'falcon_40b_instruct' : [[4.92, 4.81, 4.66, 4.38], [3.36, 3.01, 3.69, 3.54]],
    'falcon_180b_chat' : [[6.35, 6.62, 6.25, 5.79], [5.59, 5.70, 6.05, 5.17]],

    # mistral-instruct series
    'mistral_7b_instruct' : [[6.70, 6.78, 6.44, 6.18], [6.00, 6.01, 5.88, 5.49]],
    'mixtral_8x7b_instruct' : [[7.89, 7.53, 7.21, 6.72], [6.55, 6.64, 6.21, 5.53]],

    # chatglm3 series
    'chatglm3_6b' : [[5.13, 4.91, 5.06, 4.35], [3.68, 3.88, 4.09, 3.12]],

    # stablelm series
    'stablelm_zephyr_3b' : [[5.03, 5.09, 5.58, 3.15], [4.25, 3.86, 4.09, 2.11]],

    # gemma series
    'gemma_2b_it' : [[4.06, 4.18, 3.67, 3.39], [2.81, 3.11, 3.06, 2.60]],
    'gemma_7b_it' : [[5.25, 5.29, 4.94, 4.74], [3.62, 3.88, 3.51, 3.19]],

    # mamba series
    'mamba_2b8_chat' : [[1.95, 2.06, 1.73, 1.09], [1.40, 1.29, 1.29, 1.00]],
}
