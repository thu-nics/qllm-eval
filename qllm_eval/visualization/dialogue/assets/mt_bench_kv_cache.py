# results for mt-bench
# the elements in each list are the results of round1 and round2
# the elements in each sub-list are the results of precision [[FP16, KV8, KV4, KV3], [FP16, KV8, KV4, KV3]]

results = {
    # llama2-chat series
    'llama-2_7b_chat' : [[5.31, 5.25, 5.46, 4.38], [4.14, 4.21, 4.16, 2.76]],
    'llama-2_13b_chat' : [[5.72, 5.84, 5.86, 5.53], [5.05, 5.04, 4.38, 3.85]],
    'llama-2_70b_chat' : [[6.26, 6.41, 6.30, 6.25], [5.99, 5.91, 6.03, 5.66]],

    # falcon-instruct series
    'falcon_7b_instruct' : [[3.79, 3.64, 3.60, 2.92], [2.30, 2.27, 2.24, 1.86]],
    'falcon_40b_instruct' : [[4.92, 4.86, 4.76, 4.45], [3.36, 3.35, 3.40, 3.34]],
    'falcon_180b_chat' : [[6.35, 6.68, 6.61, 6.58], [5.59, 5.63, 5.70, 5.46]],

    # mistral-instruct series
    'mistral_7b_instruct' : [[6.70, 6.70, 6.55, 6.26], [6.00, 6.42, 6.22, 4.96]],
    'mixtral_8x7b_instruct' : [[7.89, 7.62, 6.97, 6.26], [6.55, 6.94, 6.34, 4.96]],

    # chatglm3 series
    'chatglm3_6b' : [[5.13, 5.08, 4.58, 4.81], [3.68, 3.54, 3.26, 3.18]],

    # stablelm series
    'stablelm_zephyr_3b' : [[5.03, 5.08, 5.08, 4.38], [4.25, 4.14, 3.78, 3.19]],

    # gemma series
    'gemma_2b_it' : [[4.06, 3.98, 3.41, 3.34], [2.81, 2.62, 2.80, 2.52]],
    'gemma_7b_it' : [[5.25, 5.28, 4.98, 4.94], [3.61, 3.63, 3.49, 3.37]],
}
