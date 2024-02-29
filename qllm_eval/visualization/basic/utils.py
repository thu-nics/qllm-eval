import numpy as np

# dataset related properties
dataset_minimum = {
    'chid': 16.67,
    'winogrande': 50.,
    'race': 25.,
    'lambada': 0.,
    'rte': 50.,
    'piqa': 50.,
    'siqa': 33.33,
}

bitwidth_idx_mapping = {
    'w_only':   {'fp16': 0, 'w8': 1, 'w4': 2, 'w3': 3, 'w2': 4},
    'w_a':      {'fp16': 0, 'w8a8': 1, 'w4a8': 2, 'w4a4': 3},
    'kv_cache': {'fp16': 0, 'kv8': 1, 'kv4': 2, 'kv3': 3, 'kv2': 4}
}


# helper functions
def normalize_results(raw_results, fp_idx=0, minimal=None, range=1, w_fp=True):
    # raw_results = result_pad(raw_results)
    has_nonzero_fp_result = raw_results[fp_idx] is not None and raw_results[fp_idx] != 0
    if has_nonzero_fp_result and w_fp:
        # do not consider the minimal value of the dataset
        fp_result = raw_results[fp_idx]
        if minimal is None:
            norm_results = [i / fp_result if i is not None else None for i in raw_results]
            print('Result Normalization Succeeded.')
        else:
            norm_results = [max((i - minimal) / (fp_result - minimal), 0) \
                                if i is not None else None for i in raw_results]
    else:
        norm_results = raw_results
        print('The input results have no FP precision, return original results.')
    assert range in [1, 100]
    if range == 100:
        norm_results = [i * 100 if i is not None else i for i in norm_results]
        print('Result Normalization Succeeded.')
    elif range == 1 and norm_results == raw_results:
        norm_results = [i / 100. if i is not None else i for i in norm_results]
        print('Result Normalization Succeeded.')
    return norm_results
