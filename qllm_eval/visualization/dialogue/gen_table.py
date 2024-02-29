import os
from assets import w_only, wa, kv

model_name_map = {
    'llama-2_7b_chat': 'LLaMA2-7B-chat',
    'llama-2_13b_chat': 'LLaMA2-13B-chat',
    'llama-2_70b_chat': 'LLaMA2-70B-chat',
    'falcon_7b_instruct': 'Falcon-7B-instruct',
    'falcon_40b_instruct': 'Falcon-40B-instruct',
    'falcon_180b_chat': 'Falcon-180B-chat',
    'mistral_7b_instruct': 'Mistral-7B-instruct-v0.2',
    'mixtral_8x7b_instruct': 'Mixtral-8x7B-instruct-v0.1',
    'chatglm3_6b': 'ChatGLM3-6B',
    'stablelm_zephyr_3b': 'StableLM-Zephyr-3B',
    'gemma_2b_it': 'Gemma-2B-it',
    'gemma_7b_it': 'Gemma-7B-it',
    'mamba_2b8_chat': "Mamba-2.8B-chat"
} 

def gen_tex_table_lines(save_path='./tables/table.tex'):
    """
        \midrule
        \multirow{2}{*}{MODEL-NAME} & 1 & rd1_fp16 & rd1_w8 & rd1_w4 & rd1_w3_awq & rd1_w3_awq & rd1_w8a8 & rd1_w4a8 & rd1_w4a4 & rd1_w4a4_sq & rd1_kv8 & rd1_kv4 & rd1_kv3 \\
        & 2 & rd2_fp16 & rd2_w8 & rd2_w4 & rd2_w3_awq & rd2_w3_awq & rd2_w8a8 & rd2_w4a8 & rd2_w4a4 & rd2_w4a4_sq & rd2_kv8 & rd2_kv4 & rd2_kv3 \\ 
    """
    # clear the previous table
    if os.path.exists(save_path):
        os.remove(save_path)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    with open(save_path, 'a') as f:
        for model, model_name in model_name_map.items():
            f.write(r'\midrule' + '\n') 
            if 'mamba' in model.lower():    # mamba doesn't have kv cache
                # round 1
                f.write(r'\multirow{2}{*}{' + model_name + r'} & 1 ' + \
                        ''.join([f'& {w_only.results[model][0][i]:.2f} ' for i in range(len(w_only.results[model][0]))]) + '& - ' + \
                        ''.join([f'& {wa.results[model][0][i]:.2f} ' for i in range(1, len(wa.results[model][0]))]) + '& - ' + \
                        ''.join([f'& - ' for _ in range(3)]) + \
                        r'\\' + '\n')
                # round2
                f.write(r'& 2 ' + \
                        ''.join([f'& {w_only.results[model][1][i]:.2f} ' for i in range(len(w_only.results[model][1]))]) + '& - ' + \
                        ''.join([f'& {wa.results[model][1][i]:.2f} ' for i in range(1, len(wa.results[model][1]))]) + '& - ' + \
                        ''.join([f'& - ' for _ in range(3)]) + \
                        r'\\' + '\n')   
            else:
                # round1
                f.write(r'\multirow{2}{*}{' + model_name + r'} & 1 ' + \
                        ''.join([f'& {w_only.results[model][0][i]:.2f} ' for i in range(len(w_only.results[model][0]))]) + '& - ' + \
                        ''.join([f'& {wa.results[model][0][i]:.2f} ' for i in range(1, len(wa.results[model][0]))]) + '& - ' + \
                        ''.join([f'& {kv.results[model][0][i]:.2f} ' for i in range(1, len(kv.results[model][0]))]) + \
                        r'\\' + '\n')
                # round2
                f.write(r'& 2 ' + \
                        ''.join([f'& {w_only.results[model][1][i]:.2f} ' for i in range(len(w_only.results[model][1]))]) + '& - ' + \
                        ''.join([f'& {wa.results[model][1][i]:.2f} ' for i in range(1, len(wa.results[model][1]))]) + '& - ' + \
                        ''.join([f'& {kv.results[model][1][i]:.2f} ' for i in range(1, len(kv.results[model][1]))]) + \
                        r'\\' + '\n')                   

def gen_md_table_lines(save_path='./tables/table.md'):
    pass

if __name__ == '__main__':
    gen_tex_table_lines()