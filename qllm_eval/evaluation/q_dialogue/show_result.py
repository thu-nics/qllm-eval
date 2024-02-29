"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def merge():
    read_path = './fschat_dat/mt_bench/model_judgment/'
    output_file = './fschat_dat/mt_bench/model_judgment/merged.jsonl'  
    if os.path.exists(output_file): # remove the old one
        os.remove(output_file)

    with open(output_file, mode='w') as writer:
        file_paths_0 = os.listdir(read_path)
        file_paths = [os.path.join(read_path, x) for x in file_paths_0 if x.endswith('.jsonl')]
        
        for file_path in tqdm(file_paths):
            with open(file_path, 'r') as reader:
                items = reader.readlines()
                for item in items:
                    writer.write(item)

sr_rd1: pd.Series
sr_rd2: pd.Series
def draw(sr_rd1, sr_rd2, mode='kv', save_path='./plot_results', all_in_one=True):
    """
    mode: ['w', 'wa', 'kv']
    if all_in_one=True, the result of w/wa/kv would be plotted in one single graph
    """
    model_families = {
        'vicuna-v1.5': ['vicuna-7b-v1.5', 'vicuna-7b-v1.5-16k', 'vicuna-13b-v1.5', 'vicuna-13b-v1.5-16k'],
        'llama2-chat': ['llama2-7b-chat', 'llama2-13b-chat', 'llama2-70b-chat'],
        'chatglm3': ['chatglm3-6b', 'chatglm3-6b-32k'],
        'longchat': ['longchat-7b-16k', 'longchat-7b-v1.5-32k', 'longchat-13b-16k'],
        'falcon-instruct': ['falcon-7b-instruct', 'falcon-40b-instruct'],
        'mistral-mixtral': ['mistral-7B-instruct-v0.2', 'mixtral-8x7B-instruct-v0.1']
    }
    bit_widths = {
        'w': [8,4,3],       # some +2
        'wa': [(8,8), (4,8)], # some +(4,4)
        'kv': [8,4,3]   # some +2
    }
    models_available = [x[0] for x in list(sr_rd1.index)]
    for model_family in model_families.keys():
        models = model_families[model_family]
        if not all_in_one:  # each mode in separate graph
            fig, axs = plt.subplots(1, len(model_families[model_family]), figsize=(15, 5))
            if len(model_families[model_family]) == 1:
                axs = [axs]
            for i, model in enumerate(models):
                if mode == 'kv' or mode == 'w':
                    axs[i].set_title(model)
                    axs[i].set_xlabel('bit_width')
                    axs[i].set_ylabel('score')
                    axs[i].set_ylim(0, 10)
                    axs[i].set_xticks([x for x in range(len(bit_widths[mode])+3)], labels=['fp16']+bit_widths[mode]+[2,1])
                    # for 2bit:
                    # if there's a judgement for 2bits, take the judgement score, else assign 1
                    if model+f'_quant_{mode}_2' in models_available:
                        score_2bit = (sr_rd1[model+f'_quant_{mode}_2'][1], sr_rd2[model+f'_quant_{mode}_2'][2])
                    else: 
                        score_2bit = (1,1)
                    # for 1bit:
                    # all assign 1
                    score_1bit = (1,1)
                    
                    # round1
                    axs[i].plot([j for j in range(len(bit_widths[mode])+3)], [sr_rd1[model][1]] + [sr_rd1[model+f'_quant_{mode}_'+str(bit_width)][1] for bit_width in bit_widths[mode]] + [score_2bit[0],score_1bit[0]], label='round 1')
                    # round2
                    axs[i].plot([j for j in range(len(bit_widths[mode])+3)], [sr_rd2[model][2]] + [sr_rd2[model+f'_quant_{mode}_'+str(bit_width)][2] for bit_width in bit_widths[mode]] + [score_2bit[1],score_1bit[1]], label='round 2')
                    axs[i].legend()
                elif mode == 'wa':
                    axs[i].set_title(model)
                    axs[i].set_xlabel('bit_width')
                    axs[i].set_ylabel('score')
                    axs[i].set_ylim(0, 10)
                    axs[i].set_xticks([x for x in range(len(bit_widths[mode])+2)], labels=['fp16']+[f'w{wbit}a{abit}' for wbit,abit in bit_widths[mode]]+['w4a4'])     
                    if model+f'_quant_w_4_a_4' in models_available:
                        score_w4a4 = (sr_rd1[model+f'_quant_w_4_a_4'][1], sr_rd2[model+f'_quant_w_4_a_4'][2])       
                    else:
                        score_w4a4 = (1,1)
                    # round1
                    axs[i].plot([j for j in range(len(bit_widths[mode])+2)], [sr_rd1[model][1]] + [sr_rd1[model+f'_quant_w_{wbit}_a_{abit}'][1] for wbit,abit in bit_widths[mode]] + [score_w4a4[0]], label='round 1')
                    # round2
                    axs[i].plot([j for j in range(len(bit_widths[mode])+2)], [sr_rd2[model][2]] + [sr_rd2[model+f'_quant_w_{wbit}_a_{abit}'][2] for wbit,abit in bit_widths[mode]] + [score_w4a4[1]], label='round 2')
                    axs[i].legend()             
            plt.suptitle(model_family+' ~ '+mode)
            plt.tight_layout()
            save_name = os.path.join(save_path, model_family+f'_{mode}.jpg')
            plt.savefig(save_name)
            print(f"results saved to {save_name}")

        else:   # kv, w, wa in one graph
            fig, axs = plt.subplots(3, len(model_families[model_family]), figsize=(15, 15))
            if len(model_families[model_family]) == 1:
                axs = [[ax] for ax in axs] 
            # print(axs)
            for i, model in enumerate(models):
                # kv, w
                for j, mode in enumerate(['kv', 'w']):
                    axs[j][i].set_title(model+' ~ '+mode)
                    axs[j][i].set_xlabel('bit_width')
                    axs[j][i].set_ylabel('score')
                    axs[j][i].set_ylim(0, 10)
                    axs[j][i].set_xticks([x for x in range(len(bit_widths[mode])+3)], labels=['fp16']+bit_widths[mode]+[2,1])
                    # for 2bit:
                    # if there's a judgement for 2bits, take the judgement score, else assign 1
                    if model+f'_quant_{mode}_2' in models_available:
                        score_2bit = (sr_rd1[model+f'_quant_{mode}_2'][1], sr_rd2[model+f'_quant_{mode}_2'][2])
                    else: 
                        score_2bit = (1,1)
                    # for 1bit:
                    # all assign 1
                    score_1bit = (1,1)

                    # round1
                    axs[j][i].plot([j for j in range(len(bit_widths[mode])+3)], [sr_rd1[model][1]] + [sr_rd1[model+f'_quant_{mode}_'+str(bit_width)][1] for bit_width in bit_widths[mode]] + [score_2bit[0],score_1bit[0]], label='round 1')
                    # round2
                    axs[j][i].plot([j for j in range(len(bit_widths[mode])+3)], [sr_rd2[model][2]] + [sr_rd2[model+f'_quant_{mode}_'+str(bit_width)][2] for bit_width in bit_widths[mode]] + [score_2bit[1],score_1bit[1]], label='round 2')
                    axs[j][i].legend()
                
                # wa
                mode = 'wa'
                j = 2
                axs[j][i].set_title(model+' ~ '+mode)
                axs[j][i].set_xlabel('bit_width')
                axs[j][i].set_ylabel('score')
                axs[j][i].set_ylim(0, 10)
                axs[j][i].set_xticks([x for x in range(len(bit_widths[mode])+2)], labels=['fp16']+[f'w{wbit}a{abit}' for wbit,abit in bit_widths[mode]]+['w4a4'])
                if model+f'_quant_w_4_a_4' in models_available:
                    score_w4a4 = (sr_rd1[model+f'_quant_w_4_a_4'][1], sr_rd2[model+f'_quant_w_4_a_4'][2])       
                else:
                    score_w4a4 = (1,1)
                # round1
                axs[j][i].plot([j for j in range(len(bit_widths[mode])+2)], [sr_rd1[model][1]] + [sr_rd1[model+f'_quant_w_{wbit}_a_{abit}'][1] for wbit,abit in bit_widths[mode]] + [score_w4a4[0]], label='round 1')
                # round2
                axs[j][i].plot([j for j in range(len(bit_widths[mode])+2)], [sr_rd2[model][2]] + [sr_rd2[model+f'_quant_w_{wbit}_a_{abit}'][2] for wbit,abit in bit_widths[mode]] + [score_w4a4[1]], label='round 2')
                axs[j][i].legend()  

            plt.suptitle(model_family+' ~ all')
            plt.tight_layout()
            save_name = os.path.join(save_path, model_family+f'_all.jpg')
            plt.savefig(save_name)
            print(f"results saved to {save_name}")        

def display_result_single(args):
    global sr_rd1
    global sr_rd2
    if args.input_file is None:
        input_file = (
            f"fschat_dat/{args.bench_name}/model_judgment/merged.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    df_1_sorted = df_1.sort_values(by="score", ascending=False)
    if not args.only_show:
        print(df_1_sorted)
    else:
        # only show the result for a specified model
        print(df_1_sorted['score'][df_1_sorted['score'].index.get_level_values('model').str.contains(args.only_show)])
    sr_rd1 = df_1['score']

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        df_2_sorted = df_2.sort_values(by="score", ascending=False)
        if not args.only_show:
            print(df_2_sorted)
        else:
            # only show the result for a specified model
            print(df_2_sorted['score'][df_2_sorted['score'].index.get_level_values('model').str.contains(args.only_show)])
        sr_rd2 = df_2['score']

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        df_3_sorted = df_3.sort_values(by="score", ascending=False)
        if not args.only_show:
            print(df_3_sorted)

def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"fschat_dat/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument('--only_show', default=None, help='Only show the results of a specified pattern')
    parser.add_argument('--draw_results', action='store_true')

    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")

    print('merging ...')
    merge()

    display_result_func(args)

    if args.draw_results:
        print('drawing ...')
        draw(sr_rd1, sr_rd2)
