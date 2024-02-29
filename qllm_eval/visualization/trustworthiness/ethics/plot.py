import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import importlib
import numpy as np

# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./figures/')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--plot_mode', type=str, default='kv', choices=['w', 'wa', 'kv'])
parser.add_argument('--dataset_name', type=str, default='Multi-Doc_QA')
parser.add_argument('--norm', action='store_true')
parser.add_argument("--num_bins", type=int, default=30, help="bin number")
parser.add_argument('--x_label', type=str, default='Bit-width')
parser.add_argument('--y_label', type=str, default='Accuracy')
parser.add_argument('--legend_loc', type=str, default='lower left')
args = parser.parse_args()


##################################### Helper Functions ######################################
def result_pad(raw_results, fp_idx=0):
    raw_fp_result = raw_results[fp_idx]
    new_results = [x if x is not None else 0 for x in raw_results]
    new_results[fp_idx] = raw_fp_result
    return new_results

def normalize_results(raw_results, fp_idx=0):
    # raw_results = result_pad(raw_results)
    if raw_results[fp_idx] is not None and raw_results[fp_idx] != 0:
        norm_results = [i / raw_results[fp_idx] * 100 if i is not None else None for i in raw_results]
        print('Result Normalization Succeeded.')
    else:
        norm_results = raw_results
        print('The input results have no FP precision, return original results.')
    return norm_results

MARKERS = ['o', '^', 's', 'd', '+']
COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
MODELS = ["Mistral_7B", "Mixtral_8x7B", "LLaMA2_7B", "LLaMA2_70B"]
TASKS = ['commonsense', 'deontology', 'justice', 'virtue']
MIN = [0.2, 0.2, 0.5, 0.6]
MAX = [0.81, 0.71, 0.8, 0.95]

###################################### Plot Functions ######################################
if __name__ == '__main__':
    # create a folder to save the resulting plot.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # plot different tasks
    for t_i, task in enumerate(TASKS):
        # plot name
        save_name = args.save_name + '.pdf' if args.save_name is not None else \
                        task + '_' + args.plot_mode + '_' + args.x_label + '.pdf'
        save_name = args.save_path + save_name

        # import data
        data = importlib.import_module("ethics.{}.{}_result".format(task, args.plot_mode))
        
        # init canvas
        fig, ax = plt.subplots(figsize=[8., 6.])

        # make the plot compact
        plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.11)

        # set figure labels
        plt.xlabel(args.x_label, fontsize=20)
        plt.ylabel(args.y_label if not args.norm else 'Normalized ' + args.y_label, fontsize=20)
        plt.ylim(MIN[t_i], MAX[t_i])
        plt.yticks(np.arange(MIN[t_i], MAX[t_i], 0.1))

        # set axes font size
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        x_axis = {
            'w':   ['FP16', 'W8', 'W4', 'W3'],
            'wa':    ['FP16', 'W8A8', 'W4A8'],
            'kv': ['FP16', 'KV8', 'KV4', 'KV3'],
        }[args.plot_mode]

        for i, model_name in enumerate(MODELS):
            y_value = eval("data.{}".format(model_name))
            ax.plot(x_axis, y_value, marker=MARKERS[0], markersize=16, label=MODELS[i], linestyle="-", color=COLORS[i])
            # ax.plot(x_axis[:len(y_value)], y_value, marker=MARKERS[j], markersize=13, linestyle="-", color=COLORS[i])

        # set legend location
        # ax.legend(loc=args.legend_loc, fontsize=20)
        ax.legend(fontsize=20)

        # legend_elements = []
        # for i, _ in enumerate(MODELS):
        #     legend_elements.append(Line2D([0], [0], marker=MARKERS[i], color='w', label=mode[i], markerfacecolor='tab:brown', markersize=16))
        # ax.legend(handles=legend_elements, fontsize=20)

        plt.savefig(save_name)
