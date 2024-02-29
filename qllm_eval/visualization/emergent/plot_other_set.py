import os
import matplotlib.pyplot as plt
import numpy as np
from importlib import import_module


# Considering we plot the figures with the GUI mode, the options below should be set mannually and locally in the
# script instead of being passed as command args.

####################################### Plot Settings #######################################
# saving path
save_path = "./pdf_test/"

# plot info
dataset_names = ["Ceval", "ARC-e", "Hellaswag"]
save_name = None  # e.g. 'test_plot'

MIN = -0.05
MAX = 1.05
# plot mode
plot_mode = "w"
assert plot_mode in ["w", "wa", "kv"]

acc = {}
for dataset in dataset_names:
    data_file_name = dataset.lower() + "_" + plot_mode + ".txt"
    acc[dataset] = []
    with open("./emergent/" + data_file_name, "r") as f:
        while True:
            txt = f.readline()
            if txt == "":
                break
            model_name = txt.split("\t")[0].replace("-", "_")
            if "32" in model_name:
                print("skip chatglm3-6b-32k")
                continue
            if "instruct" in model_name or "chat" in model_name:
                model_name = model_name.replace("_instruct", "")
                model_name = model_name.replace("_chat", "")
            acc0 = txt.split("\t")[1:]
            for i in range(1, len(acc0)):
                acc0[0] = float(acc0[0])
                if acc0[i].split("/")[-1] == "" or acc0[i].split("/")[-1] == "\n":
                    acc0[i] = None
                else:
                    acc0[i] = float(acc0[i])
                    acc0[i] = max((acc0[i] - 25) / (acc0[0] - 25), 0)
            acc0[0] = 1
            if acc[dataset] == []:
                acc[dataset] = acc0
            else:
                for j in range(len(acc[dataset])):
                    acc[dataset][j] += acc0[j]

for i in range(len(acc["ARC-e"])):
    acc["ARC-e"][i] = acc["ARC-e"][i] / 9

for i in range(len(acc["Hellaswag"])):
    acc["Hellaswag"][i] = acc["Hellaswag"][i] / 9


# label names
x_label = "Bit-width"
y_label = "Normalized Accuracy"

# legend location
legend_loc = "lower left"

# curve properties
curve_colors = []
line_style = ["-", "--", "-.", ":"]
dot_style = [".", "*", "^", ","]
MARKERS = ["o", "^", "s", "d", "+"]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]


# selected representative instances
# individuals_to_plot = {
#     'opt':      ['OPT_2B7', 'OPT_66B'],
#     'falcon':   ['Falcon_7B', 'Falcon_40B'],
#     'llama2':   ['LlaMA2_7B', 'LlaMA2_70B'],
#     'bloom':    ['Bloom_3B', 'Bloom_175B'],
#     'bloomz':   ['Bloomz_3B', 'Bloomz_175B'],
#     'chatglm3': ['ChatGLM3_6B'],
# }


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
        print("Result Normalization Succeeded.")
    else:
        norm_results = raw_results
        print("The input results have no FP precision, return original results.")
    return norm_results


###################################### Plot Functions ######################################
if __name__ == "__main__":
    # create a folder to save the resulting plot.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # plot name
    save_name = save_name + ".pdf" if save_name is not None else plot_mode + ".pdf"
    save_name = save_path + save_name

    # init canvas
    fig, ax = plt.subplots(figsize=[8.0, 6.0])

    # make the plot compact
    plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.11)

    # set figure labels
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    # set axes font size
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    x_axis = {
        "w": ["FP16", "W8", "W4", "W3", "W2"],
        "wa": ["FP16", "W8A8", "W4A8", "W4A4"],
        "kv": ["FP16", "KV8", "KV4", "KV3", "KV2"],
    }[plot_mode]

    # plot the curves
    model_families = ["opt", "falcon", "llama2", "bloom", "bloomz", "chatglm3"]
    # for i, model_family in enumerate(model_families):
    #     if len(curve_colors) > 0:
    #         curve_color = curve_colors[i] # specify the color you want to use
    #     else:
    #         curve_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i] # or use the default colors
    #     individuals = individuals_to_plot[model_family]
    #     for j, individual in enumerate(individuals):
    #         results_module = import_module('assets.' + dataset_name + '_' + plot_mode)
    #         individual_results = getattr(results_module, individual)
    #         ax.plot(x_axis, individual_results, marker=dot_style[j], label=individual.replace('_', '-'), \
    #                 linestyle=line_style[j], color=curve_color)
    print(acc)
    for i, model_name in enumerate(dataset_names):
        # y_value = eval("data.{}".format(model_name))
        ax.plot(x_axis, acc[model_name], marker=MARKERS[0], markersize=16, label=dataset_names[i], linestyle="-", color=COLORS[i])

    # set legend location
    # ax.legend(loc=legend_loc)
    ax.legend(fontsize=20, loc="lower left")

    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.ylim(MIN, MAX)

    # Warning: please do not change the figure you are previewing
    # preview the figure
    plt.savefig(save_name)
    plt.show()
