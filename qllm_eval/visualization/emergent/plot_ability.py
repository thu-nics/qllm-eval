import matplotlib.pyplot as plt
import numpy as np

modes = ["w", "wa", "kv"]
model_names = ["Falcon-180B"]
datasets = ["mmlu", "arc-c", "arc-e", "hellaswag", "gsm8k", "strategyqa", "mmlu-calib"]
for mode in modes:
    for model_name in model_names:
        acc = {}
        minimum = {"mmlu": 25, "arc-c": 25, "arc-e": 25, "hellaswag": 25, "gsm8k": 0, "strategyqa": 0, "mmlu-calib": 50}
        for dataset in datasets:
            data_file_name = dataset + "_" + mode + ".txt"
            with open("./emergent/" + data_file_name, "r") as f:
                while True:
                    txt = f.readline()
                    if txt == "":
                        break
                    fmodel_name = txt.split("\t")[0]
                    if "32k" in fmodel_name:
                        continue
                    if "instruct" in fmodel_name or "chat" in fmodel_name:
                        fmodel_name = fmodel_name.replace("-instruct", "")
                        fmodel_name = fmodel_name.replace("-chat", "")
                    if fmodel_name != model_name:
                        continue
                    acc0 = txt.split("\t")[1:]
                    for i in range(len(acc0)):
                        if acc0[i].split("/")[-1] == "" or acc0[i].split("/")[-1] == "\n":
                            acc0[i] = None
                        else:
                            acc0[i] = float(acc0[i].split("/")[-1])
                            if i == 0:
                                std = acc0[i]
                            acc0[i] = max((acc0[i] - minimum[dataset]) / (std - minimum[dataset]), 0)
                    acc[dataset] = acc0
        ability = {}
        # print(acc)
        if mode == "w":
            x_label = ["FP16", "W8", "W4", "W3", "W2"]
        elif mode == "wa":
            x_label = ["FP16", "W8A8", "W4A8", "W4A4"]
        else:
            x_label = ["FP16", "KV8", "KV4", "KV3", "KV2"]

        for i in range(len(x_label)):
            ability[x_label[i]] = []
            ability[x_label[i]].append(acc["mmlu"][i])
            ability[x_label[i]].append(acc["strategyqa"][i])
            ability[x_label[i]].append(acc["gsm8k"][i])
            ability[x_label[i]].append(acc["arc-c"][i])
            ability[x_label[i]].append(acc["mmlu-calib"][i])
            ability[x_label[i]].append(acc["mmlu"][i])
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        angles += np.pi / 10
        feature = ["ICL", "C-MR", "M-MR", "IF", "SC", "ICL"]
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        for k, v in ability.items():
            ax.plot(angles, v, "o-", linewidth=2, label=k)
            ax.fill(angles, v, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, feature, size=15)
        plt.legend(loc="upper right", fontsize=13)
        plt.savefig("./figure/radarpdf/" + model_name + "_" + mode + ".pdf", bbox_inches="tight")
        plt.close()
