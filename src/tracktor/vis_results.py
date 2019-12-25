import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_names = ["all_results_baseline", "results_1_0.15_128_1e-4_4", "results_2_0.1_128_1e-4_4",
              "results_3_0.12_128_1e-4_4", "results_4_0.08_128_1e-4_4", "results_5_0.1_128_5e-5_4",
              "results_6_0.1_16_1e-4_4"]
metrics_to_plot = ["idf1", "mota", "num_misses", "num_false_positives", "num_switches"]
sequences = ["OVERALL", "MOT17-04-FRCNN"]

array = np.array([])
res = pd.DataFrame()
for f in file_names:
    df = pd.read_pickle("../../output/finetuning_results/{}.pkl".format(f))
    #for sequence in sequences:
    res = res.append(df.loc[sequences, metrics_to_plot])

index_baseline = 0

for seq_index, seq in enumerate(sequences):
    for i, column_name in enumerate(res.keys()):
        plt.subplot(3, 2, i + 1)
        print(res.loc[seq, column_name])
        plt.title("{}#{}".format(seq[6:8] if "MOT" in seq else seq, column_name))
        plt.plot(list(range(1, len(file_names))), res.loc[seq, column_name][1:], marker='o', markerfacecolor="yellow", markersize=6, color='skyblue', linewidth=4)
        plt.axhline(y=res.loc[seq, column_name].iloc[index_baseline], color='r', linestyle='-')
    plt.show()
