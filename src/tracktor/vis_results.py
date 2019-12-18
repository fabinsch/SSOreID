import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_names = ["1", "2", "3", "4", "5", "6"]
metrics_to_plot = ["idf1", "mota", "num_misses", "num_false_positives", "num_switches"]
sequences = ["OVERALL"]

array = np.array([])
res = pd.DataFrame()

for f in file_names:
    df = pd.read_pickle("output/finetuning_results/all_results_{}.pkl".format(f))
    #for sequence in sequences:
    res = res.append(df.loc[sequences, metrics_to_plot])

for seq in sequences:
    for i, column_name in enumerate(res.keys()):
        plt.subplot(2, 3, i+1)
        plt.title("{}#{}".format(seq, column_name))
        plt.plot(file_names, res.loc[seq, column_name], marker='o', markerfacecolor="yellow", markersize=6, color='skyblue', linewidth=4)
    plt.show()
