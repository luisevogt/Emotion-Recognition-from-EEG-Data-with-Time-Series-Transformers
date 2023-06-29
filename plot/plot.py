import pickle

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bar_chart(data: dict, fig_size: tuple, x: str, y: str, caption: str, save_folder: str):
    plt.figure(figsize=fig_size)

    bar_chart = sns.barplot(data=pd.DataFrame.from_dict(data),
                            x=x, y=y)
    bar_chart.set_title(caption)

    fig = bar_chart.get_figure()
    fig.savefig(save_folder)


if __name__ == '__main__':
    class_names = {0: 'negative',
                   1: 'neutral',
                   2: 'positive'}

    path = Path('../datasets/SEED_EEG/targets_seed_size_6.pkl')

    targets = pickle.load(open(path, 'rb'))

    counts = {v: [targets.count(k) / len(targets)] for k, v in class_names.items()}
    print(counts)

    plot_bar_chart(counts, (12, 7), x='class names', y='%', caption='Class Distribution in SEED',
                   save_folder='plot/plots')


