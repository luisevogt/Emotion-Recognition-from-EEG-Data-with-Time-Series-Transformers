import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_chart(data: dict, fig_size: tuple, x: str, y: str, caption: str, save_folder: str):
    plt.figure(figsize=fig_size)

    bar_chart = sns.barplot(data=pd.DataFrame.from_dict(data),
                            x=x, y=y)
    bar_chart.set_title(caption)

    fig = bar_chart.get_figure()
    fig.savefig(save_folder)
