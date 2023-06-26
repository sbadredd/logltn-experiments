import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob

def set_size(
        width_pt: float, 
        fraction: float = 1, 
        subplots: tuple[int,int] = (1, 1),
        adjust_bottom: float = .2
) -> tuple[float,float]:
    """Excerpt from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple[float,float]
            Dimensions of figure in inches
    """
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_in = (1/(1-adjust_bottom)) * fig_height_in
    return (fig_width_in, fig_height_in)

def set_tex_style() -> None:
    # Using seaborn's style
    plt.style.use('seaborn')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    plt.rcParams.update(tex_fonts)


if __name__ == "__main__":
    # gather baselines
    baselines_paths = [
        "results/1500/prod_rl",
        "results/1500/stable_product_p2",
        "results/1500/stable_product_p6",
        "results/1500/logltn_default",
        "results/1500/logltn_lseup",
        "results/1500/logltn_max",
        "results/1500/logltn_sum"
    ]
    baselines_titles = ["Prod RL", "Stable RL (p=2)", "Stable RL (p=6)",
                        "logLTN", "logLTN-up", "logLTN-max", "logLTN-sum"]
    baselines_dfs = []
    for (baseline_path, baseline_title) in zip(baselines_paths, baselines_titles):
        test_files = glob.glob(baseline_path+"*.csv")
        df = pd.concat([pd.read_csv(f) for f in test_files])
        df["baseline"] = baseline_title
        baselines_dfs.append(df)
    df = pd.concat(baselines_dfs)
    
    # plot measures
    plot_measures = ["train_accuracy","test_accuracy"]
    plot_titles = ["Train Accuracy", "Test Accuracy"]
    set_tex_style()
    width_pt = 505.89
    subplots = (1,2)
    adjust_bottom = .4
    fig, axes = plt.subplots(*subplots,figsize=set_size(width_pt, subplots=subplots,adjust_bottom=adjust_bottom))
    if len(plot_measures)%2 != 0:
        fig.delaxes(axes[-1,-1])
    for i,(measure,title) in enumerate(zip(plot_measures, plot_titles)):
        try:
            use_ax = axes[i//subplots[1],i%subplots[1]]
        except IndexError:
            use_ax = axes[i]
        print(title)
        for baseline in df["baseline"].unique():
            df_baseline = df[df["baseline"]==baseline]
            end_measures = df_baseline[measure][df_baseline["Epoch"]==df_baseline["Epoch"].max()]
            print(f"{baseline}: {end_measures.mean()*100} +- {end_measures.std()*100}" )
        lineplot = sns.lineplot(data=df, x="Epoch", y=measure, hue="baseline", ax=use_ax)
        lineplot.set_title(title)
        use_ax.set(ylabel=None)
        handles, labels = use_ax.get_legend_handles_labels()
        if (i//subplots[1])+1 != subplots[0]: # lowest ax only
            use_ax.set(xlabel=None)
    plt.tight_layout()
    fig.savefig('MNISTAdd.pdf', format='pdf', bbox_inches='tight')
    plt.show()