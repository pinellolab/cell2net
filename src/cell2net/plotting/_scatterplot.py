import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def tf_activity_variance(
    df: pd.DataFrame,
    n_labels: int = 5,
    label_color: str = "red",
    frameon: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot TF regulation variance across cell types

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe where each row is a TF, and each column is a cell type
    """
    # copy the dataframe
    df_act = df.copy()

    df_act["var"] = df_act.var(axis=1)
    df_act = df_act.sort_values(by="var", ascending=True)

    df_act["tf"] = df_act.index
    # Map strings to evenly spaced numbers
    df_act["rank"] = range(len(df_act))

    # Identify the top 3 categories by value
    labels = df_act.nlargest(n_labels, "var")

    # plot TF variances
    fig, ax = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df_act, x="rank", y="var", ax=ax)

    # Add text annotations for the top 3 categories
    for _, row in labels.iterrows():
        ax.text(
            row["rank"],
            row["var"] + 0.5,  # Slightly offset the label above the point
            row["tf"],
            fontsize=10,
            color=label_color,
            ha="center",
        )

    ax.set_xlabel("Ranked TFs", fontsize=12)
    ax.set_ylabel("Variance of regulation activity", fontsize=12)

    plt.show()


def n_peaks_per_gene(
    df: pd.DataFrame,
    n_labels: int = 5,
    label_color: str = "red",
    frameon: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot number of associated peaks per gene

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe where each row is a TF, and each column is a cell type
    """
    # copy the dataframe
    df = df[["peak", "gene"]].drop_duplicates()
    df = df.groupby(["gene"])["peak"].count().reset_index()

    df = df.sort_values(by="peak", ascending=True)

    # Map strings to evenly spaced numbers
    df["rank"] = range(len(df))

    # Identify the top 3 categories by value
    labels = df.nlargest(n_labels, "peak")

    # plot TF variances
    fig, ax = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df, x="rank", y="peak", ax=ax)

    # Add text annotations for the top 3 categories
    for _, row in labels.iterrows():
        ax.text(
            row["rank"],
            row["peak"] + 0.5,  # Slightly offset the label above the point
            row["gene"],
            fontsize=10,
            color=label_color,
            ha="center",
        )

    ax.set_xlabel("Ranked genes", fontsize=12)
    ax.set_ylabel("Number of associated peaks", fontsize=12)

    plt.show()
