import os
import pandas as pd
import matplotlib.pyplot as plt


AUTO_PATH = "results/tables/autoencoder_retrieval_evaluation.csv"
CONTRASTIVE_PATH = "results/tables/contrastive_retrieval_evaluation.csv"

OUT_CSV = "results/tables/encoder_comparison_summary.csv"
OUT_FIG = "results/figures/encoder_comparison.png"


def get_row(df, method):
    selected = df[df["method"] == method]

    if selected.empty:
        raise ValueError(f"Method not found in results file: {method}")

    return selected.iloc[0]


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    auto_df = pd.read_csv(AUTO_PATH)
    contrastive_df = pd.read_csv(CONTRASTIVE_PATH)

    comparisons = [
        ("visual_only", "Autoencoder visual-only", auto_df),
        ("visual_only", "Contrastive visual-only", contrastive_df),
        ("hybrid_semantic_weight_0.75", "Autoencoder hybrid 0.75", auto_df),
        ("hybrid_semantic_weight_0.75", "Contrastive hybrid 0.75", contrastive_df),
        ("semantic_only_oracle", "Semantic-only oracle", auto_df),
    ]

    rows = []

    for method, label, df in comparisons:
        row = get_row(df, method)

        rows.append(
            {
                "model": label,
                "precision_at_5": row["precision_at_k_mean"],
                "recall_at_5": row["recall_at_k_mean"],
                "ndcg_at_5": row["ndcg_at_k_mean"],
                "mean_semantic_distance_at_5": row[
                    "mean_semantic_distance_at_k_mean"
                ],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)

    plot_df = summary[
        summary["model"].isin(
            [
                "Autoencoder visual-only",
                "Contrastive visual-only",
                "Autoencoder hybrid 0.75",
                "Contrastive hybrid 0.75",
            ]
        )
    ].copy()

    ax = plot_df.plot(
        x="model",
        y=["precision_at_5", "ndcg_at_5"],
        kind="bar",
        figsize=(11, 5),
        rot=10,
    )

    ax.set_title("Effect of Contrastive Training on Retrieval Quality")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(["Precision@5", "nDCG@5"], loc="upper left")

    plt.xticks(fontsize=9, ha="right")
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved summary CSV: {OUT_CSV}")
    print(f"Saved comparison figure: {OUT_FIG}")
    print()
    print(summary)


if __name__ == "__main__":
    main()