import os
import pandas as pd


AUTO_EVAL_PATH = "results/tables/autoencoder_retrieval_evaluation.csv"
CONTRASTIVE_EVAL_PATH = "results/tables/contrastive_retrieval_evaluation.csv"
ENCODER_COMPARISON_PATH = "results/tables/encoder_comparison_summary.csv"
EXPLAIN_PATH = "results/tables/retrieval_explanation.csv"
FAILURE_PATH = "results/tables/failure_cases.csv"

OUT_MD = "results/interpretation.md"
OUT_CSV = "results/tables/interpretation_web.csv"


SEMANTIC_DISTANCE_NOTE = (
    "semantic distance = 0.000 means closest normalized match after row-wise "
    "min-max scaling, not identical anatomy or identical tumor morphology."
)


def fmt(x, digits=4):
    return f"{float(x):.{digits}f}"


def get_row(df, method):
    selected = df[df["method"] == method]

    if selected.empty:
        raise ValueError(f"Method not found: {method}")

    return selected.iloc[0]


def add_web_row(
    rows,
    section,
    model="",
    case_id="",
    retrieved_case="",
    rank="",
    metric="",
    value="",
    explanation="",
    doctor_note="",
):
    rows.append(
        {
            "section": section,
            "model": model,
            "case_id": case_id,
            "retrieved_case": retrieved_case,
            "rank": rank,
            "metric": metric,
            "value": value,
            "explanation": explanation,
            "doctor_note": doctor_note,
            "semantic_distance_note": SEMANTIC_DISTANCE_NOTE,
        }
    )


def add_metric_rows(rows, section, model_name, row):
    add_web_row(
        rows,
        section=section,
        model=model_name,
        metric="precision_at_5",
        value=fmt(row["precision_at_k_mean"]),
        explanation="Mean precision@5 over all query cases.",
        doctor_note=(
            "Higher precision means more retrieved cases are semantically close "
            "to the query case."
        ),
    )

    add_web_row(
        rows,
        section=section,
        model=model_name,
        metric="recall_at_5",
        value=fmt(row["recall_at_k_mean"]),
        explanation="Mean recall@5 over all query cases.",
        doctor_note=(
            "Higher recall means the retrieved set captures more of the semantic "
            "neighborhood around the query case."
        ),
    )

    add_web_row(
        rows,
        section=section,
        model=model_name,
        metric="ndcg_at_5",
        value=fmt(row["ndcg_at_k_mean"]),
        explanation="Mean nDCG@5 over all query cases.",
        doctor_note="Higher nDCG means the most relevant cases appear earlier in the ranking.",
    )

    add_web_row(
        rows,
        section=section,
        model=model_name,
        metric="mean_semantic_distance_at_5",
        value=fmt(row["mean_semantic_distance_at_k_mean"]),
        explanation="Mean semantic distance of retrieved top-5 cases.",
        doctor_note=(
            "Lower semantic distance means closer tumor morphology according to "
            "mask-derived features."
        ),
    )


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    auto_df = pd.read_csv(AUTO_EVAL_PATH)
    contrastive_df = pd.read_csv(CONTRASTIVE_EVAL_PATH)
    encoder_df = pd.read_csv(ENCODER_COMPARISON_PATH)
    explain_df = pd.read_csv(EXPLAIN_PATH)
    failure_df = pd.read_csv(FAILURE_PATH)

    auto_visual = get_row(auto_df, "visual_only")
    auto_hybrid = get_row(auto_df, "hybrid_semantic_weight_0.75")
    contrastive_visual = get_row(contrastive_df, "visual_only")
    contrastive_hybrid = get_row(contrastive_df, "hybrid_semantic_weight_0.75")
    oracle = get_row(contrastive_df, "semantic_only_oracle")

    worst = failure_df.iloc[0]

    web_rows = []

    add_metric_rows(
        web_rows,
        section="encoder_comparison",
        model_name="autoencoder_visual_only",
        row=auto_visual,
    )

    add_metric_rows(
        web_rows,
        section="encoder_comparison",
        model_name="contrastive_visual_only",
        row=contrastive_visual,
    )

    add_metric_rows(
        web_rows,
        section="encoder_comparison",
        model_name="autoencoder_hybrid_0.75",
        row=auto_hybrid,
    )

    add_metric_rows(
        web_rows,
        section="encoder_comparison",
        model_name="contrastive_hybrid_0.75",
        row=contrastive_hybrid,
    )

    add_metric_rows(
        web_rows,
        section="encoder_comparison",
        model_name="semantic_only_oracle",
        row=oracle,
    )

    add_web_row(
        web_rows,
        section="model_interpretation",
        model="contrastive_encoder",
        metric="main_finding",
        value="visual encoder improved",
        explanation=(
            "Contrastive training improved visual-only retrieval from "
            f"Precision@5={fmt(auto_visual['precision_at_k_mean'])} to "
            f"Precision@5={fmt(contrastive_visual['precision_at_k_mean'])}, "
            f"and from nDCG@5={fmt(auto_visual['ndcg_at_k_mean'])} to "
            f"nDCG@5={fmt(contrastive_visual['ndcg_at_k_mean'])}."
        ),
        doctor_note=(
            "The visual encoder is now more aligned with tumor morphology, "
            "but semantic mask features remain the strongest signal."
        ),
    )

    for _, row in explain_df.iterrows():
        add_web_row(
            web_rows,
            section="retrieval_explanation",
            model="contrastive_hybrid_or_current_hybrid",
            case_id=row["query_case"],
            retrieved_case=row["retrieved_case"],
            rank=int(row["rank"]),
            metric="semantic_distance",
            value=fmt(row["semantic_distance"], 3),
            explanation=row["main_similarity_explanation"],
            doctor_note=(
                "This retrieved case is similar to the query case based on tumor volume, "
                "tumor composition, centroid location, and tumor extent features."
            ),
        )

        add_web_row(
            web_rows,
            section="retrieval_explanation",
            model="contrastive_hybrid_or_current_hybrid",
            case_id=row["query_case"],
            retrieved_case=row["retrieved_case"],
            rank=int(row["rank"]),
            metric="hybrid_distance",
            value=fmt(row["hybrid_distance"], 3),
            explanation=(
                "Hybrid distance combines semantic tumor features and visual embedding distance."
            ),
            doctor_note=(
                "This score should be interpreted as relative similarity, not diagnostic equivalence."
            ),
        )

        add_web_row(
            web_rows,
            section="retrieval_explanation",
            model="contrastive_hybrid_or_current_hybrid",
            case_id=row["query_case"],
            retrieved_case=row["retrieved_case"],
            rank=int(row["rank"]),
            metric="visual_distance",
            value=fmt(row["visual_distance"], 3),
            explanation="Distance from the learned image embedding.",
            doctor_note=(
                "Lower visual distance means the image encoder considers the cases visually closer."
            ),
        )

    for _, row in failure_df.iterrows():
        add_web_row(
            web_rows,
            section="failure_case",
            model="hybrid_0.75",
            case_id=row["query_case"],
            metric="failure_gap_vs_oracle",
            value=fmt(row["failure_gap_vs_oracle"], 3),
            explanation=(
                "Difference between hybrid retrieval and semantic-only oracle retrieval. "
                "Larger values indicate a stronger ranking error."
            ),
            doctor_note=(
                "Even the worst cases remain close to the oracle, so errors are mostly "
                "mild ranking errors."
            ),
        )

        add_web_row(
            web_rows,
            section="failure_case",
            model="hybrid_0.75",
            case_id=row["query_case"],
            metric="ndcg_at_5",
            value=fmt(row["ndcg_at_k"], 3),
            explanation="Ranking quality for this query case.",
            doctor_note=(
                "High nDCG means the retrieved cases remain close to the ideal semantic ranking."
            ),
        )

        add_web_row(
            web_rows,
            section="failure_case",
            model="hybrid_0.75",
            case_id=row["query_case"],
            metric="precision_at_5",
            value=fmt(row["precision_at_k"], 3),
            explanation=(
                "Fraction of top-5 retrieved cases that belong to the top semantic neighborhood."
            ),
            doctor_note=(
                "This helps identify whether the retrieved cases are clinically plausible neighbors."
            ),
        )

    web_df = pd.DataFrame(web_rows)
    web_df.to_csv(OUT_CSV, index=False)

    text = f"""# Interpretation of Retrieval Results

## 1. Overall Evaluation

The original autoencoder visual-only baseline performs poorly. It reaches Precision@5 of {fmt(auto_visual["precision_at_k_mean"])} and nDCG@5 of {fmt(auto_visual["ndcg_at_k_mean"])}. This indicates that reconstruction-based image embeddings do not reliably capture clinically meaningful tumor similarity.

After contrastive training, the visual-only encoder improves substantially. Precision@5 increases to {fmt(contrastive_visual["precision_at_k_mean"])} and nDCG@5 increases to {fmt(contrastive_visual["ndcg_at_k_mean"])}. This shows that using semantic-neighbor positives during training helps the visual encoder learn representations that are more aligned with tumor morphology.

The hybrid method also improves after contrastive training. The autoencoder hybrid model with semantic weight 0.75 reaches Precision@5 of {fmt(auto_hybrid["precision_at_k_mean"])} and nDCG@5 of {fmt(auto_hybrid["ndcg_at_k_mean"])}. The contrastive hybrid model with the same semantic weight reaches Precision@5 of {fmt(contrastive_hybrid["precision_at_k_mean"])} and nDCG@5 of {fmt(contrastive_hybrid["ndcg_at_k_mean"])}.

The semantic-only oracle reaches Precision@5 of {fmt(oracle["precision_at_k_mean"])} and nDCG@5 of {fmt(oracle["ndcg_at_k_mean"])}. This is treated as an upper bound, not as the main deployed method.

Main conclusion: contrastive training improves the visual branch, but semantic tumor-mask features remain the strongest signal. The best current system is the contrastive hybrid retrieval model.

## 2. Encoder Comparison

The contrastive encoder improves visual-only retrieval quality compared with the autoencoder baseline:

- Autoencoder visual-only Precision@5: {fmt(auto_visual["precision_at_k_mean"])}
- Contrastive visual-only Precision@5: {fmt(contrastive_visual["precision_at_k_mean"])}
- Autoencoder visual-only nDCG@5: {fmt(auto_visual["ndcg_at_k_mean"])}
- Contrastive visual-only nDCG@5: {fmt(contrastive_visual["ndcg_at_k_mean"])}

This confirms that the autoencoder mainly learns reconstruction-oriented appearance, whereas contrastive training produces embeddings that are more useful for case retrieval.

## 3. Retrieval Explanation

For the selected query case `{explain_df.iloc[0]["query_case"]}`, the hybrid system retrieves cases that are similar mainly in tumor volume, centroid location, tumor extent, and enhancing-tumor ratio.

Top retrieved cases:

"""

    for _, row in explain_df.iterrows():
        text += (
            f"- Rank {int(row['rank'])}: `{row['retrieved_case']}` "
            f"| semantic distance = {fmt(row['semantic_distance'], 3)} "
            f"| explanation: {row['main_similarity_explanation']}\n"
        )

    text += f"""

Important note: `{SEMANTIC_DISTANCE_NOTE}`

## 4. Failure-Case Analysis

The failure-case analysis shows that the hybrid method is stable rather than brittle. The worst observed case was `{worst["query_case"]}`, with a gap of {fmt(worst["failure_gap_vs_oracle"], 3)} from the semantic oracle.

Even in the worst cases, nDCG@5 remains high. The worst listed case has nDCG@5 of {fmt(worst["ndcg_at_k"], 3)} and Precision@5 of {fmt(worst["precision_at_k"], 3)}. This suggests that the errors are mostly mild ranking errors, not completely irrelevant retrievals.

## 5. Web Application Interpretation

For a future doctor-facing web application, the system should display:

- the query case,
- retrieved similar cases,
- similarity scores,
- tumor feature explanations,
- responsible doctor or case owner,
- visual and semantic similarity separately,
- and a clear warning that normalized semantic distance values indicate relative closeness rather than identical anatomy.

The web-ready CSV file is saved as `results/tables/interpretation_web.csv`.
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved report interpretation to: {OUT_MD}")
    print(f"Saved web-ready interpretation table to: {OUT_CSV}")


if __name__ == "__main__":
    main()