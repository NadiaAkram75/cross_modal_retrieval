import os
import pandas as pd


EVAL_PATH = "results/retrieval_evaluation.csv"
EXPLAIN_PATH = "results/retrieval_explanation.csv"
FAILURE_PATH = "results/failure_cases.csv"

OUT_MD = "results/interpretation.md"
OUT_CSV = "results/interpretation_web.csv"


SEMANTIC_DISTANCE_NOTE = (
    "semantic distance = 0.000 means closest normalized match after row-wise "
    "min-max scaling, not identical anatomy or identical tumor morphology."
)


def fmt(x, digits=4):
    return f"{float(x):.{digits}f}"


def add_web_row(
    rows,
    section,
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


def main():
    os.makedirs("results", exist_ok=True)

    eval_df = pd.read_csv(EVAL_PATH)
    explain_df = pd.read_csv(EXPLAIN_PATH)
    failure_df = pd.read_csv(FAILURE_PATH)

    visual = eval_df[eval_df["method"] == "visual_only"].iloc[0]
    hybrid = eval_df[eval_df["method"] == "hybrid_semantic_weight_0.75"].iloc[0]
    oracle = eval_df[eval_df["method"] == "semantic_only_oracle"].iloc[0]
    worst = failure_df.iloc[0]

    web_rows = []

    # -------------------------
    # Web CSV: global evaluation
    # -------------------------
    for method_name, row in [
        ("visual_only", visual),
        ("hybrid_semantic_weight_0.75", hybrid),
        ("semantic_only_oracle", oracle),
    ]:
        add_web_row(
            web_rows,
            section="overall_evaluation",
            metric=f"{method_name}_precision_at_5",
            value=fmt(row["precision_at_k_mean"]),
            explanation="Mean precision@5 over all query cases.",
            doctor_note="Higher precision means more retrieved cases are semantically close to the query case.",
        )

        add_web_row(
            web_rows,
            section="overall_evaluation",
            metric=f"{method_name}_ndcg_at_5",
            value=fmt(row["ndcg_at_k_mean"]),
            explanation="Mean nDCG@5 over all query cases.",
            doctor_note="Higher nDCG means the most relevant cases appear earlier in the ranking.",
        )

        add_web_row(
            web_rows,
            section="overall_evaluation",
            metric=f"{method_name}_mean_semantic_distance_at_5",
            value=fmt(row["mean_semantic_distance_at_k_mean"]),
            explanation="Mean semantic distance of retrieved top-5 cases.",
            doctor_note="Lower semantic distance means closer tumor morphology according to mask-derived features.",
        )

    # -------------------------
    # Web CSV: retrieved cases
    # -------------------------
    for _, row in explain_df.iterrows():
        add_web_row(
            web_rows,
            section="retrieval_explanation",
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
            case_id=row["query_case"],
            retrieved_case=row["retrieved_case"],
            rank=int(row["rank"]),
            metric="hybrid_distance",
            value=fmt(row["hybrid_distance"], 3),
            explanation="Hybrid distance combines semantic tumor features and visual embedding distance.",
            doctor_note="This score should be interpreted as relative similarity, not diagnostic equivalence.",
        )

        add_web_row(
            web_rows,
            section="retrieval_explanation",
            case_id=row["query_case"],
            retrieved_case=row["retrieved_case"],
            rank=int(row["rank"]),
            metric="visual_distance",
            value=fmt(row["visual_distance"], 3),
            explanation="Distance from the learned image embedding.",
            doctor_note="The current visual encoder is weaker than the semantic tumor features.",
        )

    # -------------------------
    # Web CSV: failure cases
    # -------------------------
    for _, row in failure_df.iterrows():
        add_web_row(
            web_rows,
            section="failure_case",
            case_id=row["query_case"],
            metric="failure_gap_vs_oracle",
            value=fmt(row["failure_gap_vs_oracle"], 3),
            explanation=(
                "Difference between hybrid retrieval and semantic-only oracle retrieval. "
                "Larger values indicate a stronger ranking error."
            ),
            doctor_note="Even the worst cases remain close to the oracle, so errors are mostly mild ranking errors.",
        )

        add_web_row(
            web_rows,
            section="failure_case",
            case_id=row["query_case"],
            metric="ndcg_at_5",
            value=fmt(row["ndcg_at_k"], 3),
            explanation="Ranking quality for this query case.",
            doctor_note="High nDCG means the retrieved cases remain close to the ideal semantic ranking.",
        )

        add_web_row(
            web_rows,
            section="failure_case",
            case_id=row["query_case"],
            metric="precision_at_5",
            value=fmt(row["precision_at_k"], 3),
            explanation="Fraction of top-5 retrieved cases that belong to the top semantic neighborhood.",
            doctor_note="This helps identify whether the retrieved cases are clinically plausible neighbors.",
        )

    web_df = pd.DataFrame(web_rows)
    web_df.to_csv(OUT_CSV, index=False)

    # -------------------------
    # Markdown report output
    # -------------------------
    text = f"""# Interpretation of Retrieval Results

## 1. Overall Evaluation

The visual-only retrieval baseline performs poorly. It reaches precision@5 of {fmt(visual["precision_at_k_mean"])} and nDCG@5 of {fmt(visual["ndcg_at_k_mean"])}. This shows that the learned visual embedding alone does not reliably retrieve cases with similar tumor morphology.

The hybrid method with semantic weight 0.75 performs much better. It reaches precision@5 of {fmt(hybrid["precision_at_k_mean"])} and nDCG@5 of {fmt(hybrid["ndcg_at_k_mean"])}. This means that most retrieved cases are close to the query according to tumor-mask-derived semantic features.

The semantic-only oracle reaches precision@5 of {fmt(oracle["precision_at_k_mean"])} and nDCG@5 of {fmt(oracle["ndcg_at_k_mean"])}. This is treated as an upper bound, not as the main deployed method.

Main conclusion: the current system is clinically meaningful mainly because of the semantic tumor-mask features. The visual encoder is still weak and should be improved in future work.

## 2. Retrieval Explanation

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

## 3. Failure-Case Analysis

The failure-case analysis shows that the hybrid method is stable rather than brittle. The worst observed case was `{worst["query_case"]}`, with a gap of {fmt(worst["failure_gap_vs_oracle"], 3)} from the semantic oracle.

Even in the worst cases, nDCG@5 remains high. The worst listed case has nDCG@5 of {fmt(worst["ndcg_at_k"], 3)} and precision@5 of {fmt(worst["precision_at_k"], 3)}. This suggests that the errors are mostly mild ranking errors, not completely irrelevant retrievals.

## 4. Web Application Interpretation

For a future doctor-facing web application, the system should display the query case, retrieved cases, similarity scores, tumor feature explanations, responsible doctor or case owner, and a clear warning that normalized semantic distance values indicate relative closeness rather than identical anatomy.

The web-ready CSV file is saved as `results/interpretation_web.csv`.
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved report interpretation to: {OUT_MD}")
    print(f"Saved web-ready interpretation table to: {OUT_CSV}")


if __name__ == "__main__":
    main()