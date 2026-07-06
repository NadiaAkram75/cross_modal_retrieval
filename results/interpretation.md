# Interpretation of Retrieval Results

## 1. Overall Evaluation

The original autoencoder visual-only baseline performs poorly. It reaches Precision@5 of 0.0413 and nDCG@5 of 0.5070. This indicates that reconstruction-based image embeddings do not reliably capture clinically meaningful tumor similarity.

After contrastive training, the visual-only encoder improves substantially. Precision@5 increases to 0.4082 and nDCG@5 increases to 0.7539. This shows that using semantic-neighbor positives during training helps the visual encoder learn representations that are more aligned with tumor morphology.

The hybrid method also improves after contrastive training. The autoencoder hybrid model with semantic weight 0.75 reaches Precision@5 of 0.9777 and nDCG@5 of 0.9755. The contrastive hybrid model with the same semantic weight reaches Precision@5 of 0.9946 and nDCG@5 of 0.9942.

The semantic-only oracle reaches Precision@5 of 1.0000 and nDCG@5 of 1.0000. This is treated as an upper bound, not as the main deployed method.

Main conclusion: contrastive training improves the visual branch, but semantic tumor-mask features remain the strongest signal. The best current system is the contrastive hybrid retrieval model.

## 2. Encoder Comparison

The contrastive encoder improves visual-only retrieval quality compared with the autoencoder baseline:

- Autoencoder visual-only Precision@5: 0.0413
- Contrastive visual-only Precision@5: 0.4082
- Autoencoder visual-only nDCG@5: 0.5070
- Contrastive visual-only nDCG@5: 0.7539

This confirms that the autoencoder mainly learns reconstruction-oriented appearance, whereas contrastive training produces embeddings that are more useful for case retrieval.

## 3. Retrieval Explanation

For the selected query case `BraTS20_Training_367`, the hybrid system retrieves cases that are similar mainly in tumor volume, centroid location, tumor extent, and enhancing-tumor ratio.

Top retrieved cases:

- Rank 1: `BraTS20_Training_089` | semantic distance = 0.026 | explanation: tumor_volume_fraction_brain close; extent_max close; enhancing_fraction_tumor close
- Rank 2: `BraTS20_Training_243` | semantic distance = 0.000 | explanation: tumor_volume_fraction_brain close; centroid_x close; extent_max close
- Rank 3: `BraTS20_Training_220` | semantic distance = 0.069 | explanation: tumor_volume_fraction_brain close; enhancing_fraction_tumor close; centroid_x close
- Rank 4: `BraTS20_Training_358` | semantic distance = 0.012 | explanation: tumor_volume_fraction_brain close; centroid_y close; centroid_x close
- Rank 5: `BraTS20_Training_104` | semantic distance = 0.080 | explanation: centroid_x close; tumor_volume_fraction_brain close; extent_max close


Important note: `semantic distance = 0.000 means closest normalized match after row-wise min-max scaling, not identical anatomy or identical tumor morphology.`

## 4. Failure-Case Analysis

The failure-case analysis shows that the hybrid method is stable rather than brittle. The worst observed case was `BraTS20_Training_307`, with a gap of 0.055 from the semantic oracle.

Even in the worst cases, nDCG@5 remains high. The worst listed case has nDCG@5 of 0.935 and Precision@5 of 1.000. This suggests that the errors are mostly mild ranking errors, not completely irrelevant retrievals.

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
