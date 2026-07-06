# Interpretation of Retrieval Results

## 1. Overall Evaluation

The visual-only retrieval baseline performs poorly. It reaches precision@5 of 0.0413 and nDCG@5 of 0.5070. This shows that the learned visual embedding alone does not reliably retrieve cases with similar tumor morphology.

The hybrid method with semantic weight 0.75 performs much better. It reaches precision@5 of 0.9777 and nDCG@5 of 0.9755. This means that most retrieved cases are close to the query according to tumor-mask-derived semantic features.

The semantic-only oracle reaches precision@5 of 1.0000 and nDCG@5 of 1.0000. This is treated as an upper bound, not as the main deployed method.

Main conclusion: the current system is clinically meaningful mainly because of the semantic tumor-mask features. The visual encoder is still weak and should be improved in future work.

## 2. Retrieval Explanation

For the selected query case `BraTS20_Training_367`, the hybrid system retrieves cases that are similar mainly in tumor volume, centroid location, tumor extent, and enhancing-tumor ratio.

Top retrieved cases:

- Rank 1: `BraTS20_Training_089` | semantic distance = 0.026 | explanation: tumor_volume_fraction_brain close; extent_max close; enhancing_fraction_tumor close
- Rank 2: `BraTS20_Training_243` | semantic distance = 0.000 | explanation: tumor_volume_fraction_brain close; centroid_x close; extent_max close
- Rank 3: `BraTS20_Training_220` | semantic distance = 0.069 | explanation: tumor_volume_fraction_brain close; enhancing_fraction_tumor close; centroid_x close
- Rank 4: `BraTS20_Training_358` | semantic distance = 0.012 | explanation: tumor_volume_fraction_brain close; centroid_y close; centroid_x close
- Rank 5: `BraTS20_Training_104` | semantic distance = 0.080 | explanation: centroid_x close; tumor_volume_fraction_brain close; extent_max close


Important note: `semantic distance = 0.000 means closest normalized match after row-wise min-max scaling, not identical anatomy or identical tumor morphology.`

## 3. Failure-Case Analysis

The failure-case analysis shows that the hybrid method is stable rather than brittle. The worst observed case was `BraTS20_Training_307`, with a gap of 0.055 from the semantic oracle.

Even in the worst cases, nDCG@5 remains high. The worst listed case has nDCG@5 of 0.935 and precision@5 of 1.000. This suggests that the errors are mostly mild ranking errors, not completely irrelevant retrievals.

## 4. Web Application Interpretation

For a future doctor-facing web application, the system should display the query case, retrieved cases, similarity scores, tumor feature explanations, responsible doctor or case owner, and a clear warning that normalized semantic distance values indicate relative closeness rather than identical anatomy.

The web-ready CSV file is saved as `results/interpretation_web.csv`.
