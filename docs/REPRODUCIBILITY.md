# Reproducibility

## What is released now

| Item | Where |
|---|---|
| Full thesis (methodology, ablations, algorithms, results) | [`HyMaC-Net_BSc_Thesis_BRACU_2025.pdf`](HyMaC-Net_BSc_Thesis_BRACU_2025.pdf) |
| Layer-by-layer specification with equations | [`METHOD.md`](METHOD.md) |
| Confusion matrices (12 benchmarks + cross-dataset), as JSON | [`results/confusion_matrices/`](../results/confusion_matrices/) |
| Every result table as CSV: per-dataset metrics, baselines, efficiency, ablation, seeds, cross-dataset, training configuration | [`results/`](../results/) |
| Metric recomputation, exact Wilcoxon test, consistency checks, figure generation | [`tools/`](../tools/) |

## What follows on publication

- Model and training code (PyTorch), including data loading for all twelve datasets.
- Trained weights for HyMaC-Net-S and HyMaC-Net-B per dataset.
- Training logs for the ablation runs and the 5-seed PathMNIST study.

Until then, code and weights are available to reviewers on request through the repository maintainer.

## Re-implementation checklist

| Setting | Value |
|---|---|
| Input | 224 × 224; grayscale datasets kept single-channel |
| Normalisation | Channel-wise mean and standard deviation of each dataset's training split |
| Augmentation | None for the 12 benchmarks; random horizontal flip and slight brightness/contrast jitter for the cross-dataset experiment only |
| Splits | MedMNIST v2 official train / validation / test; CPN X-ray and Kvasir stratified 70 / 15 / 15 |
| Optimiser | AdamW, lr 1e-4 (per-dataset values in [`training_config.csv`](../results/training_config.csv)), weight decay 1e-2 |
| Schedule | Cosine annealing over the training epochs |
| Epochs | Up to 100, early stopping with patience 10 on validation accuracy, best checkpoint kept |
| Batch size | 32 |
| Loss | Focal (γ = 2.0) or cross-entropy with label smoothing 0.1; class weights ∝ 1 / class frequency |
| Regularisation | Dropout 0.15 (blocks and heads), DropBlock 0.15 (stem) |
| Auxiliary head | After block 1 of 3; logits fused with weight 0.3; one loss on the fused logits |
| Precision | Mixed (`torch.cuda.amp`) |
| Seeds | 7, 21, 42, 77, 99 for the PathMNIST significance study; fixed seeds elsewhere |
| Data loading | Multi-worker `DataLoader`, pinned memory |
| Hardware | RTX 4080 Super 16 GB (local) and 2 × Tesla T4 (Kaggle) |

## Evaluating against our numbers

Compute accuracy on the official test split and compare with [`results/per_dataset_small.csv`](../results/per_dataset_small.csv) and [`results/per_dataset_base.csv`](../results/per_dataset_base.csv). Precision, recall and F1 there are support-weighted (scikit-learn `average="weighted"`); specificity is macro-averaged. To compare against a confusion matrix of your own, use `tools.metrics.confusion_metrics`.
