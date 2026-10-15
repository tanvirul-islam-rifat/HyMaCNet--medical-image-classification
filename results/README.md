# Results data

Every number in the main README comes from the files in this folder. They are plain CSV and JSON so they can be checked, reused and cited without the model code.

## Files

| File | Contents | Thesis source |
|---|---|---|
| `confusion_matrices/medical_benchmarks_base.json` | HyMaC-Net-B confusion matrices, 12 datasets, with class labels (rows = true, columns = predicted) | Figure 5.1 |
| `confusion_matrices/cross_dataset_brain_mri_base.json` | HyMaC-Net-B, BRISC-25 → Brain Tumor MRI | Figure 5.6 |
| `per_dataset_small.csv`, `per_dataset_base.csv` | Accuracy (%), AUC, weighted precision / recall / F1, macro specificity, Cohen's κ | Tables 5.1, 5.2 |
| `datasets.csv` | Modality, task, classes and size of each dataset | Table 4.1 |
| `baselines_medmnist.csv` | Published AUC and accuracy of ResNet, AutoML, MedViT and MedMamba on 10 MedMNIST datasets | Tables 5.9, 5.10 |
| `baselines_other.csv` | Published results on CPN X-ray and Kvasir | Tables 5.11, 5.12 |
| `efficiency.csv` | Parameters and GFLOPs of every compared model | Table 5.8 |
| `ablation_pathmnist.csv` | All ablation strategies (phases A–D) and the Linformer mixer test | Tables 4.3–4.6, 4.8 |
| `strategy_20_vs_21.csv` | 3-block vs 2-block candidate on 10 MedMNIST datasets | Table 4.7 |
| `seeds_pathmnist.csv` | HyMaC-Net-B and ResNet-50, 5 seeds each | Tables 5.5, 5.6 |
| `cross_dataset.csv` | HyMaC-Net-B, ResNet-50 and ViT-B/16 under domain shift | Table 5.13 |
| `training_config.csv` | Per-dataset learning rate, loss and regularisation | Table 5.4 |
| `derived/` | **Generated** by `python -m tools.metrics --write` — do not edit by hand | — |

## How accuracy is sourced

Accuracy equals support-weighted recall by definition, so every accuracy here is the weighted recall of its own row, and for HyMaC-Net-B it is cross-checked against the confusion matrix (`oa_source` column). Where the printed thesis tables show a different accuracy, that value is kept in the `oa_printed_in_thesis` column for traceability, and this repository follows the confusion matrices. The TissueMNIST Base row is taken entirely from its confusion matrix except AUC, which a confusion matrix cannot provide.

## Notes

- The OrganAMNIST, OCTMNIST, PneumoniaMNIST, DermaMNIST, BreastMNIST, BloodMNIST and TissueMNIST matrices cover the full official MedMNIST v2 test splits. The PathMNIST matrix contains 2,154 images and the Kvasir matrix 1,200.
- In `seeds_pathmnist.csv`, the HyMaC-Net-B seed-21 accuracy is 0.9443, the value consistent with that run's recall and with the range and mean reported for the five seeds.
- In `ablation_pathmnist.csv`, strategy 2's parameter count is left empty: the printed 7.0 M cannot produce its 273 MB model size. Phase A inference times were reported in seconds rather than milliseconds per image and are omitted.
- Baseline results are those published by each model's authors as cited in the thesis; they were not re-run.
