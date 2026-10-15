# Model Card — HyMaC-Net

Following the structure of *Model Cards for Model Reporting* (Mitchell et al., 2019).

## Model details

| | |
|---|---|
| **Developed by** | Sourav Das, Shibam Chakraborty, Sakibul Hasan Turja, Md. Tanvirul Islam Rifat — BRAC University (B.Sc. thesis, CSE400), supervised by Dr. Amitabha Chakrabarty and Nirjhor Datta |
| **Type** | Image classifier: CNN stem + patch embedding + 3 Mamba state-space blocks + fused pooling and auxiliary heads |
| **Variants** | HyMaC-Net-S (9.4 M parameters, 1.26 GFLOPs), HyMaC-Net-B (20.5 M, 2.36 GFLOPs) |
| **Input** | One 2-D image, 224 × 224, grayscale or RGB, normalised per dataset |
| **Output** | Class probabilities for one dataset's label set; one model is trained per dataset |
| **Framework** | PyTorch 2, `mamba-ssm`, mixed precision |
| **Status** | Research prototype. Code and weights to be released with the paper. |

## Intended use

- **Primary:** research on efficient hybrid architectures for 2-D medical image classification, and benchmarking against CNN, Transformer and state-space models.
- **Users:** machine-learning researchers.
- **Out of scope:** diagnosis, triage or any decision about a patient. The model is not a medical device, has not been evaluated prospectively or by clinicians, and has not been through regulatory review.

## Training and evaluation data

Ten MedMNIST v2 datasets (official splits, 224 × 224), CPN X-ray and Kvasir (70 / 15 / 15 splits), and BRISC-25 → Brain Tumor MRI for cross-dataset evaluation. See the [dataset table](../README.md#datasets). The data are public, de-identified research collections; they do not represent any particular hospital's patient population, and demographic attributes are not available for subgroup analysis.

## Metrics

Accuracy, AUC, precision, recall, specificity, F1 and Cohen's κ on held-out test splits; precision, recall and F1 are support-weighted and specificity is macro-averaged, as in the thesis. Per-class and macro metrics are derived from the released confusion matrices ([`results/derived/`](../results/derived/)). Efficiency: parameters, GFLOPs, training time per epoch and inference latency.

## Performance summary

| | HyMaC-Net-S | HyMaC-Net-B |
|---|---|---|
| Mean accuracy, 9 MedMNIST datasets | 88.5 % | 88.0 % |
| Best result | BloodMNIST 98.3 % | BloodMNIST 97.9 % |
| Weakest result | TissueMNIST 68.2 % | TissueMNIST 67.8 % |
| Cross-dataset brain MRI | — | 75.4 % accuracy, AUC 0.907 |

Full results: [README → Results](../README.md#results).

## Known limitations and risks

| Risk | Evidence | Consequence |
|---|---|---|
| **Missed tumours under domain shift** | Cross-dataset glioma recall 21 %; 49 of 90 gliomas predicted as *no tumour* | A false "no tumour" is the costliest error; not usable for screening on a new site without adaptation and re-validation. |
| **Minority classes** | DermaMNIST macro-F1 0.52 vs weighted F1 0.73; melanoma recall 32 %, actinic keratoses 30 %, dermatofibroma 17 % | Weighted metrics hide poor performance on rare, often clinically important classes. |
| **Possible shortcut features** | Grad-CAM activations on image borders (CPN X-ray, DermaMNIST, PathMNIST, TissueMNIST, BloodMNIST) and near the Kvasir overlay box | The model may use acquisition artefacts. A quantitative localisation or artefact-masking test is needed before any claim of clinical grounding. |
| **Low specificity on paediatric X-ray** | PneumoniaMNIST: 81 of 234 normal images predicted as pneumonia | Over-calls disease on normal radiographs. |
| **Single runs** | Only PathMNIST has a multi-seed study | Small differences between models on the small test sets (BreastMNIST: 156 images) are within run-to-run noise. |
| **No calibration analysis** | — | Probabilities should not be read as confidence without calibration. |

## Ethical considerations

The data are public and de-identified, so no new patient data were collected. The main risk is misuse as a diagnostic tool; this card and the repository state that the model is for research only. Compute was kept small on purpose — one consumer GPU — which lowers the energy cost of training and the barrier to use in low-resource settings.

## Citation

See [README → Citation](../README.md#citation).
