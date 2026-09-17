# Explainable Multimodal EHR + Imaging Risk Prediction System

**Status:** Independent ongoing research / applied project. Not yet submitted for peer review.

**Author:** Sabit Md Asad ([sabitpe97.com](https://sabitpe97.com) · [GitHub](https://github.com/Sabit400) · [ResearchGate](https://researchgate.net/profile/Sabit-Md-Asad-2))

## Overview

This project builds a multimodal risk-prediction system that fuses structured
electronic health record (EHR) data with imaging data through a dual-branch
neural network, and provides **dual, modality-specific explainability**: SHAP
attribution for the EHR branch and Grad-CAM attention maps for the imaging
branch. It also includes a **modality ablation study** — EHR-only and
imaging-only models trained under identical conditions — to provide direct
evidence that the fusion architecture adds real predictive value rather than
asserting it.

This is a hands-on extension of the multimodal / explainable-AI healthcare
themes addressed in prior published research (early disease diagnosis,
generative AI for medical imaging), built as an independent, runnable fusion
framework rather than a replication of any specific study.

## Data note

Both modalities in this project are **synthetically generated**
(`data/generate_data.py`). Real credentialed EHR and medical imaging datasets
(e.g., MIMIC-IV, NIH ChestX-ray14) require external access not available in
this project's development environment, and the "imaging" data here is a
structured grayscale opacity-pattern array — **not a real chest X-ray or any
other real medical image**.

The generator was deliberately designed with **two distinct latent risk
components** — a "systemic" component driving the EHR features and a
"structural" component driving the imaging patterns — combined to produce the
label. This was not incidental: an earlier version of the generator drove
both modalities from a single shared latent variable, and in that version the
EHR-only model alone recovered nearly all of the fusion model's accuracy,
which would have made building a fusion architecture pointless to
demonstrate. With two distinct components, neither modality alone can fully
recover the label, which is what makes the fusion result below meaningful.
This is stated plainly so results are read correctly: they demonstrate
multimodal fusion and dual-explainability methodology, not real diagnostic
performance.

## Methodology

![Methodology workflow](diagrams/methodology_workflow.svg)

1. **Data generation** — 4,000 paired synthetic records: EHR features (age,
   heart rate, respiratory rate, SpO2, temperature, WBC count, CRP,
   comorbidity score) and 64x64 grayscale imaging proxies.
2. **Dual-branch encoding** — an MLP encodes the EHR features into a 16-dim
   embedding; a small CNN encodes the image into a 16-dim embedding.
3. **Late fusion** — the two embeddings are concatenated and passed through a
   fusion classifier head, trained with a class-weighted loss (High Risk is
   the minority class at ~26% prevalence, and false negatives are more
   costly than false positives in a risk-screening context).
4. **Evaluation** — accuracy, precision/recall/F1, ROC-AUC, confusion matrix.
5. **Modality ablation** — EHR-only and imaging-only models trained under
   identical conditions (same class weighting, same epochs) for direct
   comparison against the fusion model.
6. **Dual explainability** — SHAP (`GradientExplainer`) for the EHR branch's
   feature contributions, and Grad-CAM for the imaging branch's spatial
   attention, both computed against the full fusion model (not isolated
   single-modality surrogates).
7. **Prototype serving** — a FastAPI `/predict` endpoint returning the risk
   prediction plus both explanations (EHR attribution values and a Grad-CAM
   overlay image) in a single response.

## Algorithms and tools

| Category | Tools / Algorithms |
|---|---|
| EHR branch | 2-layer MLP (PyTorch) |
| Imaging branch | 4-layer CNN (PyTorch) |
| Fusion | Late fusion (embedding concatenation) + classifier head |
| Explainability | SHAP `GradientExplainer` (EHR), Grad-CAM (imaging) |
| Serving | FastAPI, Pydantic, Uvicorn, Pillow |
| Core stack | Python, PyTorch, scikit-learn, pandas, matplotlib, seaborn |

## Results

### Fusion model performance

| Metric | Value |
|---|---|
| Overall accuracy | 77.2% |
| ROC-AUC | 0.815 |
| Low Risk — Precision / Recall / F1 | 0.87 / 0.81 / 0.84 |
| High Risk — Precision / Recall / F1 | 0.56 / 0.66 / 0.60 |

Full metrics: [`results/metrics.json`](results/metrics.json) ·
Confusion matrix: [`results/confusion_matrix.png`](results/confusion_matrix.png) ·
ROC curve: [`results/roc_curve.png`](results/roc_curve.png)

The class-weighted loss was chosen deliberately: an earlier unweighted
version reached a higher raw accuracy (78.6%) but only 40% recall on the
High Risk class. Since missing a high-risk case is generally more costly
than a false alarm in a risk-screening context, the class-weighted version
(66% High Risk recall, 77.2% accuracy) is the one reported as the primary
result — a deliberate, documented trade-off rather than an accuracy-maximizing
default.

### Modality ablation — evidence that fusion adds value

| Model | Test accuracy |
|---|---|
| EHR only | 69.2% |
| Imaging only | 69.3% |
| **Fusion (EHR + Imaging)** | **77.2%** |

![Ablation comparison](results/ablation_comparison.png)

Fusion outperforms the best single modality by **+7.9 percentage points**,
consistent with the two-component data design described above: each
modality alone only reveals half of the underlying risk signal.
Full data: [`results/ablation_results.json`](results/ablation_results.json).

### Explainability

**SHAP (EHR branch):** SpO2, heart rate, and respiratory rate are the three
most influential EHR features for the High Risk prediction — clinically
consistent with a pulmonary/systemic risk indicator.

![SHAP EHR summary](results/shap_ehr_summary.png)

**Grad-CAM (imaging branch):** for correctly-classified High Risk cases, the
imaging branch's attention consistently localizes onto the high-intensity
opacity regions the data generator used to encode structural risk —
verifiable, honest evidence that the imaging branch learned the intended
signal rather than an unrelated shortcut.

![Grad-CAM examples](results/gradcam_examples.png)

## Repository structure

```
multimodal-ehr-imaging-risk/
├── data/
│   ├── generate_data.py         # synthetic EHR + imaging dataset generator
│   ├── ehr_records.csv           # generated EHR data (4,000 records)
│   ├── images.npy                # generated imaging proxies (4000, 64, 64)
│   └── sample_images_preview.png
├── src/
│   ├── preprocessing.py          # scaling, paired train/test split
│   ├── model.py                   # dual-branch fusion architecture
│   ├── train.py                   # fusion model training (class-weighted)
│   ├── ablation.py                # EHR-only / imaging-only comparison
│   ├── evaluate.py                # metrics, confusion matrix, ROC curve
│   ├── explain_ehr.py             # SHAP explainability (EHR branch)
│   └── explain_imaging.py         # Grad-CAM explainability (imaging branch)
├── diagrams/
│   └── methodology_workflow.svg
├── prototype/
│   └── app.py                     # FastAPI multimodal prediction + explanation service
├── results/                       # generated metrics, plots, serialized model
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Generate the synthetic paired dataset
python data/generate_data.py

# 2. Train the fusion model (class-weighted loss)
python src/train.py

# 3. Train the single-modality ablation models
python src/ablation.py

# 4. Full evaluation (metrics, confusion matrix, ROC)
python src/evaluate.py

# 5. SHAP explainability (EHR branch)
python src/explain_ehr.py

# 6. Grad-CAM explainability (imaging branch)
python src/explain_imaging.py

# 7. Run the prototype API
cd prototype && uvicorn app:app --reload --port 8002
# POST to http://localhost:8002/predict — see app.py for the request schema
# (EHR fields + a base64-encoded 64x64 grayscale PNG)
```

## Known limitations

- **Synthetic data, both modalities.** Results characterize the fusion and
  explainability methodology, not real diagnostic performance. The imaging
  data is a structured synthetic proxy, not a real medical image of any kind.
- **Small image resolution (64x64).** Chosen for fast iteration during
  methodology development; a production system would use full-resolution
  imaging and a deeper, pretrained CNN backbone (e.g., a ResNet fine-tuned on
  real, credentialed chest X-ray data).
- **Late fusion only.** This project uses embedding-level late fusion. Other
  fusion strategies (early fusion, cross-attention between modalities) are
  not evaluated here and may perform differently.
- **Single synthetic cohort.** No external validation cohort or
  distribution-shift testing is included, which would be essential before
  any real-world use.

## Relationship to prior published work

This project is motivated by, and methodologically related to, prior
peer-reviewed research on machine learning for early disease diagnosis and
generative AI for medical imaging. It is an independent, hands-on framework
— built to demonstrate applied, reproducible capability in multimodal fusion
and dual explainability — rather than a replication of any specific
published study. All code, data generation, and results in this repository
are original work produced for this project.


