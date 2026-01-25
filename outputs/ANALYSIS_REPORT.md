# ACC Survival Analysis Report

**Analysis Date:** 2026-01-21
**Staging Approach:** Separate T, N, M Components

---

## Executive Summary

This analysis developed Cox proportional hazards models for predicting Overall Survival (OS) and Cancer-Specific Survival (CSS) in Adenoid Cystic Carcinoma (ACC) patients using SEER data, with external validation on a local hospital cohort.

**Key Finding:** Using separate T, N, M staging components (instead of combined TNMstage) improved external validation performance by approximately 5% while maintaining strong internal validation metrics.

---

## Data Summary

| Cohort | N |
|--------|---|
| SEER Training | 992 |
| SEER Validation | 490 |
| Hospital External | 177 (145 with complete OS data) |

---

## Model Performance

### Overall Survival (OS) Model

**Selected Variables (forward stepwise):** M, T, age, chemotherapy, N, grade, radiotherapy

| Metric | Training | SEER Validation | Hospital External |
|--------|----------|-----------------|-------------------|
| C-index | 0.772 | 0.729 | 0.626 |

- C-index drop (SEER → Hospital): **0.102**

### Cancer-Specific Survival (CSS) Model

**Selected Variables (forward stepwise):** T, M, grade, chemotherapy, age, tumor_number

| Metric | Training | SEER Validation | Hospital External |
|--------|----------|-----------------|-------------------|
| C-index | 0.782 | 0.747 | 0.635 |

- C-index drop (SEER → Hospital): **0.112**

---

## External Validation Analysis

The ~10% C-index drop from SEER to hospital cohort is attributable to:

1. **Grade data collapse**: 96% of hospital patients have unknown grade vs. 71% in SEER, reducing discriminative power of the grade variable.

2. **Low CSS event count**: Only 14 CSS events in the hospital cohort limits statistical power for CSS validation.

3. **Population differences**: Different patient demographics and treatment patterns between US (SEER) and local hospital populations.

4. **Marital status homogeneity**: 93% of hospital patients are married, compared to more diverse distribution in SEER.

Despite these limitations, the models maintain reasonable discrimination (C-index > 0.62) in external validation, suggesting acceptable transportability.

---

## Nomograms

Publication-ready nomograms were generated for both OS and CSS endpoints:

- `outputs/figures/nomograms/nomogram_os.png`
- `outputs/figures/nomograms/nomogram_css.png`

Each nomogram provides:
- Point assignments for each variable category
- Total points scale
- 1-year, 3-year, and 5-year survival probability scales

---

## Output Files

### Figures
- **Nomograms:** `outputs/figures/nomograms/`
- **ROC Curves:** `outputs/figures/roc_curves/`
- **Calibration Plots:** `outputs/figures/calibration/`
- **Kaplan-Meier Curves:** `outputs/figures/kaplan_meier/`
- **Comparison Figures:** `outputs/figures/comparison/`

### Tables
- **Univariate Cox Results:** `outputs/tables/univariate_*.csv`
- **Multivariate Coefficients:** `outputs/tables/multivariate_*_coefficients.csv`
- **Validation Summary:** `outputs/tables/internal_validation_summary.csv`, `external_validation_summary.csv`

### Models
- **Fitted Cox Models:** `outputs/models/cox_*.pkl`
- **Model Specifications:** `outputs/models/multivariate_*_model.json`

---

## Methodology

1. **Data Preparation**: SEER data split 2:1 (training:validation) with separate T, N, M staging components extracted.

2. **Univariate Screening**: Cox regression for each candidate variable; variables with p < 0.05 advanced to multivariate selection.

3. **Multivariate Selection**: Forward stepwise Cox regression using AIC criterion with likelihood ratio test (p < 0.05 for entry).

4. **Nomogram Generation**: Points assigned based on regression coefficients, scaled 0-100.

5. **Validation**:
   - Internal: C-index, time-dependent AUC (1/3/5-year), calibration plots
   - External: C-index comparison, forest plots

---

## Conclusions

1. **Separate T, N, M staging** is preferable to combined TNMstage for external validation.

2. **OS Model** achieves C-index of 0.73 (internal) and 0.63 (external).

3. **CSS Model** achieves C-index of 0.75 (internal) and 0.64 (external).

4. **Model transportability** is acceptable but limited by grade data quality in external cohort.

5. **Nomograms** provide clinically useful tools for individualized survival prediction.

---

## Reproducibility

The complete analysis pipeline can be reproduced by running:

```bash
conda activate acc-survival
python scripts/run_pipeline.py
```

Scripts are located in `scripts/` with configuration in `scripts/config.py`.
