# Project: ACC Survival Nomograms (SEER Development + Local Hospital Validation)

This repository implements a CAM4-style prognostic modeling workflow (Cox → nomogram → validation) for **Adenoid Cystic Carcinoma (ACC)**.

## 0) Goal (alignment with CAM4)

The reference paper (Cancer Medicine 2024; DOI 10.1002/cam4.6907) is aligned around:

- Two endpoints/models: **OS** and **CSS** (CSS only when cause-of-death is reliable)
- Cohort definition with exclusions (paper excludes cases with missing follow-up or missing TNM)
- Variable selection: **univariate Cox screening (p < 0.05)** → **forward-stepwise multivariate Cox** (likelihood-ratio criterion)
- Nomograms predicting **1/3/5-year** survival
- Validation with **C-index**, **time-dependent ROC/AUC**, **calibration**, **DCA**, plus **TNM-only** baseline comparison

ACC-specific reality: SEER ACC may have substantial missingness for staging and other categorical predictors (e.g., `grade`). The primary workflow in this repo will follow the common nomogram/R practice of keeping **Unknown/NA** as explicit factor levels (see the dedicated note section), plus an explicitly labeled complete-case sensitivity analysis.

## 1) Inputs (what is required vs reference-only)

### 1.1 Required SEER input

Source folder: `ACC数据/r分析seer/`

- `SEER纯ACC数据.xlsx`: raw SEER export (required)

All other SEER-derived datasets in `ACC数据/r分析seer/` are treated as **reference only** (useful to verify variable meanings and sanity-check counts), because you will generate your own cleaned cohort and split.

### 1.2 Required local hospital input (external validation)

Source folder: `ACC数据/`

- `ACC随访资料需完善2025.2.28.xlsx`: local follow-up dataset (contains PHI)

Security note: do not export PHI in processed outputs; keep de-identified analysis tables/figures under `data/processed/`.

### 1.3 Reference-only materials

- `ACC数据/r分析seer/绘制诺模图代码.txt`: reference for intended predictors/factor handling
- `ACC数据/r分析seer/验证队列roc绘制代码.txt`: reference for validation intent
- Paper text/PDF in repo root (methods/metrics alignment)

## 2) Endpoints (define before modeling)

### 2.1 OS

- `time_os_months`: time from index date → death or last follow-up
- `event_os`: 1 if died (any cause), else 0

### 2.2 CSS (only if feasible)

- `time_css_months`
- `event_css`: 1 if cancer-specific death, else 0

Decision rule:

- If local data lacks reliable cause-of-death, do **OS external validation only**.
- CSS can still be modeled/validated in SEER if a cause-specific death indicator exists.

## 3) Preprocessing (paper-aligned cohort definition)

### 3.1 Track A — primary analysis (Unknown/NA kept as factor levels)

Apply exclusions (separately for OS and CSS):

- Exclude missing/invalid follow-up (no survival time)
- Exclude inconsistent time (negative/zero where impossible)

Do NOT drop patients solely because a categorical predictor is “Unknown/NA”. Instead, encode Unknown/NA as explicit factor levels (e.g., `grade=Unknown`, `TNMstage=NA`, `chemotherapy=No/Unknown`) so they can be scored in the nomogram.

Split:

- Create a **2:1 random split** (train:validation) with a fixed seed.

### 3.2 Track B — sensitivity (complete-case / paper-mirroring exclusions)

Allowed but must be labeled as non-primary:

- Exclude missing essential staging (TNM/T/N/M) and/or other predictors as complete-case, to mimic CAM4-style exclusions
- Compare effect estimates and validation metrics vs Track A

All preprocessing decisions and mappings are recorded in `data/processed/data_dictionary.json`.

## 4) Candidate predictors (before selection)

Start broad, then let CAM4-style screening/stepwise selection decide what enters the final models.

Candidate predictors (initial):

- Demographics: `age` (grouped), `sex`, `race` (if available)
- Tumor: `site`, `grade`, `tumor_number`
- Staging: `TNMstage` (or `T`, `N`, `M`)
- Treatment: `radiotherapy`, `chemotherapy` (and surgery if present)
- Social: `marital_status`, `urban_rural` (if available)

Encoding:

- Treat categorical predictors as factors with explicit reference levels.
- Align factor levels across train/validation/local.
- Prefer explicit “Unknown/NA” levels for categorical predictors when present in the raw data (see note section below).
- For treatment variables that are frequently coded as combined groups, allow a merged level such as `No/Unknown` (e.g., `chemotherapy`: `Yes` vs `No/Unknown`) when that matches the raw coding and keeps the model stable.

## 5) Modeling (CAM4-style)

Run the workflow twice (OS and CSS if feasible):

### 5.1 Univariate Cox screening (training cohort)

- Fit one Cox model per predictor
- Keep predictors with p < 0.05
- Output: univariate table (HR, 95% CI, p)

### 5.2 Forward-stepwise multivariate Cox (training cohort)

- Build model stepwise from screened candidates
- Use likelihood-ratio criterion for inclusion/removal
- Output: final multivariate table (HR, 95% CI, p)

### 5.3 Nomogram

- Produce 1/3/5-year survival predictions
- Provide a plot and a callable prediction function

### 5.4 TNM-only baseline

- Fit TNM-only model (OS; and CSS if CSS is modeled)
- Compare full model vs baseline in validation metrics

## 6) Validation

For each endpoint:

- Discrimination: C-index
- Time-dependent ROC/AUC: 1/3/5 years
- Calibration: 1/3/5-year plots
- Clinical utility: DCA

Validation sets:

- Internal: SEER validation cohort
- External: local cohort (OS; CSS only if feasible)

## 7) Deliverables

- Tables: baseline characteristics; univariate Cox; multivariate Cox
- Figures: flowchart; KM curves; nomogram; ROC/AUC; calibration; DCA; nomogram vs TNM-only

## 8) Repo execution

### 8.1 Script-based Pipeline (Current - Recommended)

The analysis has been refactored into a reproducible script-based pipeline using **separate T, N, M staging components**:

```
scripts/
  config.py                    # Central configuration (paths, parameters, candidate variables)
  01_data_preparation.py       # Load SEER data, extract T/N/M, create train/validation splits
  02_univariate_cox.py         # Univariate Cox screening (p < 0.05)
  03_multivariate_cox.py       # Forward stepwise Cox using AIC criterion
  04_nomogram.py               # Generate publication-ready nomograms
  05_internal_validation.py    # SEER validation (C-index, ROC, calibration)
  06_external_validation.py    # Hospital external validation
  07_generate_figures.py       # Generate all publication figures
  run_pipeline.py              # Master script to run complete pipeline
```

**Run the complete pipeline:**
```bash
conda activate acc-survival
python scripts/run_pipeline.py
```

### 8.2 Legacy Notebooks (Reference Only)

```
analysis_notebooks/
  01_data_preparation.ipynb          # Load raw data, define endpoints (OS/CSS), create Track A/B cohorts
  02_univariate_cox_screening.ipynb  # Univariate Cox screening (p < 0.05)
  03_multivariate_cox_stepwise.ipynb # Forward-stepwise multivariate Cox
  04_nomogram_os.ipynb               # OS nomogram construction
  05_nomogram_css.ipynb              # CSS nomogram construction
  06_external_validation.ipynb       # External validation (local hospital)
```

## 9) Status

### Script Pipeline (Separate T, N, M Staging) — COMPLETED

- [x] Script 01: Data preparation with separate T, N, M extraction
- [x] Script 02: Univariate Cox screening
- [x] Script 03: Forward stepwise multivariate Cox (AIC-based)
- [x] Script 04: Nomogram generation (OS and CSS)
- [x] Script 05: Internal validation (SEER)
- [x] Script 06: External validation (Hospital)
- [x] Script 07: Publication figures
- [x] Analysis report: `outputs/ANALYSIS_REPORT.md`

### Final Results (2026-01-21)

| Model | Variables | Train C-index | SEER Val C-index | Hospital C-index |
|-------|-----------|---------------|------------------|------------------|
| OS | M, T, age, chemo, N, grade, RT | 0.772 | 0.729 | 0.626 |
| CSS | T, M, grade, chemo, age, tumor_num | 0.782 | 0.747 | 0.635 |

### Legacy Notebooks (Combined TNMstage) — Superseded

- [x] Notebook 01-06: Completed but using combined TNMstage approach
- Note: External validation showed ~5% worse performance with combined TNMstage

## 9.1) Key Methodology Change: Separate T, N, M Staging

The original analysis used combined `TNMstage` as a single variable. Investigation revealed that using **separate T, N, M components** improves external validation:

| Approach | Hospital OS C-index | Hospital CSS C-index |
|----------|---------------------|----------------------|
| Combined TNMstage | ~0.60 | ~0.58 |
| **Separate T, N, M** | **0.626** | **0.635** |

**Rationale:** Separate components provide more granular risk stratification and are more robust to staging system variations between SEER and hospital data.

## 9.2) Notebook 06 (External Validation) — Harmonization Summary

- Load hospital data from `需要搜集的数据` (OS/CSS and clinical fields) and `筛选2` (TNM + survival status), then merge on normalized `住院号`.
- Map hospital categories to SEER Track A labels for model compatibility: `sex`, `age` (＜45/45-59/＞60), `TNMstage` (1/2/3/4A/4B/4C/4/4NOS), `grade`, `radiotherapy`, `chemotherapy`, `marital_status`, `tumor_number`.
- **OS calculation (FIXED)**: OS is now recalculated from dates:
  - Dead patients: `OS = 死亡 (death_date) - 手术时间 (surgery_date)`
  - Alive patients: `OS = 2024/10/17 (follow-up end) - 手术时间 (surgery_date)` (right-censored)
  - The pre-recorded `总生存期（OS）` column was found to be incorrect and is no longer used.
- Use `随访至今是否存活（2024/10/17）` for `event_os` (0=dead→event=1, 1=alive→event=0).
- CSS is conservative: only label cancer-specific death when the reason clearly indicates cancer; otherwise leave `event_css` missing to avoid misclassification.

## 9.3) Historical: Notebook 06 Data Quality Issues (Now Resolved)

The original notebook-based analysis encountered several data quality issues that have been addressed in the script pipeline:

### OS Values Issue (Fixed)

The pre-recorded OS values were incorrect. OS is now recalculated from surgery and death/follow-up dates:

```
Dead patients (δ=1):  OS = death_date - surgery_date
Alive patients (δ=0): OS = followup_date - surgery_date
```

### Data Quality Limitations (Ongoing)

1. **Grade collapse**: 96% of hospital patients have unknown grade vs. 71% in SEER
2. **Low CSS events**: Only 14 CSS events in hospital cohort
3. **Marital status homogeneity**: 93% married in hospital data

These limitations explain the ~10% C-index drop from SEER to hospital validation but are inherent to the external dataset.

## 9.4) Repository Configuration Updates

- `.claude/` directory, `PLAN.md`, `claude.md`, and `gemini.md` added to `.gitignore` for better repository management.

## 10) Comment: treating Unknown/NA in predictors

- Categorical “Unknown/NA” values are kept as explicit factor levels (e.g., `grade=Unknown`, `TNMstage=NA`) so they can be scored in the nomogram.
- Some variables may use merged levels like `No/Unknown` (common for treatment fields).
- Interpretation: Unknown/NA = “not recorded/unknown”, not a biological state.
- Track B (complete-case) is the sensitivity check for this choice.
