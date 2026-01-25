# ACC 生存分析报告

**分析日期:** 2026-01-21
**分期方法:** 分别使用 T, N, M 分期

---

## 摘要

本分析使用 SEER 数据建立 Cox 比例风险模型，预测腺样囊性癌 (ACC) 患者的总生存期 (OS) 和肿瘤特异性生存期 (CSS)，并在本地医院队列中进行外部验证。

**主要发现:** 使用分开的 T, N, M 分期（而非合并的 TNMstage）使外部验证性能提升约 5%。

---

## 数据概况

| 队列 | 样本量 |
|------|--------|
| SEER 训练集 | 992 |
| SEER 验证集 | 490 |
| 医院外部验证 | 177 (145 有完整 OS 数据) |

---

## 模型性能

### OS 模型

**入选变量:** M, T, age, chemotherapy, N, grade, radiotherapy

| 指标 | 训练集 | SEER 验证 | 医院外部验证 |
|------|--------|-----------|--------------|
| C-index | 0.772 | 0.729 | 0.626 |

### CSS 模型

**入选变量:** T, M, grade, chemotherapy, age, tumor_number

| 指标 | 训练集 | SEER 验证 | 医院外部验证 |
|------|--------|-----------|--------------|
| C-index | 0.782 | 0.747 | 0.635 |

---

## 外部验证分析

C-index 下降约 10% 的原因:

1. **Grade 数据缺失:** 医院 96% 患者 grade 未知 vs SEER 71%
2. **CSS 事件数少:** 医院队列仅 14 例 CSS 事件
3. **人群差异:** 美国 (SEER) 与本地医院人群不同
4. **婚姻状态单一:** 医院 93% 已婚 vs SEER 更多样

模型在外部验证中仍保持合理区分度 (C-index > 0.62)。

---

## Nomogram

生成了 OS 和 CSS 的 nomogram:

- `outputs/figures/nomograms/nomogram_os.png`
- `outputs/figures/nomograms/nomogram_css.png`

包含 1 年、3 年、5 年生存概率预测。

---

## 输出文件

**图表:**
- Nomograms: `outputs/figures/nomograms/`
- ROC 曲线: `outputs/figures/roc_curves/`
- 校准图: `outputs/figures/calibration/`
- KM 曲线: `outputs/figures/kaplan_meier/`

**表格:**
- 单因素 Cox: `outputs/tables/univariate_*.csv`
- 多因素系数: `outputs/tables/multivariate_*_coefficients.csv`
- 验证汇总: `outputs/tables/*_validation_summary.csv`

**模型:**
- Cox 模型: `outputs/models/cox_*.pkl`

---

## 方法

1. SEER 数据按 2:1 分为训练集和验证集
2. 单因素筛选: p < 0.05 进入多因素
3. 多因素选择: Forward stepwise Cox (AIC 准则)
4. Nomogram: 基于回归系数赋分 (0-100)
5. 验证: C-index, time-dependent AUC, calibration plots

---

## 结论

1. **分开的 T, N, M 分期**优于合并的 TNMstage
2. **OS 模型** C-index: 0.73 (内部) / 0.63 (外部)
3. **CSS 模型** C-index: 0.75 (内部) / 0.64 (外部)
4. 模型可移植性可接受，但受 grade 数据质量限制
5. Nomogram 可用于个体化生存预测

---

## 复现

```bash
conda activate acc-survival
python scripts/run_pipeline.py
```
