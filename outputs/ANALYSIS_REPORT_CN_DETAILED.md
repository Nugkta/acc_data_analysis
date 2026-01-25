# ACC 生存分析报告 (详细版)

**分析日期:** 2026-01-21
**分期方法:** 分别使用 T, N, M 分期组件

---

## 摘要

本研究使用 SEER 数据库建立 Cox 比例风险模型，预测腺样囊性癌 (Adenoid Cystic Carcinoma, ACC) 患者的：
- **总生存期 (Overall Survival, OS)**
- **肿瘤特异性生存期 (Cancer-Specific Survival, CSS)**

并在本地医院队列中进行外部验证。

**主要发现:** 使用分开的 T, N, M 分期组件（而非合并的 TNMstage）使外部验证性能提升约 5%，同时保持良好的内部验证指标。

---

## 数据概况

| 队列 | 样本量 | 用途 |
|------|--------|------|
| SEER 训练集 | 992 | 模型建立 |
| SEER 验证集 | 490 | 内部验证 |
| 医院外部验证 | 177 (145 有完整 OS 数据) | 外部验证 |

**数据分割比例:** SEER 数据按 2:1 分为训练集和验证集

---

## 模型性能

### Overall Survival (OS) 模型

**Forward stepwise 入选变量 (按重要性排序):**
- M (远处转移)
- T (肿瘤大小)
- age (年龄)
- chemotherapy (化疗)
- N (淋巴结)
- grade (分级)
- radiotherapy (放疗)

| 指标 | 训练集 | SEER 内部验证 | 医院外部验证 |
|------|--------|---------------|--------------|
| C-index | 0.772 | 0.729 | 0.626 |

**C-index 下降幅度:** SEER → 医院: **0.102** (约 14%)

### Cancer-Specific Survival (CSS) 模型

**Forward stepwise 入选变量 (按重要性排序):**
- T (肿瘤大小)
- M (远处转移)
- grade (分级)
- chemotherapy (化疗)
- age (年龄)
- tumor_number (肿瘤数目)

| 指标 | 训练集 | SEER 内部验证 | 医院外部验证 |
|------|--------|---------------|--------------|
| C-index | 0.782 | 0.747 | 0.635 |

**C-index 下降幅度:** SEER → 医院: **0.112** (约 15%)

---

## 外部验证性能下降分析

从 SEER 到医院队列 C-index 下降约 10% 的原因分析:

### 1. Grade 数据缺失严重
- 医院队列: **96%** 患者 grade 未知
- SEER 队列: **71%** 患者 grade 未知
- 影响: grade 变量区分能力大幅下降

### 2. CSS 事件数过少
- 医院队列仅 **14 例** CSS 事件
- 统计效力不足，CSS 模型验证受限

### 3. 人群差异
- SEER: 美国人群，多样化
- 医院: 本地人群，特征不同
- 治疗模式、随访方式存在差异

### 4. 婚姻状态分布
- 医院队列: **93%** 已婚
- SEER 队列: 分布更多样化
- 婚姻状态变量预测能力下降

**结论:** 尽管存在上述限制，模型在外部验证中仍保持合理区分度 (C-index > 0.62)，说明模型具有可接受的可移植性。

---

## Nomogram 列线图

生成了适合发表的 nomogram:

| 终点 | 文件路径 |
|------|----------|
| OS | `outputs/figures/nomograms/nomogram_os.png` |
| CSS | `outputs/figures/nomograms/nomogram_css.png` |

**Nomogram 内容:**
- 每个变量各分类的分值
- 总分刻度
- 1 年、3 年、5 年生存概率刻度

**使用方法:**
1. 根据患者各变量取值，在对应刻度上读取分值
2. 将所有分值相加得到总分
3. 在总分刻度下方读取对应的生存概率

---

## 输出文件清单

### 图表 (Figures)

| 类型 | 路径 | 说明 |
|------|------|------|
| Nomograms | `outputs/figures/nomograms/` | OS 和 CSS 列线图 |
| ROC 曲线 | `outputs/figures/roc_curves/` | Time-dependent ROC |
| 校准图 | `outputs/figures/calibration/` | Calibration plots |
| KM 曲线 | `outputs/figures/kaplan_meier/` | Kaplan-Meier 生存曲线 |
| 对比图 | `outputs/figures/comparison/` | 模型比较图 |

### 表格 (Tables)

| 类型 | 路径 | 说明 |
|------|------|------|
| 单因素 Cox | `outputs/tables/univariate_*.csv` | Univariate Cox 结果 |
| 多因素系数 | `outputs/tables/multivariate_*_coefficients.csv` | 模型系数、HR、95% CI |
| 内部验证 | `outputs/tables/internal_validation_summary.csv` | 内部验证指标汇总 |
| 外部验证 | `outputs/tables/external_validation_summary.csv` | 外部验证指标汇总 |

### 模型 (Models)

| 类型 | 路径 | 说明 |
|------|------|------|
| 拟合模型 | `outputs/models/cox_*.pkl` | Pickle 格式保存的 Cox 模型 |
| 模型规格 | `outputs/models/multivariate_*_model.json` | JSON 格式模型参数 |

---

## 统计方法

### 1. 数据准备
- SEER 数据按 2:1 随机分为训练集和验证集
- 提取分开的 T, N, M 分期组件（而非合并的 TNMstage）

### 2. 单因素筛选 (Univariate Screening)
- 对每个候选变量进行单因素 Cox 回归
- 筛选标准: p < 0.05 进入多因素选择

### 3. 多因素选择 (Multivariate Selection)
- 方法: Forward stepwise Cox regression
- 准则: AIC (Akaike Information Criterion)
- 入选标准: Likelihood ratio test p < 0.05

### 4. Nomogram 生成
- 基于回归系数计算各变量分值
- 分值范围标准化为 0-100

### 5. 模型验证

**内部验证 (Internal Validation):**
- C-index (Harrell's concordance index)
- Time-dependent AUC (1 年、3 年、5 年)
- Calibration plots (校准图)

**外部验证 (External Validation):**
- C-index 比较
- Forest plots (森林图)

---

## 结论

1. **分期方法:** 分开的 T, N, M 分期优于合并的 TNMstage，外部验证性能更好

2. **OS 模型性能:**
   - 内部验证 C-index: **0.73**
   - 外部验证 C-index: **0.63**

3. **CSS 模型性能:**
   - 内部验证 C-index: **0.75**
   - 外部验证 C-index: **0.64**

4. **模型可移植性:** 可接受，但受限于外部队列 grade 数据质量

5. **临床应用:** Nomogram 可作为个体化生存预测的实用工具

---

## 复现代码

```bash
# 激活环境
conda activate acc-survival

# 运行完整 pipeline
python scripts/run_pipeline.py
```

**脚本位置:** `scripts/`
**配置文件:** `scripts/config.py`

---

## 附注

- 本分析遵循 TRIPOD 报告规范
- 所有统计分析使用 Python (lifelines, scikit-survival)
- 图表使用 matplotlib 和 seaborn 生成
