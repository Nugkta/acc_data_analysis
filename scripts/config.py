"""
Configuration file for ACC Survival Analysis Pipeline.
Defines paths, parameters, and common settings.

Input: None (configuration only)
Output: Config constants used by all pipeline scripts
Pos: Central configuration - imported by all analysis scripts
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_DIR = Path("/Users/stan/Documents/UOM/myn_project")

# Input data
RAW_SEER_PATH = PROJECT_DIR / "ACC数据/r分析seer/SEER纯ACC数据.xlsx"
HOSPITAL_PATH = PROJECT_DIR / "ACC数据/ACC随访资料需完善2025.2.28.xlsx"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
DATA_DIR = OUTPUT_DIR / "data"

# Figure subdirectories
NOMOGRAM_DIR = FIGURES_DIR / "nomograms"
ROC_DIR = FIGURES_DIR / "roc_curves"
CALIBRATION_DIR = FIGURES_DIR / "calibration"
KM_DIR = FIGURES_DIR / "kaplan_meier"
COMPARISON_DIR = FIGURES_DIR / "comparison"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.33  # 2:1 train:validation split

# Time points for validation (months)
TIME_POINTS = [12, 36, 60]  # 1, 3, 5 years
TIME_LABELS = ['1-year', '3-year', '5-year']

# Univariate screening threshold
UNIVARIATE_P_THRESHOLD = 0.05

# Cox model regularization
COX_PENALIZER = 0.01

# Hospital follow-up end date
from datetime import datetime
FOLLOWUP_END_DATE = datetime(2024, 10, 17)

# =============================================================================
# CANDIDATE VARIABLES
# =============================================================================
# Using separate T, N, M instead of combined TNMstage
CANDIDATE_VARS = [
    'age',
    'sex',
    'site',
    'grade',
    'radiotherapy',
    'chemotherapy',
    'tumor_number',
    'race',
    'marital_status',
    'T',  # Separate T stage
    'N',  # Separate N stage
    'M',  # Separate M stage
]

# Variables for TNM-only baseline model
TNM_ONLY_VARS = ['T', 'N', 'M']

# =============================================================================
# PLOTTING SETTINGS
# =============================================================================
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#E63946',
    'tertiary': '#2A9D8F',
    'quaternary': '#F4A261',
    'neutral': '#6C757D'
}

# =============================================================================
# COLUMN MAPPINGS (Raw SEER to processed)
# =============================================================================
SEER_COLUMN_MAP = {
    'ID': '编号',
    'age': '年龄',
    'sex': '性别',
    'site_raw': '原发部位',
    'grade': '分化级别(thru 2017)',
    'radiotherapy': '放疗',
    'chemotherapy': '化疗',
    'tumor_number': '肿瘤数量',
    'race': '种族',
    'marital_status': '婚姻',
    'urban_rural': '城乡',
    'time_os': '存活月数',
    'event_os': '生存（截止至研究日期）',
    'T': 'T',
    'N': 'N',
    'M': 'M',
    'css_classification': 'SEER cause-specific death classification',
}
