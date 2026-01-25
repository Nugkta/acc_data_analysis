#!/usr/bin/env python3
"""
04_nomogram.py - Generate nomograms for OS and CSS prediction.

Input: Fitted Cox models from outputs/models/
Output: Nomogram figures and prediction functions
Pos: Fourth step - creates clinical prediction tools

Usage: python scripts/04_nomogram.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import CoxPHFitter
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, NOMOGRAM_DIR, TIME_POINTS, TIME_LABELS,
    FIGURE_DPI, COLORS
)

# Label mappings for cleaner display
LABEL_MAP = {
    # Grade labels (Chinese to English)
    '1分化好': 'Grade I',
    '2中分化': 'Grade II',
    '3分化差': 'Grade III',
    '4未分化间变性': 'Grade IV',
    '不明': 'Unknown',
    # Age labels
    '＜45': '<45',
    '＞60': '>60',
    # Tumor number
    '＞1': '>1',
    # Treatment
    'No/Unknown': 'No/Unk',
    'No/Unknow': 'No/Unk',
}


def clean_label(label):
    """Clean and map labels for display."""
    label = str(label)
    return LABEL_MAP.get(label, label)


def create_nomogram(cph, model_spec, train_df, endpoint, output_path):
    """
    Create a publication-quality nomogram visualization for the Cox model.

    Points are normalized so all values are non-negative and the reference
    category for each variable starts at 0.
    """
    selected_vars = model_spec['selected_variables']
    coef_dict = cph.params_.to_dict()

    # Step 1: Build raw coefficient data for each variable
    var_data = {}
    for var in selected_vars:
        var_features = [(k, v) for k, v in coef_dict.items() if k.startswith(f'{var}_')]

        if not var_features:
            continue

        # Get categories and coefficients
        categories = [k.replace(f'{var}_', '') for k, v in var_features]
        coefficients = [v for k, v in var_features]

        # Find reference category (the one not in dummies)
        if hasattr(train_df[var], 'cat'):
            all_cats = [str(c) for c in train_df[var].cat.categories]
        else:
            all_cats = [str(c) for c in train_df[var].unique()]

        # Reference is the category not in the feature names
        ref_cats = [c for c in all_cats if c not in categories]
        ref_cat = ref_cats[0] if ref_cats else 'Ref'

        # Insert reference at beginning with coefficient 0
        categories = [ref_cat] + categories
        coefficients = [0.0] + coefficients

        var_data[var] = {
            'categories': categories,
            'coefficients': coefficients
        }

    # Step 2: Find the global minimum coefficient to shift all to non-negative
    all_coefs = []
    for var, data in var_data.items():
        all_coefs.extend(data['coefficients'])

    global_min = min(all_coefs)

    # Step 3: Calculate points (shift so minimum is 0, scale to reasonable range)
    # Scale factor: max coefficient range maps to 100 points
    max_coef = max(all_coefs)
    coef_range = max_coef - global_min
    scale_factor = 100 / coef_range if coef_range > 0 else 1

    # Calculate points for each variable
    nomogram_data = {}
    for var, data in var_data.items():
        points = [(c - global_min) * scale_factor for c in data['coefficients']]
        nomogram_data[var] = {
            'categories': data['categories'],
            'coefficients': data['coefficients'],
            'points': points
        }

    # Step 4: Calculate total points range
    min_total_points = sum(min(d['points']) for d in nomogram_data.values())
    max_total_points = sum(max(d['points']) for d in nomogram_data.values())

    # Step 5: Create the figure
    n_vars = len(nomogram_data)
    fig_height = max(10, 3 + n_vars * 1.0 + 4)  # Extra space for survival scales
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Layout parameters
    left_margin = 2.5
    scale_start = left_margin + 0.5
    scale_width = 10
    scale_end = scale_start + scale_width

    y_position = n_vars + 5
    row_height = 1.0

    # Draw Points scale at top (0-100)
    ax.text(left_margin - 0.2, y_position, 'Points', fontsize=11, fontweight='bold',
            va='center', ha='right')
    point_ticks = np.arange(0, 101, 10)
    for pt in point_ticks:
        x_pos = scale_start + (pt / 100) * scale_width
        ax.plot([x_pos, x_pos], [y_position - 0.15, y_position + 0.15], 'k-', linewidth=1)
        ax.text(x_pos, y_position + 0.35, f'{int(pt)}', ha='center', fontsize=9)
    ax.plot([scale_start, scale_end], [y_position, y_position], 'k-', linewidth=2)

    y_position -= 1.8

    # Draw each variable
    row_colors = ['#f5f5f5', '#ffffff']

    for i, (var, data) in enumerate(nomogram_data.items()):
        # Background stripe
        rect = mpatches.Rectangle((0, y_position - 0.5), scale_end + 1, row_height,
                                   facecolor=row_colors[i % 2], edgecolor='none', zorder=0)
        ax.add_patch(rect)

        # Variable name (clean display)
        var_display = var.replace('_', ' ').title()
        ax.text(left_margin - 0.2, y_position, var_display, fontsize=10, fontweight='bold',
                va='center', ha='right')

        # Draw scale line
        ax.plot([scale_start, scale_end], [y_position, y_position], 'k-', linewidth=1.5, zorder=1)

        # Get min/max points for this variable to position ticks
        min_pts = min(data['points'])
        max_pts = max(data['points'])
        pts_range = max_pts - min_pts

        # Draw category ticks and labels - sort by points for better layout
        sorted_items = sorted(zip(data['categories'], data['points']), key=lambda x: x[1])

        # Track positions to detect overlaps
        label_positions = []

        for j, (cat, pts) in enumerate(sorted_items):
            # Position based on 0-100 scale
            x_pos = scale_start + (pts / 100) * scale_width

            # Tick mark
            ax.plot([x_pos, x_pos], [y_position - 0.12, y_position + 0.12], 'k-', linewidth=1.5, zorder=2)

            # Clean and format category label
            cat_display = clean_label(cat)
            if len(cat_display) > 10:
                cat_display = cat_display[:8] + '..'

            # Smart label positioning - check for overlaps
            min_spacing = 0.6  # Minimum spacing in x units

            # Find appropriate y_offset based on nearby labels
            base_offsets = [-0.4, -0.7, -1.0]  # Three levels of staggering

            y_offset = base_offsets[0]
            for prev_x, prev_offset in label_positions:
                if abs(x_pos - prev_x) < min_spacing:
                    # Find next available offset level
                    for offset in base_offsets:
                        if offset != prev_offset:
                            y_offset = offset
                            break

            label_positions.append((x_pos, y_offset))

            ax.text(x_pos, y_position + y_offset, cat_display, ha='center', fontsize=8,
                    rotation=0, va='top', zorder=3)

            # Point value above tick
            ax.text(x_pos, y_position + 0.22, f'{pts:.0f}', ha='center', fontsize=6,
                    color='#666666', zorder=3)

        y_position -= row_height + 0.7

    # Draw Total Points scale
    y_position -= 0.3
    ax.text(left_margin - 0.2, y_position, 'Total Points', fontsize=11, fontweight='bold',
            va='center', ha='right')

    total_range = max_total_points - min_total_points
    n_ticks = 11
    total_ticks = np.linspace(min_total_points, max_total_points, n_ticks)

    for pt in total_ticks:
        x_pos = scale_start + ((pt - min_total_points) / total_range) * scale_width if total_range > 0 else scale_start + scale_width / 2
        ax.plot([x_pos, x_pos], [y_position - 0.15, y_position + 0.15], 'k-', linewidth=1)
        ax.text(x_pos, y_position + 0.35, f'{int(pt)}', ha='center', fontsize=9)
    ax.plot([scale_start, scale_end], [y_position, y_position], 'k-', linewidth=2)

    y_position -= 1.5

    # Draw survival probability scales
    baseline_surv = cph.baseline_survival_

    for t_idx, (t, label) in enumerate(zip(TIME_POINTS, TIME_LABELS)):
        # Get baseline survival at time t
        idx = np.abs(baseline_surv.index - t).argmin()
        s0_t = baseline_surv.iloc[idx, 0]

        y_position -= 0.3
        ax.text(left_margin - 0.2, y_position, f'{label} Survival', fontsize=10,
                fontweight='bold', va='center', ha='right')
        ax.plot([scale_start, scale_end], [y_position, y_position], 'k-', linewidth=1.5)

        # Survival probability values to show
        surv_probs = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        plotted_positions = []
        for sp in surv_probs:
            if s0_t > 0 and s0_t < 1 and sp > 0 and sp < 1:
                try:
                    # S(t) = S0(t)^exp(LP) => LP = log(log(S(t))/log(S0(t)))
                    lp = np.log(np.log(sp) / np.log(s0_t))

                    # Convert LP to total points
                    # LP = (total_points_raw) where total_points_raw is sum of coefficients
                    # We shifted by global_min and scaled by scale_factor
                    # So: LP = (displayed_total_points / scale_factor) + global_min * n_vars
                    # Solving: displayed_total_points = (LP - global_min * n_vars) * scale_factor

                    # Actually simpler: LP = sum of coefficients
                    # displayed_total_points = sum of displayed points = (sum of coeffs - n_vars * global_min) * scale_factor
                    # So: sum of coeffs = displayed_total_points / scale_factor + n_vars * global_min
                    # And: LP = sum of coeffs = displayed_total_points / scale_factor + n_vars * global_min

                    n_vars_count = len(nomogram_data)
                    total_pts = (lp - n_vars_count * global_min) * scale_factor

                    if min_total_points <= total_pts <= max_total_points:
                        x_pos = scale_start + ((total_pts - min_total_points) / total_range) * scale_width if total_range > 0 else scale_start + scale_width / 2

                        # Check if too close to existing label
                        too_close = any(abs(x_pos - px) < 0.4 for px in plotted_positions)

                        if scale_start <= x_pos <= scale_end and not too_close:
                            ax.plot([x_pos, x_pos], [y_position - 0.12, y_position + 0.12], 'k-', linewidth=1)
                            ax.text(x_pos, y_position - 0.35, f'{sp:.2f}', ha='center', fontsize=8, va='top')
                            plotted_positions.append(x_pos)
                except:
                    pass

        y_position -= row_height

    # Styling
    ax.set_xlim(-0.5, scale_end + 1)
    ax.set_ylim(y_position - 1, n_vars + 6)
    ax.axis('off')

    # Title
    endpoint_name = 'Overall Survival' if endpoint == 'os' else 'Cancer-Specific Survival'
    ax.set_title(f'{endpoint_name} Nomogram\n(Separate T, N, M Staging)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    return nomogram_data


def main():
    print("=" * 70)
    print("04. NOMOGRAM GENERATION")
    print("=" * 70)

    # Ensure output directory exists
    NOMOGRAM_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_pickle(DATA_DIR / 'train.pkl')

    for endpoint in ['os', 'css']:
        print(f"\n{'=' * 70}")
        print(f"NOMOGRAM - {endpoint.upper()}")
        print(f"{'=' * 70}")

        # Load model
        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'r') as f:
            model_spec = json.load(f)

        with open(MODELS_DIR / f'cox_{endpoint}_model.pkl', 'rb') as f:
            cph = pickle.load(f)

        print(f"\nVariables: {model_spec['selected_variables']}")
        print(f"C-index: {model_spec['performance']['training_c_index']:.4f}")

        # Create nomogram
        output_path = NOMOGRAM_DIR / f'nomogram_{endpoint}.png'
        nomogram_data = create_nomogram(cph, model_spec, train_df, endpoint, output_path)

        print(f"\nNomogram saved: {output_path}")

        # Print point summary
        print("\nPoint assignments:")
        for var, data in nomogram_data.items():
            print(f"  {var}:")
            for cat, pts in zip(data['categories'], data['points']):
                print(f"    {cat}: {pts:.1f} points")

        # Save nomogram data for reference
        with open(MODELS_DIR / f'nomogram_{endpoint}_data.json', 'w') as f:
            json.dump(nomogram_data, f, indent=2, default=str)

    print(f"\n--- Output Files ---")
    print(f"  Nomograms: {NOMOGRAM_DIR}")
    print("\n✓ Nomogram generation complete!")


if __name__ == '__main__':
    main()
