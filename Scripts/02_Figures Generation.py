#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Final publication-quality phytoplankton relative-composition box plot.

Correct data:
    *_REL columns from 01_Community_Composition.csv

Plot:
    Box        = Q1–Q3
    Black line = Median
    Whiskers   = 1.5 × IQR
    Dark dots  = Monthly observations
    White diamond = Mean
    Text above = Mean ± SD

Complete descriptive statistics are also saved separately:
    N, Mean, SD, Median, Min, Q1, Q3, Max
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

DATA_FILE = (
    ROOT
    / "Results"
    / "01_State_Discovery"
    / "01_Community_Composition.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_03"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "01_State_Discovery"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. CORRECT RELATIVE-COMPOSITION COLUMNS
# =============================================================================

REL_COLUMNS = [
    "DIATO_REL",
    "DINO_REL",
    "HAPTO_REL",
    "GREEN_REL",
    "PROKAR_REL",
    "PROCHLO_REL",
]

DISPLAY_LABELS = [
    "DIATO",
    "DINO",
    "HAPTO",
    "GREEN",
    "PROKAR",
    "PROCHLO",
]


# =============================================================================
# 3. COLOURS
# =============================================================================

COLORS = [
    "#D66A5E",   # DIATO
    "#E3A33C",   # DINO
    "#72A96B",   # HAPTO
    "#4FA6A1",   # GREEN
    "#657DB7",   # PROKAR
    "#9B70AE",   # PROCHLO
]


# =============================================================================
# 4. LARGE PUBLICATION FONT SETTINGS
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 40,

    "xtick.labelsize": 32,
    "ytick.labelsize": 31,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 5. LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_FILE)


missing = [
    col for col in REL_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing relative-composition columns: {missing}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# 6. EXTRACT RELATIVE COMPOSITION AND CONVERT TO %
# =============================================================================

community = df[REL_COLUMNS].copy()


for col in REL_COLUMNS:

    community[col] = pd.to_numeric(
        community[col],
        errors="coerce"
    )


# Relative values were stored as fractions (0–1).
community = community * 100.0


# Rename for easier reporting.
community.columns = DISPLAY_LABELS


# =============================================================================
# 7. IMPORTANT VALIDATION
# =============================================================================

monthly_sum = community.sum(axis=1)


print("\nRelative-composition validation")
print("=" * 70)

print(
    f"Mean monthly total : {monthly_sum.mean():.6f}%"
)

print(
    f"Minimum total      : {monthly_sum.min():.6f}%"
)

print(
    f"Maximum total      : {monthly_sum.max():.6f}%"
)


if not np.allclose(
    monthly_sum,
    100.0,
    atol=1e-5
):
    print(
        "\nWARNING: Monthly relative-composition values "
        "do not sum exactly to 100%."
    )


# =============================================================================
# 8. DESCRIPTIVE STATISTICS
# =============================================================================

stats = pd.DataFrame(
    index=DISPLAY_LABELS
)


stats["N"] = community.count()

stats["Mean"] = community.mean()

stats["SD"] = community.std(ddof=1)

stats["Median"] = community.median()

stats["Min"] = community.min()

stats["Q1"] = community.quantile(0.25)

stats["Q3"] = community.quantile(0.75)

stats["Max"] = community.max()


stats.index.name = "Phytoplankton_Group"


STATS_FILE = (
    RESULT_DIR
    / "Phytoplankton_Relative_Composition_Descriptive_Statistics.csv"
)


stats.to_csv(
    STATS_FILE,
    float_format="%.4f"
)


print("\n")
print("=" * 95)

print(
    "PHYTOPLANKTON RELATIVE-COMPOSITION "
    "DESCRIPTIVE STATISTICS (%)"
)

print("=" * 95)

print(
    stats.round(2).to_string()
)


# =============================================================================
# 9. PREPARE DATA
# =============================================================================

plot_data = [

    community[group]
    .dropna()
    .to_numpy()

    for group in DISPLAY_LABELS
]


positions = np.arange(
    1,
    len(DISPLAY_LABELS) + 1
)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14.5, 9.2)
)


# =============================================================================
# 11. BOXPLOT
#
# Standard Tukey whiskers (1.5 × IQR).
# Min / Max remain available in the statistics table.
# =============================================================================

box = ax.boxplot(

    plot_data,

    positions=positions,

    widths=0.58,

    patch_artist=True,

    showfliers=False,

    whis=1.5,

    medianprops={
        "color": "#202020",
        "linewidth": 3.2
    },

    whiskerprops={
        "color": "#303030",
        "linewidth": 2.0
    },

    capprops={
        "color": "#303030",
        "linewidth": 2.0
    },

    boxprops={
        "edgecolor": "#303030",
        "linewidth": 1.8
    }
)


for patch, color in zip(
    box["boxes"],
    COLORS
):

    patch.set_facecolor(color)

    patch.set_alpha(0.82)


# =============================================================================
# 12. MONTHLY OBSERVATIONS
# =============================================================================

rng = np.random.default_rng(42)


for i, values in enumerate(
    plot_data,
    start=1
):

    jitter = rng.normal(
        loc=i,
        scale=0.055,
        size=len(values)
    )

    ax.scatter(

        jitter,
        values,

        s=38,

        facecolor="#333333",

        edgecolor="white",

        linewidth=0.40,

        alpha=0.36,

        zorder=3
    )


# =============================================================================
# 13. MEAN DIAMONDS
# =============================================================================

for i, group in enumerate(
    DISPLAY_LABELS,
    start=1
):

    mean_value = stats.loc[
        group,
        "Mean"
    ]

    ax.scatter(

        i,
        mean_value,

        marker="D",

        s=180,

        facecolor="white",

        edgecolor="#202020",

        linewidth=2.1,

        zorder=6
    )


# =============================================================================
# 14. MEAN ± SD LABELS
# =============================================================================

global_max = community.max().max()


# Small proportional offset rather than the previous fixed +1.
annotation_offset = global_max * 0.035


for i, group in enumerate(
    DISPLAY_LABELS,
    start=1
):

    mean_value = stats.loc[
        group,
        "Mean"
    ]

    sd_value = stats.loc[
        group,
        "SD"
    ]

    max_value = stats.loc[
        group,
        "Max"
    ]


    label = (
        f"{mean_value:.1f} ± {sd_value:.1f}"
    )


    ax.text(

        i,

        max_value + annotation_offset,

        label,

        ha="center",
        va="bottom",

        fontsize=25,

        color="#202020"
    )


# =============================================================================
# 15. AXES
# =============================================================================

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    DISPLAY_LABELS
)


ax.set_ylabel(
    "Relative composition (%)",
    labelpad=15
)

ax.set_xlabel("")


# Enough space only for annotations.
ymax = (
    global_max
    + annotation_offset * 3.0
)


# If actual data fit comfortably below 100,
# don't unnecessarily force a full 0–100 range.
ymax = min(
    max(ymax, 60),
    100
)


ax.set_ylim(
    0,
    ymax
)


ax.tick_params(
    axis="x",
    labelsize=32,
    pad=10
)

ax.tick_params(
    axis="y",
    labelsize=31,
    pad=6
)


# =============================================================================
# 16. GRID AND SPINES
# =============================================================================

ax.grid(

    axis="y",

    linewidth=0.75,

    alpha=0.18
)

ax.set_axisbelow(True)


ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)


ax.spines["left"].set_linewidth(1.4)

ax.spines["bottom"].set_linewidth(1.4)


# =============================================================================
# 17. LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 18. SAVE
# =============================================================================

TIFF_FILE = (
    OUTPUT_DIR
    / "Phytoplankton_Relative_Composition_Boxplot_Final.tiff"
)

PDF_FILE = (
    OUTPUT_DIR
    / "Phytoplankton_Relative_Composition_Boxplot_Final.pdf"
)

SVG_FILE = (
    OUTPUT_DIR
    / "Phytoplankton_Relative_Composition_Boxplot_Final.svg"
)


fig.savefig(

    TIFF_FILE,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    PDF_FILE,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


fig.savefig(

    SVG_FILE,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(fig)


# =============================================================================
# 19. OUTPUT
# =============================================================================

print("\nFigure saved:")

print(f"TIFF : {TIFF_FILE}")
print(f"PDF  : {PDF_FILE}")
print(f"SVG  : {SVG_FILE}")

print(
    f"\nStatistics table:\n{STATS_FILE}"
)


# In[5]:


"""
Publication-quality horizontal boxplot for environmental variables.

Style matched to the phytoplankton boxplot:
- horizontal layout
- z-score scale on x-axis
- coloured boxplots + jittered points
- mean diamond marker
- mean ± SD written directly near each variable row
- no separate side panel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "Data"
    / "04_Final_Current_Environmental_Modeling_Data.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_03"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "02_Environmental_Analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FINAL ENVIRONMENTAL VARIABLES
# =============================================================================

FEATURES = [
    "SST",
    "NO3",
    "PO4",
    "SPCo2",
    "MLD",
    "SSS",
    "SSH",
    "PAR",
    "PDO",
    "NINO_3.4",
    "WPI",
    "MHW_MeanInt",
    "MHW_MaxInt",
    "MHW_CumInt",
]


# =============================================================================
# 3. DISPLAY LABELS
# =============================================================================

DISPLAY_LABELS = {
    "SST": "SST",
    "NO3": r"NO$_3$",
    "PO4": r"PO$_4$",
    "SPCo2": r"$p$CO$_2$",
    "MLD": "MLD",
    "SSS": "SSS",
    "SSH": "SSH",
    "PAR": "PAR",
    "PDO": "PDO",
    "NINO_3.4": "Niño 3.4",
    "WPI": "WPI",
    "MHW_MeanInt": r"MHW$_{\mathrm{mean}}$",
    "MHW_MaxInt": r"MHW$_{\mathrm{max}}$",
    "MHW_CumInt": r"MHW$_{\mathrm{cum}}$",
}


# =============================================================================
# 4. COLOURS
# =============================================================================

COLORS = [
    "#D98578",   # SST
    "#D6A84E",   # NO3
    "#7CB26D",   # PO4
    "#7EB7B5",   # pCO2
    "#8093C1",   # MLD
    "#7EC5D0",   # SSS
    "#90B4D8",   # SSH
    "#E3C364",   # PAR
    "#A58AD1",   # PDO
    "#B58FD6",   # Niño 3.4
    "#7A93CF",   # WPI
    "#D8A07F",   # MHWmean
    "#D98787",   # MHWmax
    "#C96F87",   # MHWcum
]


# =============================================================================
# 5. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 28,
    "axes.labelsize": 36,
    "xtick.labelsize": 24,
    "ytick.labelsize": 25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 6. LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_FILE)

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(
        f"Missing environmental variables: {missing}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )

env = df[FEATURES].copy()

for feature in FEATURES:
    env[feature] = pd.to_numeric(env[feature], errors="coerce")


# =============================================================================
# 7. DESCRIPTIVE STATISTICS IN ORIGINAL UNITS
# =============================================================================

stats = pd.DataFrame(index=FEATURES)
stats["N"] = env.count()
stats["Mean"] = env.mean()
stats["SD"] = env.std(ddof=1)
stats["Median"] = env.median()
stats["Min"] = env.min()
stats["Q1"] = env.quantile(0.25)
stats["Q3"] = env.quantile(0.75)
stats["Max"] = env.max()
stats.index.name = "Environmental_Variable"

stats_file = RESULT_DIR / "Environmental_Descriptive_Statistics.csv"
stats.to_csv(stats_file, float_format="%.6f")

print("\n" + "=" * 95)
print("ENVIRONMENTAL VARIABLE DESCRIPTIVE STATISTICS")
print("=" * 95)
print(stats.round(3).to_string())


# =============================================================================
# 8. STANDARDIZE VARIABLES (Z-SCORE)
# =============================================================================

env_z = pd.DataFrame(index=env.index)

for feature in FEATURES:
    mu = env[feature].mean()
    sd = env[feature].std(ddof=1)

    if np.isclose(sd, 0):
        raise ValueError(f"{feature} has zero standard deviation.")

    env_z[feature] = (env[feature] - mu) / sd


# =============================================================================
# 9. PREPARE PLOT DATA
# =============================================================================

plot_features = FEATURES[::-1]
plot_data = [env_z[f].dropna().to_numpy() for f in plot_features]
plot_labels = [DISPLAY_LABELS[f] for f in plot_features]
plot_colors = COLORS[::-1]
positions = np.arange(1, len(plot_features) + 1)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(15.5, 11.0))


# =============================================================================
# 11. HORIZONTAL BOXPLOTS
# =============================================================================

box = ax.boxplot(
    plot_data,
    vert=False,
    positions=positions,
    widths=0.58,
    patch_artist=True,
    showfliers=False,
    whis=1.5,
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "#222222",
        "markeredgewidth": 2.2,
        "markersize": 14
    },
    medianprops={"color": "#222222", "linewidth": 2.7},
    whiskerprops={"color": "#444444", "linewidth": 1.8},
    capprops={"color": "#444444", "linewidth": 1.8},
    boxprops={"edgecolor": "#444444", "linewidth": 1.9},
)

for patch, color in zip(box["boxes"], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.88)


# =============================================================================
# 12. OVERLAY JITTERED POINTS
# =============================================================================

rng = np.random.default_rng(42)

for y_pos, values in zip(positions, plot_data):
    jitter = rng.normal(loc=y_pos, scale=0.11, size=len(values))

    ax.scatter(
        values,
        jitter,
        s=24,
        color="#303030",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.33,
        zorder=3
    )


# =============================================================================
# 13. ZERO REFERENCE LINE
# =============================================================================

ax.axvline(
    0,
    color="#666666",
    linestyle="--",
    linewidth=1.4,
    alpha=0.70,
    zorder=1
)


# =============================================================================
# 14. AXES
# =============================================================================

ax.set_yticks(positions)
ax.set_yticklabels(plot_labels)

ax.set_xlabel("Standardized value (z-score)", labelpad=12)
ax.set_ylabel("")

ax.tick_params(axis="y", pad=8)
ax.tick_params(axis="x", pad=6)


# =============================================================================
# 15. X LIMITS
# =============================================================================

all_z = env_z.to_numpy().ravel()
all_z = all_z[np.isfinite(all_z)]

left_lim = np.floor(all_z.min() - 0.5)
right_lim = np.ceil(all_z.max() + 1.2)

ax.set_xlim(left_lim, right_lim)


# =============================================================================
# 16. GRID / SPINES
# =============================================================================

ax.grid(axis="x", color="#D3D3D3", linewidth=0.7, alpha=0.45)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)


# =============================================================================
# 17. FORMAT NUMBERS
# =============================================================================

def pretty_number(value):
    av = abs(value)

    if av >= 100:
        return f"{value:.1f}"
    elif av >= 10:
        return f"{value:.2f}"
    elif av >= 1:
        return f"{value:.2f}"
    elif av >= 0.01:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"


# =============================================================================
# 18. WRITE MEAN ± SD ON EACH ROW
# =============================================================================

for y_pos, feature, values in zip(positions, plot_features, plot_data):
    mean_value = stats.loc[feature, "Mean"]
    sd_value = stats.loc[feature, "SD"]

    label = f"{pretty_number(mean_value)} ± {pretty_number(sd_value)}"

    x_text = np.nanpercentile(values, 97) + 0.35
    x_text = min(x_text, right_lim - 0.25)

    ax.text(
        x_text,
        y_pos + 0.34,
        label,
        ha="left",
        va="bottom",
        fontsize=21,
        color="#222222"
    )


# =============================================================================
# 19. SAVE FIGURE
# =============================================================================

tiff_file = OUTPUT_DIR / "Environmental_Variables_Boxplot_Final.tiff"
pdf_file = OUTPUT_DIR / "Environmental_Variables_Boxplot_Final.pdf"
svg_file = OUTPUT_DIR / "Environmental_Variables_Boxplot_Final.svg"

fig.tight_layout(pad=0.9)

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

fig.savefig(
    svg_file,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 20. OUTPUT
# =============================================================================

print("\nFigure saved successfully:")
print(f"TIFF : {tiff_file}")
print(f"PDF  : {pdf_file}")
print(f"SVG  : {svg_file}")

print(f"\nStatistics table:\n{stats_file}")


# In[1]:


"""Create Main Figure 1(a): final phytoplankton analysis workflow.

The script creates one standalone publication panel for later assembly in
PowerPoint. It saves a 1000-dpi TIFF and a vector PDF without reading or
recalculating any analysis results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# =============================================================================
# 1. OUTPUT AND STYLE SETTINGS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = "Figure_1a_Analysis_Workflow"
TIFF_DPI = 1000
ADD_PANEL_LABEL = True

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})


# =============================================================================
# 2. DRAWING HELPERS
# =============================================================================
def add_box(axis, center_x, center_y, width, height, title, details,
            facecolor, edgecolor, title_color="#17202A"):
    """Draw one rounded workflow box."""
    left = center_x - width / 2
    bottom = center_y - height / 2
    box = FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor,
        transform=axis.transAxes, clip_on=False
    )
    axis.add_patch(box)
    axis.text(
        center_x, center_y + height * 0.20, title,
        ha="center", va="center", fontsize=13.2,
        fontweight="bold", color=title_color,
        transform=axis.transAxes
    )
    axis.text(
        center_x, center_y - height * 0.13, details,
        ha="center", va="center", fontsize=10.8,
        color="#263238", linespacing=1.28,
        transform=axis.transAxes
    )
    return box


def add_arrow(axis, start, end, color="#566573", connectionstyle="arc3"):
    """Draw a consistent directional arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15,
        linewidth=1.6, color=color, connectionstyle=connectionstyle,
        transform=axis.transAxes, clip_on=False
    )
    axis.add_patch(arrow)


# =============================================================================
# 3. CREATE THE WORKFLOW PANEL
# =============================================================================
fig, ax = plt.subplots(figsize=(12.0, 6.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Restrained colour palette: blue = community processing,
# green = final ecological states, orange = environmental prediction.
blue_fill = "#EAF2F8"
blue_edge = "#2E86C1"
green_fill = "#E9F7EF"
green_edge = "#239B56"
orange_fill = "#FEF1E6"
orange_edge = "#D97706"
purple_fill = "#F3ECF9"
purple_edge = "#7D3C98"

# Section labels
ax.text(
    0.025, 0.935, "ECOLOGICAL-STATE CONSTRUCTION",
    fontsize=13.5, fontweight="bold", color=blue_edge,
    ha="left", va="center", transform=ax.transAxes
)
ax.plot(
    [0.025, 0.975], [0.905, 0.905], color="#D5D8DC", linewidth=1.0,
    transform=ax.transAxes, clip_on=False
)

# Upper workflow row
upper_y = 0.70
upper_x = [0.10, 0.30, 0.50, 0.70, 0.90]
box_w, box_h = 0.165, 0.235

add_box(
    ax, upper_x[0], upper_y, box_w, box_h,
    "Monthly community data",
    "240 months (2003–2022)\nSix phytoplankton groups",
    blue_fill, blue_edge
)
add_box(
    ax, upper_x[1], upper_y, box_w, box_h,
    "Compositional processing",
    "Relative abundances\nCentered log-ratio (CLR)",
    blue_fill, blue_edge
)
add_box(
    ax, upper_x[2], upper_y, box_w, box_h,
    "Temporal detrending",
    "Group-wise linear trend\nremoved from CLR values",
    blue_fill, blue_edge
)
add_box(
    ax, upper_x[3], upper_y, box_w, box_h,
    "PCA reduction",
    "PC1–PC2 retained\n85.47% cumulative variance",
    blue_fill, blue_edge
)
add_box(
    ax, upper_x[4], upper_y, box_w, box_h,
    "K-means state discovery",
    "K = 3 ecological states\nvalidated by stability tests",
    green_fill, green_edge
)

for left_x, right_x in zip(upper_x[:-1], upper_x[1:]):
    add_arrow(
        ax,
        (left_x + box_w / 2 + 0.006, upper_y),
        (right_x - box_w / 2 - 0.006, upper_y)
    )

# Lower section and workflow row
ax.text(
    0.025, 0.485, "ENVIRONMENTAL STATE PREDICTION",
    fontsize=13.5, fontweight="bold", color=orange_edge,
    ha="left", va="center", transform=ax.transAxes
)
ax.plot(
    [0.025, 0.975], [0.455, 0.455], color="#D5D8DC", linewidth=1.0,
    transform=ax.transAxes, clip_on=False
)

lower_y = 0.25
lower_x = [0.12, 0.37, 0.62, 0.87]
lower_w, lower_h = 0.205, 0.235

add_box(
    ax, lower_x[0], lower_y, lower_w, lower_h,
    "Environmental predictors (X)",
    "14 current-month variables\nphysical and climate conditions",
    orange_fill, orange_edge
)
add_box(
    ax, lower_x[1], lower_y, lower_w, lower_h,
    "Ecological-state target (y)",
    "Final labels from detrended\nCLR–PCA–K-means analysis",
    green_fill, green_edge
)
add_box(
    ax, lower_x[2], lower_y, lower_w, lower_h,
    "Nested temporal validation",
    "Leave-one-year-out testing\nOptuna-TPE model optimization",
    purple_fill, purple_edge
)
add_box(
    ax, lower_x[3], lower_y, lower_w, lower_h,
    "Prediction and interpretation",
    "Six final models + baseline\nmetrics, calibration and SHAP",
    orange_fill, orange_edge
)

# X and y converge into temporal modelling.
add_arrow(
    ax,
    (lower_x[0] + lower_w / 2 + 0.006, lower_y),
    (lower_x[2] - lower_w / 2 - 0.006, lower_y),
    connectionstyle="arc3,rad=-0.12"
)
add_arrow(
    ax,
    (lower_x[1] + lower_w / 2 + 0.006, lower_y + 0.035),
    (lower_x[2] - lower_w / 2 - 0.006, lower_y + 0.035)
)
add_arrow(
    ax,
    (lower_x[2] + lower_w / 2 + 0.006, lower_y),
    (lower_x[3] - lower_w / 2 - 0.006, lower_y)
)

# Connect discovered states to their role as the target variable.
add_arrow(
    ax,
    (upper_x[4], upper_y - box_h / 2 - 0.006),
    (lower_x[1], lower_y + lower_h / 2 + 0.006),
    color=green_edge,
    connectionstyle="arc3,rad=0.20"
)

if ADD_PANEL_LABEL:
    ax.text(
        0.002, 0.995, "(a)", transform=ax.transAxes,
        ha="left", va="top", fontsize=17, fontweight="bold", color="black"
    )

fig.subplots_adjust(left=0.015, right=0.985, bottom=0.035, top=0.975)


# =============================================================================
# 4. SAVE TIFF AND PDF
# =============================================================================
tiff_file = OUTPUT_DIR / f"{OUTPUT_STEM}.tiff"
pdf_file = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"

fig.savefig(
    tiff_file, dpi=TIFF_DPI, format="tiff", bbox_inches="tight",
    pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"}
)
fig.savefig(
    pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04
)
plt.show()
plt.close(fig)

print("\n" + "=" * 90)
print("FIGURE 1(a) WORKFLOW COMPLETED")
print("=" * 90)
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[6]:


"""Monthly relative composition of six phytoplankton groups."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Paths and variables
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Data" / "01_Raw_Species_Environmental_Monthly_Data.xlsx"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["DIATO", "DINO", "HAPTO", "GREEN", "PROKAR", "PROCHLO"]
COLORS = ["#D55E00", "#E69F00", "#CC79A7", "#009E73", "#0072B2", "#56B4E9"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# Load and validate data
df = pd.read_excel(DATA_FILE)
required = ["Months", *GROUPS]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Months"] = pd.to_datetime(df["Months"], errors="coerce")
if df["Months"].isna().any() or df["Months"].duplicated().any():
    raise ValueError("Invalid or duplicated monthly dates were found.")
if df[GROUPS].isna().any().any() or (df[GROUPS] <= 0).any().any():
    raise ValueError("Phytoplankton values must be complete and positive.")

df = df.sort_values("Months").reset_index(drop=True)
relative_percent = df[GROUPS].div(df[GROUPS].sum(axis=1), axis=0) * 100


# Create the standalone panel: no panel letter and no main title
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.stackplot(
    df["Months"], *[relative_percent[group] for group in GROUPS],
    labels=GROUPS, colors=COLORS, alpha=0.92, linewidth=0.15,
    edgecolor="white"
)

ax.set_ylabel("Relative community composition (%)")
ax.set_xlabel("Year")
ax.set_ylim(0, 100)
ax.set_xlim(df["Months"].min(), df["Months"].max())
ax.set_yticks(range(0, 101, 20))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y", color="#BFC5CA", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(
    ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01),
    frameon=False, columnspacing=1.25, handlelength=1.4
)

fig.tight_layout(pad=0.6)


# Save publication files
tiff_file = OUTPUT_DIR / "Figure_1b_Monthly_Relative_Composition.tiff"
pdf_file = OUTPUT_DIR / "Figure_1b_Monthly_Relative_Composition.pdf"

fig.savefig(
    tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
    pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"}
)
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 1(b) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[7]:


"""Time-group heatmap of CLR-transformed phytoplankton composition."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm


# Paths and variables
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Data" / "01_Raw_Species_Environmental_Monthly_Data.xlsx"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["DIATO", "DINO", "HAPTO", "GREEN", "PROKAR", "PROCHLO"]

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 20,
    "font.weight": "normal",
    "axes.labelsize": 22,
    "axes.labelweight": "normal",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# Load, validate and transform the six-group composition
df = pd.read_excel(DATA_FILE)
required = ["Months", *GROUPS]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Months"] = pd.to_datetime(df["Months"], errors="coerce")
df = df.sort_values("Months").reset_index(drop=True)
if df["Months"].isna().any() or df["Months"].duplicated().any():
    raise ValueError("Invalid or duplicated monthly dates were found.")
if df[GROUPS].isna().any().any() or (df[GROUPS] <= 0).any().any():
    raise ValueError("CLR transformation requires complete positive values.")

composition = df[GROUPS].div(df[GROUPS].sum(axis=1), axis=0)
log_composition = np.log(composition)
clr = log_composition.sub(log_composition.mean(axis=1), axis=0)


# Create the standalone heatmap: no panel letter and no main title
fig, ax = plt.subplots(figsize=(14, 5.8))
date_values = mdates.date2num(df["Months"])
limit = np.abs(clr.to_numpy()).max()

image = ax.imshow(
    clr[GROUPS].to_numpy().T,
    aspect="auto",
    interpolation="nearest",
    origin="upper",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    extent=[date_values.min(), date_values.max(), len(GROUPS) - 0.5, -0.5]
)

ax.set_yticks(range(len(GROUPS)))
ax.set_yticklabels(GROUPS)
ax.set_ylabel("Phytoplankton group", labelpad=10)
ax.set_xlabel("Year", labelpad=7)
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="both", length=4, width=0.8)

colorbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.035)
colorbar.set_label("CLR value", rotation=90, labelpad=12,
                   fontsize=22, fontweight="normal", family="Calibri")
colorbar.ax.tick_params(labelsize=18)

fig.tight_layout(pad=0.8)


# Save publication files
tiff_file = OUTPUT_DIR / "Figure_1c_CLR_Time_Group_Heatmap.tiff"
pdf_file = OUTPUT_DIR / "Figure_1c_CLR_Time_Group_Heatmap.pdf"

fig.savefig(
    tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
    pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"}
)
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 1(c) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")
print(f"Maximum absolute CLR value: {limit:.4f}")


# In[5]:


"""Original and detrended CLR PC1 on the same final PCA basis."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "Results" / "01_State_Discovery"
MODEL_DIR = ROOT / "Models"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLR_FILE = RESULT_DIR / "02_CLR_Values.csv"
DETREND_FILE = RESULT_DIR / "03_Detrended_CLR_Values.csv"
PCA_SCORE_FILE = RESULT_DIR / "04_PCA_Scores.csv"
PCA_MODEL_FILE = MODEL_DIR / "CLR_PCA_Model.pkl"


# =============================================================================
# FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# LOAD SAVED FINAL OUTPUTS
# =============================================================================

clr = pd.read_csv(CLR_FILE)
dclr = pd.read_csv(DETREND_FILE)
saved_scores = pd.read_csv(PCA_SCORE_FILE)

bundle = joblib.load(PCA_MODEL_FILE)

pca = bundle["pca_model"]
PHYTO = bundle["phytoplankton_groups"]


# =============================================================================
# PREPARE DATA
# =============================================================================

original_cols = [f"{g}_CLR" for g in PHYTO]
detrended_cols = [f"{g}_DCLR" for g in PHYTO]

X_original = clr[original_cols].copy()
X_detrended = dclr[detrended_cols].copy()

# PCA was fitted using detrended column names
X_original.columns = detrended_cols

dates = pd.to_datetime(clr["Months"])


# =============================================================================
# PROJECT BOTH DATASETS ONTO THE SAME FINAL PCA BASIS
# =============================================================================

original_scores = pca.transform(X_original)
detrended_scores = pca.transform(X_detrended)

pc1_original = original_scores[:, 0]
pc1_detrended = detrended_scores[:, 0]


# =============================================================================
# VERIFY FINAL PCA REPRODUCTION
# =============================================================================

max_diff = abs(
    pc1_detrended - saved_scores["PC1"].to_numpy()
).max()

correlation = pd.Series(pc1_original).corr(
    pd.Series(pc1_detrended)
)

print(f"Maximum difference from saved final PC1 = {max_diff:.12f}")
print(f"Correlation between original and detrended PC1 = {correlation:.3f}")


# =============================================================================
# CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 4.8))

ax.plot(
    dates,
    pc1_original,
    color="#A94432",
    linewidth=2.0,
    label="Original CLR",
    zorder=2
)

ax.plot(
    dates,
    pc1_detrended,
    color="#356D9F",
    linewidth=2.0,
    label="Detrended CLR",
    zorder=3
)

ax.axhline(
    0,
    color="#7F858A",
    linestyle="--",
    linewidth=1.0,
    alpha=0.85,
    zorder=1
)


# =============================================================================
# AXES
# =============================================================================

ax.set_ylabel("PC1 score")
ax.set_xlabel("Year")

ax.set_xlim(dates.min(), dates.max())

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# =============================================================================
# GRID AND SPINES
# =============================================================================

ax.grid(
    axis="y",
    color="#BFC5CA",
    linewidth=0.6,
    alpha=0.55
)

ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)


# =============================================================================
# LEGEND
# =============================================================================

ax.legend(
    loc="upper right",
    frameon=False,
    handlelength=2.8
)


# =============================================================================
# LAYOUT
# =============================================================================

fig.tight_layout(pad=0.6)


# =============================================================================
# SAVE PUBLICATION FILES
# =============================================================================

tiff_file = OUTPUT_DIR / "Figure_1c_Original_Detrended_PC1.tiff"
pdf_file = OUTPUT_DIR / "Figure_1c_Original_Detrended_PC1.pdf"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)

print("\nFigure 1(c) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[9]:


"""Detrended CLR-PCA explained-variance plot for Main Figure 2(a)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Results" / "01_State_Discovery" / "06_PCA_Explained_Variance.csv"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 20, "font.weight": "normal",
    "axes.labelsize": 22, "axes.labelweight": "normal",
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42
})


# Read saved final PCA results.
variance = pd.read_csv(DATA_FILE)
required = ["Component", "Explained_Variance_Percent", "Cumulative_Variance_Percent"]
missing = [column for column in required if column not in variance.columns]
if missing:
    raise ValueError(f"Missing PCA columns: {missing}")

variance["Component"] = variance["Component"].astype(str)
explained = variance["Explained_Variance_Percent"]
cumulative = variance["Cumulative_Variance_Percent"]
components = variance["Component"]


# Standalone plot: no main title and no panel label.
fig, ax = plt.subplots(figsize=(10.5, 5.8))
bars = ax.bar(
    components, explained, width=0.72, color="#5B8DB8",
    edgecolor="#35658A", linewidth=0.8, label="Individual variance"
)
ax.set_ylabel("Explained variance (%)", labelpad=12)
ax.set_xlabel("Principal component", labelpad=7)
ax.set_ylim(0, 105)
ax.grid(axis="y", color="#BFC5CA", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
ax.spines[["top"]].set_visible(False)

for bar, value in zip(bars, explained):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2.1, f"{value:.1f}",
            ha="center", va="bottom", fontsize=16, family="Calibri")

ax2 = ax.twinx()
line = ax2.plot(
    components, cumulative, color="#9C2B1B", marker="o", markersize=8,
    linewidth=2.3, label="Cumulative variance"
)
ax2.axhline(85, color="#5D6D7E", linestyle="--", linewidth=1.3,
            label="85% threshold")
ax2.set_ylabel("Cumulative variance (%)", labelpad=12)
ax2.set_ylim(0, 105)
ax2.tick_params(axis="y", labelsize=18)
ax2.spines[["top"]].set_visible(False)

handles = [bars, line[0], ax2.lines[1]]
labels = ["Individual variance", "Cumulative variance", "85% threshold"]
ax.legend(handles, labels, loc="center right", frameon=False)

fig.tight_layout(pad=0.8)

tiff_file = OUTPUT_DIR / "Figure_2a_Detrended_PCA_Scree.tiff"
pdf_file = OUTPUT_DIR / "Figure_2a_Detrended_PCA_Scree.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 2(a) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")
print(f"PC1-PC2 cumulative variance: {cumulative.iloc[1]:.3f}%")


# In[26]:


"""Final detrended CLR-PCA biplot with K-means ecological states – tight, attractive layout.

The plot combines saved PC1-PC2 scores, final state labels and PCA loading
directions. It does not rerun PCA or K-means clustering.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. FILES, LABELS AND STYLE
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "Results" / "01_State_Discovery"
SCORE_FILE = RESULT_DIR / "09_Final_Ecological_States.csv"
LOADING_FILE = RESULT_DIR / "05_PCA_Loadings.csv"
VARIANCE_FILE = RESULT_DIR / "06_PCA_Explained_Variance.csv"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_LABELS = {1: "State 1", 2: "State 2", 3: "State 3"}
STATE_COLORS = {1: "#D55E00", 2: "#009E73", 3: "#3C78A8"}
LABEL_OFFSETS = {
    "DIATO": (-0.10, 0.08),
    "DINO": (-0.12, -0.10),
    "HAPTO": (0.10, -0.12),
    "GREEN": (0.13, -0.20),
    "PROKAR": (0.12, 0.06),
    "PROCHLO": (0.09, 0.10)
}

# Larger fonts for a more attractive appearance
plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 22,
    "font.weight": "normal",
    "axes.labelsize": 24,
    "axes.labelweight": "normal",
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 2. LOAD SAVED FINAL PCA AND STATE RESULTS
# =============================================================================
scores = pd.read_csv(SCORE_FILE)
loadings = pd.read_csv(LOADING_FILE, index_col="Phytoplankton_Group")
variance = pd.read_csv(VARIANCE_FILE)

required_score_columns = ["PC1", "PC2", "Ecological_State"]
missing_scores = [column for column in required_score_columns if column not in scores.columns]
if missing_scores:
    raise ValueError(f"Missing score columns: {missing_scores}")
if not {"PC1", "PC2"}.issubset(loadings.columns):
    raise ValueError("PCA loading file must contain PC1 and PC2.")
if set(scores["Ecological_State"].unique()) != {1, 2, 3}:
    raise ValueError("Expected final ecological states 1, 2 and 3.")

loadings = loadings[["PC1", "PC2"]]
pc1_var = variance.loc[variance["Component"] == "PC1", "Explained_Variance_Percent"].iloc[0]
pc2_var = variance.loc[variance["Component"] == "PC2", "Explained_Variance_Percent"].iloc[0]


# =============================================================================
# 3. SCALE LOADING ARROWS TO THE SCORE SPACE FOR DISPLAY ONLY
# =============================================================================
max_score_x = max(abs(scores["PC1"].min()), abs(scores["PC1"].max()))
max_score_y = max(abs(scores["PC2"].min()), abs(scores["PC2"].max()))
max_loading_x = abs(loadings["PC1"]).max()
max_loading_y = abs(loadings["PC2"]).max()

arrow_scale = 0.64 * min(max_score_x / max_loading_x, max_score_y / max_loading_y)
scaled_loadings = loadings * arrow_scale


# =============================================================================
# 4. DRAW TIGHT BIPLOT WITH FULL BOX AND INSIDE LEGEND
# =============================================================================
fig, ax = plt.subplots(figsize=(10.0, 7.5))   # slightly larger but tighter limits

# Scatter points
for state in [1, 2, 3]:
    subset = scores[scores["Ecological_State"] == state]
    ax.scatter(
        subset["PC1"], subset["PC2"], s=80,
        color=STATE_COLORS[state], alpha=0.7,
        edgecolor="white", linewidth=0.50,
        label=STATE_LABELS[state], zorder=2
    )
    centroid = subset[["PC1", "PC2"]].mean()
    ax.scatter(
        centroid["PC1"], centroid["PC2"], marker="X", s=240,
        color=STATE_COLORS[state], edgecolor="black", linewidth=0.85,
        zorder=4
    )

# PCA loading arrows and labels with white background
for group, row in scaled_loadings.iterrows():
    arrow_x, arrow_y = row["PC1"], row["PC2"]
    ax.annotate(
        "", xy=(arrow_x, arrow_y), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": "#1C2833",
                   "lw": 1.75, "mutation_scale": 15}, zorder=5
    )
    offset_x, offset_y = LABEL_OFFSETS.get(group, (0.08, 0.08))
    ax.text(
        arrow_x + offset_x, arrow_y + offset_y, group,
        fontsize=17, family="Calibri", fontweight="normal",
        ha="center", va="center", color="#17202A", zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none")
    )

# Axes lines
ax.axhline(0, color="#7F8C8D", linewidth=0.85, zorder=0)
ax.axvline(0, color="#7F8C8D", linewidth=0.85, zorder=0)

# Smart axis labels (one decimal)
ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", labelpad=6)
ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", labelpad=10)

ax.grid(color="#D5D8DC", linewidth=0.55, alpha=0.55)
ax.set_axisbelow(True)

# Full box (all spines)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.85)
    spine.set_color("black")

# ----- TIGHTEN AXIS LIMITS FURTHER -----
# Use 1.08 multiplier on all sides to trim white space while keeping labels
x_margin = 1.08
y_margin = 1.08
ax.set_xlim(-max_score_x * x_margin, max_score_x * x_margin)
ax.set_ylim(-max_score_y * y_margin, max_score_y * y_margin)

# Legend inside – choose a location that minimises overlap; try upper left
ax.legend(
    loc="upper left", frameon=True, facecolor="white", edgecolor="black",
    framealpha=0.85, borderaxespad=0.3, handletextpad=0.4
)

fig.tight_layout(pad=0.5)


# =============================================================================
# 5. SAVE TIFF AND PDF
# =============================================================================
tiff_file = OUTPUT_DIR / "Figure_2b_Detrended_PCA_KMeans_Biplot.tiff"
pdf_file = OUTPUT_DIR / "Figure_2b_Detrended_PCA_KMeans_Biplot.pdf"
fig.savefig(
    tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
    pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"}
)
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\n" + "=" * 90)
print("FIGURE 2(b) PCA-K-MEANS BIPLOT COMPLETED (tight & attractive)")
print("=" * 90)
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")
print("\nSaved loading directions:")
print(loadings.round(4).to_string())


# In[12]:


"""PCA score plot of the final ecological states with convex hulls and centroids."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "Results" / "01_State_Discovery"

PCA_FILE = RESULT_DIR / "04_PCA_Scores.csv"
STATE_FILE = RESULT_DIR / "09_Final_Ecological_States.csv"
VAR_FILE = RESULT_DIR / "06_PCA_Explained_Variance.csv"

OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

# Larger fonts because this panel will later be combined with other panels.
plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 24,
    "font.weight": "normal",

    "axes.labelsize": 28,
    "axes.labelweight": "normal",

    "xtick.labelsize": 23,
    "ytick.labelsize": 23,

    "legend.fontsize": 21,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def find_first_existing_column(dataframe, candidates, table_name):
    """Return the first matching column name from a candidate list."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find any of {candidates} in {table_name}. "
        f"Available columns are: {list(dataframe.columns)}"
    )


def add_convex_hull(
    ax,
    x,
    y,
    facecolor,
    edgecolor,
    alpha=0.16,
    linewidth=1.8
):
    """Add a semi-transparent convex hull polygon."""

    points = np.column_stack([x, y])

    # At least three unique points are required.
    unique_points = np.unique(points, axis=0)

    if unique_points.shape[0] < 3:
        return

    hull = ConvexHull(unique_points)

    hull_points = unique_points[hull.vertices]

    polygon = Polygon(
        hull_points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round"
    )

    ax.add_patch(polygon)


# =============================================================================
# 4. LOAD SAVED DATA
# =============================================================================

pca_scores = pd.read_csv(PCA_FILE)
states = pd.read_csv(STATE_FILE)
variance = pd.read_csv(VAR_FILE)


# =============================================================================
# 5. IDENTIFY IMPORTANT COLUMNS
# =============================================================================

pc1_col = find_first_existing_column(
    pca_scores,
    ["PC1", "pc1", "PC_1"],
    "04_PCA_Scores.csv"
)

pc2_col = find_first_existing_column(
    pca_scores,
    ["PC2", "pc2", "PC_2"],
    "04_PCA_Scores.csv"
)

state_col = find_first_existing_column(
    states,
    ["Ecological_State", "State", "Final_State"],
    "09_Final_Ecological_States.csv"
)


# Possible shared identifiers
possible_keys = [
    "Date",
    "date",
    "Month",
    "month",
    "Time",
    "Datetime",
    "Observation_ID",
    "ID",
    "Sample_ID"
]

shared_keys = [
    key
    for key in possible_keys
    if key in pca_scores.columns and key in states.columns
]


# =============================================================================
# 6. MERGE PCA SCORES WITH STATE LABELS
# =============================================================================

if shared_keys:

    merge_key = shared_keys[0]

    plot_df = pd.merge(
        pca_scores,
        states[[merge_key, state_col]],
        on=merge_key,
        how="inner"
    )

else:

    # Fall back to row-wise alignment when row counts match.
    if len(pca_scores) != len(states):

        raise ValueError(
            "No shared merge key was found between PCA scores and "
            "ecological states, and row counts do not match."
        )

    plot_df = pca_scores.copy()

    plot_df[state_col] = states[state_col].values


# =============================================================================
# 7. EXPLAINED VARIANCE FOR AXIS LABELS
# =============================================================================

variance_pc_col = find_first_existing_column(
    variance,
    ["PC", "Component", "Principal_Component"],
    "06_PCA_Explained_Variance.csv"
)

variance_value_col = find_first_existing_column(
    variance,
    [
        "Explained_Variance_Percent",
        "ExplainedVariancePercent",
        "Percent",
        "Variance_Explained_Percent"
    ],
    "06_PCA_Explained_Variance.csv"
)


pc1_var = variance.loc[
    variance[variance_pc_col].astype(str).str.upper() == "PC1",
    variance_value_col
]

pc2_var = variance.loc[
    variance[variance_pc_col].astype(str).str.upper() == "PC2",
    variance_value_col
]


if len(pc1_var) == 0 or len(pc2_var) == 0:

    raise ValueError(
        "Could not find PC1/PC2 explained variance values in "
        "06_PCA_Explained_Variance.csv."
    )


pc1_var = float(pc1_var.iloc[0])
pc2_var = float(pc2_var.iloc[0])


# =============================================================================
# 8. STANDARDISE STATE LABELS
# =============================================================================

def normalise_state_label(value):
    """Convert different state formats to State 1, State 2, State 3."""

    text = str(value).strip()

    if text in {"1", "2", "3"}:
        return f"State {text}"

    if text.lower() in {"state 1", "state 2", "state 3"}:
        return text.title()

    return text


plot_df["State_Display"] = (
    plot_df[state_col]
    .apply(normalise_state_label)
)


ordered_states = [
    "State 1",
    "State 2",
    "State 3"
]


# =============================================================================
# 9. STATE COLOURS
# =============================================================================

STATE_COLOURS = {
    "State 1": "#D55E5E",   # muted red
    "State 2": "#2A9D8F",   # teal
    "State 3": "#7B6BB3",   # soft purple
}

STATE_EDGE_COLOURS = {
    "State 1": "#B44343",
    "State 2": "#1F7F73",
    "State 3": "#63559A",
}


# =============================================================================
# 10. CREATE PCA SCORE PLOT
# =============================================================================

fig, ax = plt.subplots(
    figsize=(9.6, 8.6)
)


for state in ordered_states:

    subset = plot_df.loc[
        plot_df["State_Display"] == state
    ].copy()

    if subset.empty:
        continue

    x = subset[pc1_col].to_numpy()
    y = subset[pc2_col].to_numpy()

    colour = STATE_COLOURS[state]
    edge_colour = STATE_EDGE_COLOURS[state]


    # -------------------------------------------------------------------------
    # 10.1 CONVEX HULL
    # -------------------------------------------------------------------------

    add_convex_hull(
        ax=ax,
        x=x,
        y=y,
        facecolor=colour,
        edgecolor=edge_colour,
        alpha=0.14,
        linewidth=1.7
    )


    # -------------------------------------------------------------------------
    # 10.2 INDIVIDUAL MONTHLY OBSERVATIONS
    # -------------------------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=58,
        color=colour,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.90,
        zorder=3
    )


    # -------------------------------------------------------------------------
    # 10.3 STATE CENTROID
    # -------------------------------------------------------------------------

    centroid_x = x.mean()
    centroid_y = y.mean()

    ax.scatter(
        centroid_x,
        centroid_y,
        s=245,
        marker="X",
        color=edge_colour,
        edgecolor="white",
        linewidth=1.2,
        zorder=4
    )


# =============================================================================
# 11. ZERO REFERENCE LINES
# =============================================================================

ax.axhline(
    y=0,
    color="#333333",
    linewidth=1.3,
    linestyle=(0, (4, 4)),
    zorder=1
)

ax.axvline(
    x=0,
    color="#333333",
    linewidth=1.3,
    linestyle=(0, (4, 4)),
    zorder=1
)


# =============================================================================
# 12. AXIS LABELS
# =============================================================================

ax.set_xlabel(
    f"PC1 ({pc1_var:.2f}%)",
    labelpad=10
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.2f}%)",
    labelpad=10
)


# =============================================================================
# 13. GRID AND FULL BOX
# =============================================================================

ax.grid(
    color="#D3D7DB",
    linewidth=0.65,
    alpha=0.55
)

ax.set_axisbelow(True)


for spine in ax.spines.values():

    spine.set_visible(True)

    spine.set_linewidth(1.1)

    spine.set_color("#4A4A4A")


# =============================================================================
# 14. LEGEND — INSIDE UPPER-LEFT EMPTY REGION
# =============================================================================

legend_handles = []

for state in ordered_states:

    legend_handles.append(

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=STATE_COLOURS[state],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=11,
            label=state
        )
    )


ax.legend(
    handles=legend_handles,

    # Move legend to the upper-left empty region
    loc="upper left",

    # Small inward offset so it does not touch the plot border
    bbox_to_anchor=(0.00, 1.00),

    frameon=False,
    fancybox=False,

    edgecolor="#B5B5B5",
    facecolor="white",
    framealpha=0.94,

    borderpad=0.60,
    handletextpad=0.60,

    labelspacing=0.45
)


# =============================================================================
# 15. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.9
)


# =============================================================================
# 16. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_2b_PCA_State_Score_Plot.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_2b_PCA_State_Score_Plot.pdf"
)


fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)


plt.show()

plt.close(fig)


# =============================================================================
# 17. OUTPUT SUMMARY
# =============================================================================

print("\nFigure 2(b) files saved:")

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)


print("\nState counts shown in the PCA score plot:")

print(
    plot_df["State_Display"]
    .value_counts()
    .reindex(ordered_states)
)


print(
    "\nPC1 + PC2 explained variance: "
    f"{pc1_var + pc2_var:.2f}%"
)


# In[28]:


"""PCA loading vector plot for the retained detrended PCs (PC1 and PC2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "Results"
    / "01_State_Discovery"
    / "05_PCA_Loadings.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_02"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PCS = ["PC1", "PC2"]


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 23,
    "font.weight": "normal",

    "axes.labelsize": 27,
    "axes.labelweight": "normal",

    "xtick.labelsize": 21,
    "ytick.labelsize": 21,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. LOAD SAVED PCA LOADINGS
# =============================================================================

loadings = pd.read_csv(
    DATA_FILE,
    index_col="Phytoplankton_Group"
)

missing = [
    component
    for component in PCS
    if component not in loadings.columns
]

if missing:
    raise ValueError(
        f"Missing retained PCA loading columns: {missing}"
    )

loadings = loadings[PCS].copy()


# =============================================================================
# 4. COLOUR SETTINGS
# =============================================================================

ARROW_COLOR = "#355C7D"
POINT_COLOR = "#C65D3A"
TEXT_COLOR = "#303030"

ZERO_LINE_COLOR = "#555555"
GRID_COLOR = "#D5D9DC"


# =============================================================================
# 5. BALANCED LABEL POSITIONS
# =============================================================================

# Offsets are specified in DISPLAY POINTS rather than PCA-coordinate units.
# This gives more consistent visual spacing around the arrow endpoints.

LABEL_OFFSETS = {
    "DIATO":   (-10,  8),
    "DINO":    (-10, -10),
    "HAPTO":   (  8, -10),
    "GREEN":   (  8,  -3),
    "PROKAR":  (-6,  -4),
    "PROCHLO": (  8,   7),
}


# =============================================================================
# 6. AXIS LIMITS
# =============================================================================

max_loading = np.abs(
    loadings[PCS].to_numpy()
).max()

axis_limit = max(
    0.85,
    max_loading * 1.18
)


# =============================================================================
# 7. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.8, 7.2)
)


# =============================================================================
# 8. ZERO REFERENCE LINES
# =============================================================================

ax.axhline(
    y=0,
    color=ZERO_LINE_COLOR,
    linewidth=1.3,
    linestyle=(0, (4, 4)),
    zorder=1
)

ax.axvline(
    x=0,
    color=ZERO_LINE_COLOR,
    linewidth=1.3,
    linestyle=(0, (4, 4)),
    zorder=1
)


# =============================================================================
# 9. DRAW PCA LOADING VECTORS
# =============================================================================

for group, row in loadings.iterrows():

    pc1 = float(row["PC1"])
    pc2 = float(row["PC2"])

    # -------------------------------------------------------------------------
    # 9.1 Vector
    # -------------------------------------------------------------------------

    ax.annotate(
        "",
        xy=(pc1, pc2),
        xytext=(0, 0),

        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            linewidth=2.4,
            mutation_scale=18,
            shrinkA=0,
            shrinkB=0
        ),

        zorder=3
    )


    # -------------------------------------------------------------------------
    # 9.2 Endpoint marker
    # -------------------------------------------------------------------------

    ax.scatter(
        pc1,
        pc2,
        s=105,
        color=POINT_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=4
    )


    # -------------------------------------------------------------------------
    # 9.3 Balanced direct label
    # -------------------------------------------------------------------------

    dx, dy = LABEL_OFFSETS.get(
        group,
        (8, 8)
    )

    # Horizontal alignment follows which side of the point the label occupies.
    if dx < 0:
        horizontal_alignment = "right"
    elif dx > 0:
        horizontal_alignment = "left"
    else:
        horizontal_alignment = "center"

    # Vertical alignment follows label direction.
    if dy < 0:
        vertical_alignment = "top"
    elif dy > 0:
        vertical_alignment = "bottom"
    else:
        vertical_alignment = "center"

    ax.annotate(
        group,

        xy=(pc1, pc2),

        xytext=(dx, dy),
        textcoords="offset points",

        ha=horizontal_alignment,
        va=vertical_alignment,

        fontsize=19,
        color=TEXT_COLOR,
        family="Calibri",

        annotation_clip=True,

        zorder=5
    )


# =============================================================================
# 10. AXES
# =============================================================================

ax.set_xlim(
    -axis_limit,
    axis_limit
)

ax.set_ylim(
    -axis_limit,
    axis_limit
)

ax.set_xlabel(
    "PC1 loading",
    labelpad=10
)

ax.set_ylabel(
    "PC2 loading",
    labelpad=10
)


# =============================================================================
# 11. EQUAL ASPECT
# =============================================================================

ax.set_aspect(
    "equal",
    adjustable="box"
)


# =============================================================================
# 12. GRID
# =============================================================================

ax.grid(
    color=GRID_COLOR,
    linewidth=0.7,
    alpha=0.55
)

ax.set_axisbelow(True)


# =============================================================================
# 13. FULL BOX
# =============================================================================

for spine in ax.spines.values():

    spine.set_visible(True)
    spine.set_linewidth(1.05)
    spine.set_color("#555555")


# =============================================================================
# 14. TICK LOCATIONS
# =============================================================================

tick_limit = np.floor(
    axis_limit * 10
) / 10

ticks = np.arange(
    -tick_limit,
    tick_limit + 0.001,
    0.4
)

ax.set_xticks(ticks)
ax.set_yticks(ticks)


# =============================================================================
# 15. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.9
)


# =============================================================================
# 16. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_2c_PCA_Loading_Vectors.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_2c_PCA_Loading_Vectors.pdf"
)


fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)


plt.show()
plt.close(fig)


# =============================================================================
# 17. OUTPUT SUMMARY
# =============================================================================

print("\nFigure 2(c) — PCA loading-vector plot saved:")

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)


print("\nRetained PCA loadings:")

print(
    loadings.round(3)
)


# In[11]:


"""PCA biplot: ecological states + scaled phytoplankton loading vectors."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "Results" / "01_State_Discovery"

PCA_FILE = RESULT_DIR / "04_PCA_Scores.csv"
STATE_FILE = RESULT_DIR / "09_Final_Ecological_States.csv"
VAR_FILE = RESULT_DIR / "06_PCA_Explained_Variance.csv"
LOADINGS_FILE = RESULT_DIR / "05_PCA_Loadings.csv"

OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 28,

    "axes.labelsize": 32,

    "xtick.labelsize": 24,
    "ytick.labelsize": 24,

    "legend.fontsize": 24,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def find_first_existing_column(dataframe, candidates, table_name):

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find any of {candidates} in {table_name}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def add_convex_hull(
    ax,
    x,
    y,
    facecolor,
    edgecolor,
    alpha=0.14,
    linewidth=1.7
):

    points = np.column_stack([x, y])
    unique_points = np.unique(points, axis=0)

    if unique_points.shape[0] < 3:
        return

    hull = ConvexHull(unique_points)
    hull_points = unique_points[hull.vertices]

    polygon = Polygon(
        hull_points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round"
    )

    ax.add_patch(polygon)


def normalise_state_label(value):

    text = str(value).strip()

    if text in {"1", "2", "3"}:
        return f"State {text}"

    if text.lower() in {"state 1", "state 2", "state 3"}:
        return text.title()

    return text


# =============================================================================
# 4. LOAD DATA
# =============================================================================

pca_scores = pd.read_csv(PCA_FILE)
states = pd.read_csv(STATE_FILE)
variance = pd.read_csv(VAR_FILE)

loadings = pd.read_csv(
    LOADINGS_FILE,
    index_col="Phytoplankton_Group"
)


# =============================================================================
# 5. IDENTIFY COLUMNS
# =============================================================================

pc1_col = find_first_existing_column(
    pca_scores,
    ["PC1", "pc1", "PC_1"],
    "04_PCA_Scores.csv"
)

pc2_col = find_first_existing_column(
    pca_scores,
    ["PC2", "pc2", "PC_2"],
    "04_PCA_Scores.csv"
)

state_col = find_first_existing_column(
    states,
    ["Ecological_State", "State", "Final_State"],
    "09_Final_Ecological_States.csv"
)


# =============================================================================
# 6. ALIGN PCA SCORES AND STATES
# =============================================================================

possible_keys = [
    "Date",
    "date",
    "Month",
    "month",
    "Time",
    "Datetime",
    "Observation_ID",
    "ID",
    "Sample_ID"
]

shared_keys = [
    key
    for key in possible_keys
    if key in pca_scores.columns
    and key in states.columns
]

if shared_keys:

    merge_key = shared_keys[0]

    plot_df = pd.merge(
        pca_scores,
        states[[merge_key, state_col]],
        on=merge_key,
        how="inner"
    )

else:

    if len(pca_scores) != len(states):
        raise ValueError(
            "PCA scores and ecological states cannot be aligned."
        )

    plot_df = pca_scores.copy()
    plot_df[state_col] = states[state_col].values


plot_df["State_Display"] = (
    plot_df[state_col]
    .apply(normalise_state_label)
)

ordered_states = [
    "State 1",
    "State 2",
    "State 3"
]


# =============================================================================
# 7. EXPLAINED VARIANCE
# =============================================================================

variance_pc_col = find_first_existing_column(
    variance,
    ["PC", "Component", "Principal_Component"],
    "06_PCA_Explained_Variance.csv"
)

variance_value_col = find_first_existing_column(
    variance,
    [
        "Explained_Variance_Percent",
        "ExplainedVariancePercent",
        "Percent",
        "Variance_Explained_Percent"
    ],
    "06_PCA_Explained_Variance.csv"
)

pc1_var = float(
    variance.loc[
        variance[variance_pc_col]
        .astype(str)
        .str.upper() == "PC1",
        variance_value_col
    ].iloc[0]
)

pc2_var = float(
    variance.loc[
        variance[variance_pc_col]
        .astype(str)
        .str.upper() == "PC2",
        variance_value_col
    ].iloc[0]
)


# =============================================================================
# 8. LOADINGS
# =============================================================================

loadings = loadings[["PC1", "PC2"]].copy()


# =============================================================================
# 9. COLOURS
# =============================================================================

STATE_COLOURS = {
    "State 1": "#D55E5E",
    "State 2": "#2A9D8F",
    "State 3": "#7B6BB3",
}

STATE_EDGE_COLOURS = {
    "State 1": "#B44343",
    "State 2": "#1F7F73",
    "State 3": "#63559A",
}

ARROW_COLOR = "#2F4F6F"
TEXT_COLOR = "#252525"


# =============================================================================
# 10. LOADING LABEL OFFSETS
# =============================================================================

LABEL_OFFSETS = {
    "DIATO":   (-10,   9),
    "DINO":    (-10,  -9),
    "HAPTO":   (  8, -10),
    "GREEN":   (  8,  -6),
    "PROKAR":  ( 10,   8),
    "PROCHLO": (  8,   8),
}


# =============================================================================
# 11. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(9.8, 8.8)
)


# =============================================================================
# 12. ECOLOGICAL STATES
# =============================================================================

for state in ordered_states:

    subset = plot_df.loc[
        plot_df["State_Display"] == state
    ]

    if subset.empty:
        continue

    x = subset[pc1_col].to_numpy()
    y = subset[pc2_col].to_numpy()

    colour = STATE_COLOURS[state]
    edge_colour = STATE_EDGE_COLOURS[state]

    # Convex hull
    add_convex_hull(
        ax,
        x,
        y,
        colour,
        edge_colour
    )

    # Monthly observations
    ax.scatter(
        x,
        y,
        s=58,
        color=colour,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        zorder=3
    )

    # Centroid
    ax.scatter(
        x.mean(),
        y.mean(),
        s=245,
        marker="X",
        color=edge_colour,
        edgecolor="white",
        linewidth=1.2,
        zorder=5
    )


# =============================================================================
# 13. ZERO LINES
# =============================================================================

ax.axhline(
    0,
    color="#444444",
    linewidth=1.25,
    linestyle=(0, (4, 4)),
    zorder=1
)

ax.axvline(
    0,
    color="#444444",
    linewidth=1.25,
    linestyle=(0, (4, 4)),
    zorder=1
)


# =============================================================================
# 14. SCALE LOADINGS FOR DISPLAY
# =============================================================================

# One common scaling factor is applied to ALL loading vectors.
# Directions and relative vector lengths are therefore preserved.

score_x_range = (
    plot_df[pc1_col].max()
    - plot_df[pc1_col].min()
)

score_y_range = (
    plot_df[pc2_col].max()
    - plot_df[pc2_col].min()
)

loading_x_max = np.abs(loadings["PC1"]).max()
loading_y_max = np.abs(loadings["PC2"]).max()

scale_x = 0.34 * score_x_range / loading_x_max
scale_y = 0.34 * score_y_range / loading_y_max

LOAD_SCALE = min(
    scale_x,
    scale_y
)

print(
    f"Loading display scale = {LOAD_SCALE:.3f}"
)


# =============================================================================
# 15. DRAW LOADING VECTORS ON SAME PCA PLOT
# =============================================================================

for group, row in loadings.iterrows():

    x_loading = float(row["PC1"]) * LOAD_SCALE
    y_loading = float(row["PC2"]) * LOAD_SCALE

    # Vector
    ax.annotate(
        "",
        xy=(x_loading, y_loading),
        xytext=(0, 0),

        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            linewidth=2.2,
            mutation_scale=17
        ),

        zorder=6
    )

    # Endpoint
    ax.scatter(
        x_loading,
        y_loading,
        s=75,
        color="#C65D3A",
        edgecolor="white",
        linewidth=0.8,
        zorder=7
    )

    # Label position
    dx, dy = LABEL_OFFSETS.get(
        group,
        (8, 8)
    )

    if dx < 0:
        ha = "right"
    elif dx > 0:
        ha = "left"
    else:
        ha = "center"

    if dy < 0:
        va = "top"
    elif dy > 0:
        va = "bottom"
    else:
        va = "center"

    ax.annotate(
        group,

        xy=(x_loading, y_loading),

        xytext=(dx, dy),
        textcoords="offset points",

        ha=ha,
        va=va,

        fontsize=22,
        family="Calibri",
        color=TEXT_COLOR,

        zorder=8
    )


# =============================================================================
# 16. AXES
# =============================================================================

ax.set_xlabel(
    f"PC1 ({pc1_var:.2f}%)",
    labelpad=10
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.2f}%)",
    labelpad=10
)


# =============================================================================
# 17. GRID
# =============================================================================

ax.grid(
    color="#D3D7DB",
    linewidth=0.65,
    alpha=0.50
)

ax.set_axisbelow(True)


# =============================================================================
# 18. FULL BOX
# =============================================================================

for spine in ax.spines.values():

    spine.set_visible(True)
    spine.set_linewidth(1.1)
    spine.set_color("#4A4A4A")


# =============================================================================
# 19. STATE LEGEND
# =============================================================================

legend_handles = []

for state in ordered_states:

    legend_handles.append(

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",

            markerfacecolor=
                STATE_COLOURS[state],

            markeredgecolor="white",

            markersize=11,

            label=state
        )
    )


ax.legend(
    handles=legend_handles,

    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),

    frameon=False,

    handletextpad=0.55,
    labelspacing=0.42
)


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.9
)


# =============================================================================
# 21. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_2b_PCA_State_Loading_Biplot.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_2b_PCA_State_Loading_Biplot.pdf"
)


fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


print("\nPCA state-loading biplot saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[12]:


"""Mean relative phytoplankton composition of the three ecological states."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Results" / "01_State_Discovery" / "10_State_Composition.csv"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["DIATO", "DINO", "HAPTO", "GREEN", "PROKAR", "PROCHLO"]
STATE_COLORS = ["#D55E00", "#009E73", "#3C78A8"]

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 20, "font.weight": "normal",
    "axes.labelsize": 22, "axes.labelweight": "normal",
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42
})


# Read final saved state composition table (values are percentages).
composition = pd.read_csv(DATA_FILE, index_col="Ecological_State")
missing = [group for group in GROUPS if group not in composition.columns]
if missing:
    raise ValueError(f"Missing phytoplankton groups: {missing}")
composition = composition.loc[[1, 2, 3], GROUPS]


# Standalone grouped-bar panel: no main title and no panel label.
fig, ax = plt.subplots(figsize=(12.5, 6.3))
x = np.arange(len(GROUPS))
width = 0.25

for position, state in enumerate([1, 2, 3]):
    ax.bar(x + (position - 1) * width, composition.loc[state], width=width,
           color=STATE_COLORS[position], edgecolor="white", linewidth=0.55,
           label=f"State {state}")

ax.set_xticks(x, GROUPS)
ax.set_xlabel("Phytoplankton group", labelpad=8)
ax.set_ylabel("Mean relative composition (%)", labelpad=12)
ax.set_ylim(0, max(75, composition.to_numpy().max() * 1.14))
ax.grid(axis="y", color="#BFC5CA", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=3, loc="upper left", frameon=False, handlelength=1.3)
fig.tight_layout(pad=0.8)

tiff_file = OUTPUT_DIR / "Figure_2d_State_Community_Composition.tiff"
pdf_file = OUTPUT_DIR / "Figure_2d_State_Community_Composition.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 2(d) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[13]:


"""Monthly occurrence heatmap of the final ecological states."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Results" / "01_State_Discovery" / "11_Monthly_State_Occurrence.csv"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STATE_COLUMNS = ["State_1_Percent", "State_2_Percent", "State_3_Percent"]

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 20, "font.weight": "normal",
    "axes.labelsize": 22, "axes.labelweight": "normal",
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "pdf.fonttype": 42, "ps.fonttype": 42
})


# Read saved final seasonal state occurrence (%).
occurrence = pd.read_csv(DATA_FILE)
missing = [column for column in ["Month_Number", *STATE_COLUMNS] if column not in occurrence.columns]
if missing:
    raise ValueError(f"Missing seasonal-occurrence columns: {missing}")
occurrence = occurrence.sort_values("Month_Number")
if occurrence["Month_Number"].tolist() != list(range(1, 13)):
    raise ValueError("Expected one row for each calendar month (1 to 12).")

matrix = occurrence[STATE_COLUMNS].to_numpy().T


# Standalone heatmap: no main title and no panel label.
fig, ax = plt.subplots(figsize=(12.5, 4.8))
image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

ax.set_xticks(range(12), MONTHS)
ax.set_yticks(range(3), ["State 1", "State 2", "State 3"])
ax.set_xlabel("Month", labelpad=8)
ax.set_ylabel("Ecological state", labelpad=12)

for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        value = matrix[row, column]
        colour = "white" if value >= 50 else "black"
        ax.text(column, row, f"{value:.0f}", ha="center", va="center",
                fontsize=17, color=colour, family="Calibri")

for edge in np.arange(-0.5, 12, 1):
    ax.axvline(edge, color="white", linewidth=1.0)
for edge in np.arange(-0.5, 3, 1):
    ax.axhline(edge, color="white", linewidth=1.0)

colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
colorbar.set_label("Monthly occurrence (%)", rotation=90, labelpad=12,
                   fontsize=22, family="Calibri", fontweight="normal")
colorbar.ax.tick_params(labelsize=18)
fig.tight_layout(pad=0.8)

tiff_file = OUTPUT_DIR / "Figure_2e_Seasonal_State_Occurrence.tiff"
pdf_file = OUTPUT_DIR / "Figure_2e_Seasonal_State_Occurrence.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 2(e) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[14]:


"""Categorical monthly timeline of the final ecological states."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Results" / "01_State_Discovery" / "09_Final_Ecological_States.csv"
OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLORS = {1: "#D55E00", 2: "#009E73", 3: "#3C78A8"}
STATE_LABELS = {1: "State 1", 2: "State 2", 3: "State 3"}

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 20, "font.weight": "normal",
    "axes.labelsize": 22, "axes.labelweight": "normal",
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42
})


# Read saved final monthly state assignments.
states = pd.read_csv(DATA_FILE, usecols=["Months", "Ecological_State"])
states["Months"] = pd.to_datetime(states["Months"], errors="coerce")
states = states.sort_values("Months").reset_index(drop=True)
if states["Months"].isna().any() or states["Months"].duplicated().any():
    raise ValueError("Invalid or duplicate state-assignment dates were found.")
if set(states["Ecological_State"].unique()) != {1, 2, 3}:
    raise ValueError("Expected final ecological states 1, 2 and 3.")


# Build one rectangle for each uninterrupted state run.
run_starts = states.index[
    states["Ecological_State"].ne(states["Ecological_State"].shift())
].tolist()
run_starts.append(len(states))
runs = []
for start_index, end_index in zip(run_starts[:-1], run_starts[1:]):
    state = int(states.loc[start_index, "Ecological_State"])
    start_date = states.loc[start_index, "Months"]
    end_date = (
        states.loc[end_index, "Months"]
        if end_index < len(states)
        else states.loc[len(states) - 1, "Months"] + pd.offsets.MonthBegin(1)
    )
    start_number = mdates.date2num(start_date)
    width = mdates.date2num(end_date) - start_number
    runs.append((start_number, width, state))


# Standalone categorical timeline: no main title and no panel label.
fig, ax = plt.subplots(figsize=(14, 3.8))
for start_number, width, state in runs:
    ax.broken_barh([(start_number, width)], (0, 1), facecolors=STATE_COLORS[state],
                   edgecolors="white", linewidth=0.30)

ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("Ecological state sequence", labelpad=12)
ax.set_xlabel("Year", labelpad=7)
ax.set_xlim(mdates.date2num(states["Months"].min()),
            mdates.date2num(states["Months"].max() + pd.offsets.MonthBegin(1)))
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.spines[["top", "right", "left"]].set_visible(False)

legend_handles = [Patch(facecolor=STATE_COLORS[state], label=STATE_LABELS[state])
                  for state in [1, 2, 3]]
ax.legend(handles=legend_handles, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.03), frameon=False, handlelength=1.35)
fig.tight_layout(pad=0.8)

tiff_file = OUTPUT_DIR / "Figure_2f_Temporal_State_Sequence.tiff"
pdf_file = OUTPUT_DIR / "Figure_2f_Temporal_State_Sequence.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nFigure 2(f) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[18]:


"""Supplementary Figure S1(a): internal clustering-quality comparison.

Uses the saved final PC1-PC2 scores. It does not change final K-means labels
or train any predictive model.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*"
)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

SCORE_FILE = (
    ROOT
    / "Results"
    / "01_State_Discovery"
    / "04_PCA_Scores.csv"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "01_State_Discovery"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Supplementary_Figures"
    / "Figure_S1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. SETTINGS
# =============================================================================

METHODS = [
    "K-means",
    "GMM",
    "Ward"
]

METHOD_COLORS = {
    "K-means": "#0072B2",
    "GMM": "#D55E00",
    "Ward": "#009E73"
}

K_VALUES = range(2, 7)
RANDOM_STATE = 42


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 26,
    "font.weight": "normal",

    "axes.labelsize": 27,
    "axes.labelweight": "normal",

    "xtick.labelsize": 25,
    "ytick.labelsize": 25,

    "legend.fontsize": 27,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD SAVED FINAL PC1-PC2 SCORES
# =============================================================================

scores = pd.read_csv(SCORE_FILE)

if not {"PC1", "PC2"}.issubset(scores.columns):

    raise ValueError(
        "Saved PCA score table must contain PC1 and PC2."
    )


X = scores[["PC1", "PC2"]].to_numpy(float)

if not np.isfinite(X).all():

    raise ValueError(
        "Non-finite PC scores were found."
    )


# =============================================================================
# 5. INTERNAL CLUSTER VALIDATION
# =============================================================================

records = []

for k in K_VALUES:

    candidates = {

        "K-means": KMeans(
            n_clusters=k,
            n_init=50,
            random_state=RANDOM_STATE
        ),

        "GMM": GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=20,
            random_state=RANDOM_STATE
        ),

        "Ward": AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )
    }


    for method, estimator in candidates.items():

        labels = estimator.fit_predict(X)

        records.append({

            "Method": method,
            "K": k,

            "Silhouette":
                silhouette_score(X, labels),

            "Calinski_Harabasz":
                calinski_harabasz_score(X, labels),

            "Davies_Bouldin":
                davies_bouldin_score(X, labels)
        })


comparison = pd.DataFrame(records)


comparison.to_csv(
    RESULT_DIR
    / "12_Supplementary_Clustering_Comparison.csv",
    index=False
)


# =============================================================================
# 6. METRIC SETTINGS
# =============================================================================

metrics = [

    (
        "Silhouette",
        "Silhouette score",
        "Higher is better"
    ),

    (
        "Calinski_Harabasz",
        "Calinski–Harabasz score",
        "Higher is better"
    ),

    (
        "Davies_Bouldin",
        "Davies–Bouldin index",
        "Lower is better"
    )
]


# =============================================================================
# 7. CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16.8, 5.9),
    sharex=True
)


# =============================================================================
# 8. DRAW THREE METRIC PANELS
# =============================================================================

for axis, (column, ylabel, note) in zip(
    axes,
    metrics
):


    # -------------------------------------------------------------------------
    # 8.1 Highlight selected K = 3
    # -------------------------------------------------------------------------

    axis.axvspan(
        2.88,
        3.12,
        color="#8FA4B4",
        alpha=0.10,
        zorder=0
    )

    axis.axvline(
        3,
        color="#718596",
        linestyle=(0, (3, 3)),
        linewidth=1.3,
        alpha=0.90,
        zorder=1
    )


    # -------------------------------------------------------------------------
    # 8.2 Method curves
    # -------------------------------------------------------------------------

    for method in METHODS:

        subset = (
            comparison[
                comparison["Method"] == method
            ]
            .sort_values("K")
        )

        axis.plot(
            subset["K"],
            subset[column],

            marker="o",
            markersize=8.5,

            linewidth=2.4,

            color=METHOD_COLORS[method],

            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=1.0,

            label=method,

            zorder=3
        )


    # -------------------------------------------------------------------------
    # 8.3 Axes
    # -------------------------------------------------------------------------

    axis.set_xticks(
        list(K_VALUES)
    )

    axis.set_xlim(
        1.8,
        6.2
    )

    axis.set_xlabel(
        "Number of clusters (K)",
        labelpad=6
    )

    axis.set_ylabel(
        ylabel,
        labelpad=7
    )


    # -------------------------------------------------------------------------
    # 8.4 Higher / lower note
    # -------------------------------------------------------------------------

    axis.text(
        0.03,
        0.045,

        note,

        transform=axis.transAxes,

        fontsize=22,
        color="#667784",

        ha="left",
        va="bottom"
    )


    # -------------------------------------------------------------------------
    # 8.5 Grid
    # -------------------------------------------------------------------------

    axis.grid(
        axis="y",
        color="#C7CDD2",
        linewidth=0.65,
        alpha=0.55
    )

    axis.set_axisbelow(True)


    # -------------------------------------------------------------------------
    # 8.6 Clean borders
    # -------------------------------------------------------------------------

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)

    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")


# =============================================================================
# 9. ONE SHARED LEGEND
# =============================================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,

    loc="upper center",

    bbox_to_anchor=(0.5, 1.015),

    ncol=3,

    frameon=False,

    handlelength=2.1,
    handletextpad=0.55,
    columnspacing=1.8
)


# =============================================================================
# 10. LAYOUT
# =============================================================================

fig.tight_layout(
    rect=[0.00, 0.00, 1.00, 0.90],
    pad=0.7,
    w_pad=2.1
)


# =============================================================================
# 11. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_S1a_Internal_Clustering_Quality.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_S1a_Internal_Clustering_Quality.pdf"
)


fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)


plt.show()
plt.close(fig)


# =============================================================================
# 12. OUTPUT SUMMARY
# =============================================================================

print("\nSupplementary Figure S1(a) files saved:")

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)

print(
    "Comparison table: "
    f"{RESULT_DIR / '12_Supplementary_Clustering_Comparison.csv'}"
)


# In[15]:


"""Supplementary Figure S1(b): bootstrap stability of shortlisted solutions."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parent.parent
SCORE_FILE = ROOT / "Results" / "01_State_Discovery" / "04_PCA_Scores.csv"
RESULT_DIR = ROOT / "Results" / "01_State_Discovery"
OUTPUT_DIR = ROOT / "Figures" / "Supplementary_Figures" / "Figure_S1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_RUNS = 200
RANDOM_STATE = 42
SOLUTIONS = [("K-means", 2), ("K-means", 3), ("GMM", 2), ("GMM", 3)]
COLORS = ["#0072B2", "#0072B2", "#D55E00", "#D55E00"]

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 20, "font.weight": "normal",
    "axes.labelsize": 19, "axes.labelweight": "normal",
    "xtick.labelsize": 17, "ytick.labelsize": 18,
    "pdf.fonttype": 42, "ps.fonttype": 42
})


def make_clusterer(method, k, seed):
    if method == "K-means":
        return KMeans(n_clusters=k, n_init=50, random_state=seed)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=20,
                           random_state=seed)


# Saved PC1-PC2 values are the sole input; final state labels remain unchanged.
scores = pd.read_csv(SCORE_FILE)
if not {"PC1", "PC2"}.issubset(scores.columns):
    raise ValueError("Saved PCA score table must contain PC1 and PC2.")
X = scores[["PC1", "PC2"]].to_numpy(float)
if not np.isfinite(X).all():
    raise ValueError("Non-finite PC scores were found.")

rng = np.random.default_rng(RANDOM_STATE)
records = []
for method, k in SOLUTIONS:
    reference = make_clusterer(method, k, RANDOM_STATE).fit_predict(X)
    for run in range(1, BOOTSTRAP_RUNS + 1):
        indices = rng.integers(0, len(X), size=len(X))
        bootstrap_model = make_clusterer(method, k, RANDOM_STATE + run)
        bootstrap_model.fit(X[indices])
        resampled_labels = bootstrap_model.predict(X)
        records.append({
            "Solution": f"{method}, K={k}", "Method": method, "K": k,
            "Bootstrap_Run": run,
            "Adjusted_Rand_Index": adjusted_rand_score(reference, resampled_labels)
        })

stability = pd.DataFrame(records)
summary = stability.groupby(["Solution", "Method", "K"], as_index=False)["Adjusted_Rand_Index"].agg(
    Mean_ARI="mean", Median_ARI="median", SD_ARI="std", Minimum_ARI="min"
)
summary["ARI_Above_0_80_Percent"] = [
    (stability.loc[stability["Solution"] == solution, "Adjusted_Rand_Index"] >= 0.80).mean() * 100
    for solution in summary["Solution"]
]
stability.to_csv(RESULT_DIR / "13_Supplementary_Bootstrap_ARI_Runs.csv", index=False)
summary.to_csv(RESULT_DIR / "13b_Supplementary_Bootstrap_ARI_Summary.csv", index=False)


# Standalone stability distribution: no main title and no panel label.
labels = [f"K-means\nK=2", f"K-means\nK=3", f"GMM\nK=2", f"GMM\nK=3"]
data = [
    stability.loc[(stability["Method"] == method) & (stability["K"] == k), "Adjusted_Rand_Index"].to_numpy()
    for method, k in SOLUTIONS
]

fig, ax = plt.subplots(figsize=(10.8, 6.2))
violins = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
for body, color in zip(violins["bodies"], COLORS):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.38)

box = ax.boxplot(data, widths=0.22, patch_artist=True, showfliers=False,
                 medianprops={"color": "black", "linewidth": 1.5},
                 whiskerprops={"color": "#34495E", "linewidth": 1.0},
                 capprops={"color": "#34495E", "linewidth": 1.0})
for patch, color in zip(box["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.90)
    patch.set_edgecolor("white")

ax.axhline(0.80, color="#5D6D7E", linestyle="--", linewidth=1.3)
ax.text(4.42, 0.805, "ARI = 0.80", fontsize=15, color="#566573", ha="right")
ax.set_xticks(range(1, 5), labels)
ax.set_ylabel("Adjusted Rand index (ARI)", labelpad=12)
ax.set_ylim(0.20, 1.03)
ax.grid(axis="y", color="#BFC5CA", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.8)

tiff_file = OUTPUT_DIR / "Figure_S1b_Bootstrap_Cluster_Stability.tiff"
pdf_file = OUTPUT_DIR / "Figure_S1b_Bootstrap_Cluster_Stability.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nSupplementary Figure S1(b) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")
print("\nBootstrap stability summary:")
print(summary.round(3).to_string(index=False))


# In[27]:


"""Supplementary Figure S1(c): sensitivity to retaining PC3 – modern lollipop plot."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parent.parent
SCORE_FILE = ROOT / "Results" / "01_State_Discovery" / "04_PCA_Scores.csv"
RESULT_DIR = ROOT / "Results" / "01_State_Discovery"
OUTPUT_DIR = ROOT / "Figures" / "Supplementary_Figures" / "Figure_S1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SOLUTIONS = [("K-means", 2), ("K-means", 3), ("GMM", 2), ("GMM", 3)]
# Colours: one for K‑means, one for GMM
METHOD_COLORS = {"K-means": "#0072B2", "GMM": "#D55E00"}

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 18,
    "font.weight": "normal",
    "axes.labelsize": 20,
    "axes.labelweight": "normal",
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def cluster(method, k, X):
    if method == "K-means":
        return KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(X)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=20,
                           random_state=RANDOM_STATE).fit_predict(X)


# Compute sensitivity scores
scores = pd.read_csv(SCORE_FILE)
if not {"PC1", "PC2", "PC3"}.issubset(scores.columns):
    raise ValueError("Saved PCA score table must contain PC1, PC2 and PC3.")
X_2pc = scores[["PC1", "PC2"]].to_numpy(float)
X_3pc = scores[["PC1", "PC2", "PC3"]].to_numpy(float)

records = []
for method, k in SOLUTIONS:
    labels_2pc = cluster(method, k, X_2pc)
    labels_3pc = cluster(method, k, X_3pc)
    records.append({
        "Method": method, "K": k,
        "ARI_PC1_PC2_vs_PC1_PC3": adjusted_rand_score(labels_2pc, labels_3pc)
    })

sensitivity = pd.DataFrame(records)
sensitivity["Solution"] = sensitivity["Method"] + ", K=" + sensitivity["K"].astype(str)
sensitivity.to_csv(RESULT_DIR / "14_Supplementary_PC_Dimension_Sensitivity.csv", index=False)


# =============================================================================
# MODERN LOLLIPOP PLOT (replaces the old bar chart)
# =============================================================================
plot_data = sensitivity.copy()
# Order from high to low for a nicer display, but keep as given
plot_data = plot_data.sort_values("ARI_PC1_PC2_vs_PC1_PC3", ascending=True)

fig, ax = plt.subplots(figsize=(9.0, 4.5))

# Draw a horizontal line at zero (baseline)
ax.axvline(0, color="#7F8C8D", linewidth=1.0, zorder=0)

# Plot lollipops: stem + circle
for i, row in plot_data.iterrows():
    method = row["Method"]
    ari = row["ARI_PC1_PC2_vs_PC1_PC3"]
    label = row["Solution"]
    colour = METHOD_COLORS[method]
    # Stem (thin line)
    ax.plot([0, ari], [i, i], color=colour, lw=1.2, zorder=1)
    # Circle (head)
    ax.scatter(ari, i, s=120, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
    # ARI value next to the dot
    ax.text(ari + 0.005, i, f"{ari:.3f}", va="center", ha="left",
            fontsize=16, fontweight="normal", color="#2C3E50", zorder=4)

# Reference line at ARI = 0.90
ax.axvline(0.90, color="#5D6D7E", linestyle="--", linewidth=1.5, zorder=0,
           alpha=0.8)
ax.text(0.90 + 0.005, len(plot_data) - 0.4, "ARI = 0.90",
        fontsize=16, color="#566573", va="bottom", ha="left")

# Set y‑axis labels (solution names)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data["Solution"], fontsize=17)
ax.set_xlim(0.65, 1.03)   # leave room for labels
ax.set_xlabel("ARI between PC1–PC2 and PC1–PC3 labels", labelpad=8)
ax.grid(axis="x", color="#BFC5CA", linewidth=0.6, alpha=0.55)
ax.set_axisbelow(True)
# Remove top and right spines, but keep bottom and left
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color("#7F8C8D")
ax.spines["bottom"].set_color("#7F8C8D")

fig.tight_layout(pad=0.6)

# Save
tiff_file = OUTPUT_DIR / "Figure_S1c_PC_Dimension_Sensitivity.tiff"
pdf_file = OUTPUT_DIR / "Figure_S1c_PC_Dimension_Sensitivity.pdf"
fig.savefig(tiff_file, dpi=1000, format="tiff", bbox_inches="tight",
            pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()
plt.close(fig)

print("\nSupplementary Figure S1(c) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")
print("\nPC-dimension sensitivity:")
print(sensitivity.round(3).to_string(index=False))


# In[1]:


"""Standardised environmental profiles of the final ecological states."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "Results" / "02_Environmental_Analysis"
PROFILE_FILE = RESULT_DIR / "05_Standardized_Environmental_State_Means.csv"
TEST_FILE = RESULT_DIR / "06_Environmental_State_Tests.csv"

OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 20,
    "font.weight": "normal",

    "axes.labelsize": 22,
    "axes.labelweight": "normal",

    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    # Make mathematical notation visually consistent with Calibri
    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. PUBLICATION-QUALITY FEATURE LABELS
# =============================================================================

# Raw dataframe names remain unchanged in the analysis.
# These labels are used only for displaying variables in the figure.

FEATURE_LABELS = {

    # Nutrients / carbonate chemistry
    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    # Surface seawater pCO2
    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    # Physical / environmental variables
    "SSS": "SSS",
    "SST": "SST",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    # Marine heatwave variables
    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    # Alternative shortened names, if present in saved files
    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    # Climate indices
    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO",
}


# =============================================================================
# 4. LOAD SAVED RESULTS
# =============================================================================

# Saved standardised state means.
profile = pd.read_csv(PROFILE_FILE, index_col="Variable")
tests = pd.read_csv(TEST_FILE)

required = ["State_1", "State_2", "State_3"]

missing = [
    column for column in required
    if column not in profile.columns
]

if missing:
    raise ValueError(
        f"Missing state-profile columns: {missing}"
    )

if not {"Variable", "Epsilon_Squared"}.issubset(tests.columns):
    raise ValueError(
        "Environmental test table must contain "
        "Variable and Epsilon_Squared."
    )


# =============================================================================
# 5. ORDER VARIABLES BY STATE-SEPARATION EFFECT SIZE
# =============================================================================

variable_order = [
    name for name in tests["Variable"]
    if name in profile.index
]

profile = profile.loc[variable_order, required]


# Publication-quality labels corresponding to the ordered variables
display_labels = [
    FEATURE_LABELS.get(variable, variable)
    for variable in profile.index
]


# =============================================================================
# 6. CREATE HEATMAP
# =============================================================================

# Standalone panel:
# - no main title
# - no panel label
# - symmetric colour scale around zero

fig, ax = plt.subplots(figsize=(8.8, 10.0))

limit = max(
    1.05,
    np.abs(profile.to_numpy()).max()
)

image = ax.imshow(
    profile.to_numpy(),
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-limit,
        vcenter=0,
        vmax=limit
    ),
    aspect="auto"
)


# =============================================================================
# 7. AXES AND SCIENTIFIC LABELS
# =============================================================================

ax.set_xticks(
    range(3),
    ["State 1", "State 2", "State 3"]
)

ax.set_yticks(
    range(len(profile.index)),
    display_labels
)

ax.set_xlabel(
    "Ecological state",
    labelpad=8
)

ax.set_ylabel(
    "Environmental variable",
    labelpad=12
)


# =============================================================================
# 8. ADD CELL VALUES
# =============================================================================

for row in range(profile.shape[0]):

    for column in range(profile.shape[1]):

        value = profile.iloc[row, column]

        text_colour = (
            "white"
            if abs(value) > limit * 0.58
            else "black"
        )

        ax.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=16,
            color=text_colour,
            family="Calibri"
        )


# =============================================================================
# 9. CELL SEPARATION LINES
# =============================================================================

for edge in np.arange(-0.5, 3, 1):
    ax.axvline(
        edge,
        color="white",
        linewidth=1.0
    )

for edge in np.arange(-0.5, len(profile.index), 1):
    ax.axhline(
        edge,
        color="white",
        linewidth=1.0
    )


# =============================================================================
# 10. COLORBAR
# =============================================================================

colorbar = fig.colorbar(
    image,
    ax=ax,
    pad=0.03,
    fraction=0.055
)

colorbar.set_label(
    "Standardised mean",
    rotation=90,
    labelpad=12,
    fontsize=22,
    family="Calibri",
    fontweight="normal"
)

colorbar.ax.tick_params(
    labelsize=18
)


# =============================================================================
# 11. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 12. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR /
    "Figure_3a_Environmental_State_Profile.tiff"
)

pdf_file = (
    OUTPUT_DIR /
    "Figure_3a_Environmental_State_Profile.pdf"
)

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 13. OUTPUT SUMMARY
# =============================================================================

print("\nFigure 3(a) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")

print("\nVariables displayed as:")
for raw_name, display_name in zip(profile.index, display_labels):
    print(f"  {raw_name:<18} -> {display_name}")


# In[11]:


"""Standardised environmental profiles of the final ecological states
as a horizontal three-state dot plot."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "Results" / "02_Environmental_Analysis"
PROFILE_FILE = RESULT_DIR / "05_Standardized_Environmental_State_Means.csv"
TEST_FILE = RESULT_DIR / "06_Environmental_State_Tests.csv"

OUTPUT_DIR = ROOT / "Figures" / "Main_Figures" / "Figure_03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 26,
    "font.weight": "normal",

    "axes.labelsize": 28,
    "axes.labelweight": "normal",

    "xtick.labelsize": 24,
    "ytick.labelsize": 24,

    "legend.fontsize": 24,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. PUBLICATION-QUALITY FEATURE LABELS
# =============================================================================

FEATURE_LABELS = {
    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "SSS": "SSS",
    "SST": "SST",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO",
}


# =============================================================================
# 4. STATE LABELS / COLOURS
# =============================================================================

STATE_INFO = [
    ("State_1", "State 1", "#D55E5E"),   # muted red
    ("State_2", "State 2", "#2A9D8F"),   # teal
    ("State_3", "State 3", "#7B6BB3"),   # soft purple
]

STATE_OFFSETS = {
    "State_1": -0.18,
    "State_2":  0.00,
    "State_3":  0.18,
}


# =============================================================================
# 5. LOAD SAVED RESULTS
# =============================================================================

profile = pd.read_csv(PROFILE_FILE, index_col="Variable")
tests = pd.read_csv(TEST_FILE)

required_profile_cols = ["State_1", "State_2", "State_3"]
missing_profile = [
    col for col in required_profile_cols
    if col not in profile.columns
]

if missing_profile:
    raise ValueError(
        f"Missing state-profile columns: {missing_profile}"
    )

required_test_cols = ["Variable", "Epsilon_Squared"]
missing_tests = [
    col for col in required_test_cols
    if col not in tests.columns
]

if missing_tests:
    raise ValueError(
        f"Missing environmental test columns: {missing_tests}"
    )


# =============================================================================
# 6. ORDER VARIABLES BY EFFECT SIZE
# =============================================================================

tests = tests.sort_values(
    "Epsilon_Squared",
    ascending=False
).reset_index(drop=True)

variable_order = [
    variable for variable in tests["Variable"]
    if variable in profile.index
]

profile = profile.loc[variable_order, required_profile_cols].copy()

display_labels = [
    FEATURE_LABELS.get(variable, variable)
    for variable in profile.index
]


# =============================================================================
# 7. CREATE LONG DATA FOR PLOTTING
# =============================================================================

plot_rows = []

for i, variable in enumerate(profile.index):

    for state_key, state_label, state_color in STATE_INFO:

        plot_rows.append({
            "Variable": variable,
            "Display_Label": FEATURE_LABELS.get(variable, variable),
            "State_Key": state_key,
            "State_Label": state_label,
            "Color": state_color,
            "Value": float(profile.loc[variable, state_key]),
            "Y_Base": i,
            "Y": i + STATE_OFFSETS[state_key],
        })

plot_df = pd.DataFrame(plot_rows)


# =============================================================================
# 8. X-AXIS LIMITS
# =============================================================================

max_abs_value = np.abs(profile.to_numpy()).max()
x_limit = max(1.10, np.ceil((max_abs_value + 0.10) * 10) / 10)


# =============================================================================
# 9. CREATE DOT PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10.8, 7.8))


# -------------------------------------------------------------------------
# 9.1 Light connecting range line for each variable
# -------------------------------------------------------------------------

for i, variable in enumerate(profile.index):

    row = profile.loc[variable]
    xmin = float(row.min())
    xmax = float(row.max())

    ax.hlines(
        y=i,
        xmin=xmin,
        xmax=xmax,
        color="#B8BDC3",
        linewidth=1.6,
        zorder=1
    )


# -------------------------------------------------------------------------
# 9.2 Three state points
# -------------------------------------------------------------------------

for state_key, state_label, state_color in STATE_INFO:

    subset = plot_df.loc[
        plot_df["State_Key"] == state_key
    ].copy()

    ax.scatter(
        subset["Value"],
        subset["Y"],
        s=105,
        color=state_color,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.96,
        label=state_label,
        zorder=3
    )


# -------------------------------------------------------------------------
# 9.3 Zero reference line
# -------------------------------------------------------------------------

ax.axvline(
    0,
    color="#666666",
    linewidth=1.2,
    linestyle=(0, (4, 4)),
    zorder=0
)


# =============================================================================
# 10. AXES
# =============================================================================

ax.set_yticks(
    np.arange(len(profile.index)),
    display_labels
)

ax.invert_yaxis()

ax.set_xlabel(
    "Standardised mean",
    labelpad=10
)

ax.set_ylabel(
    "Environmental variable",
    labelpad=10
)

ax.set_xlim(-x_limit, x_limit)


# =============================================================================
# 11. GRID / SPINES
# =============================================================================

ax.grid(
    axis="x",
    color="#D3D7DB",
    linewidth=0.65,
    alpha=0.55
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# =============================================================================
# 12. LEGEND
# =============================================================================

ax.legend(
    loc="lower right",
    bbox_to_anchor=(0.985, 0.02),
    ncol=1,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.6,
    labelspacing=0.45
)

# =============================================================================
# 13. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.8)


# =============================================================================
# 14. SAVE FIGURE
# =============================================================================

tiff_file = OUTPUT_DIR / "Figure_3a_Environmental_State_Profile_DotPlot.tiff"
pdf_file = OUTPUT_DIR / "Figure_3a_Environmental_State_Profile_DotPlot.pdf"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 15. OUTPUT SUMMARY
# =============================================================================

print("\nFigure 3(a) dot-plot files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")

print("\nVariables displayed as:")
for raw_name, display_name in zip(profile.index, display_labels):
    print(f"  {raw_name:<18} -> {display_name}")


# In[8]:


"""Ranked environmental effect sizes across the final ecological states."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT
    / "Results"
    / "02_Environmental_Analysis"
    / "06_Environmental_State_Tests.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_03"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 28,
    "font.weight": "normal",

    "axes.labelsize": 32,
    "axes.labelweight": "normal",

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 26,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 3. PUBLICATION-QUALITY FEATURE LABELS
# =============================================================================

FEATURE_LABELS = {

    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "SSS": "SSS",
    "SST": "SST",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO",
}


# =============================================================================
# 4. LOAD SAVED STATISTICAL RESULTS
# =============================================================================

tests = pd.read_csv(DATA_FILE)

required = [
    "Variable",
    "Epsilon_Squared",
    "Significant_FDR"
]

missing = [
    column for column in required
    if column not in tests.columns
]

if missing:
    raise ValueError(
        f"Missing environmental-test columns: {missing}"
    )


# =============================================================================
# 5. PREPARE DATA
# =============================================================================

tests = (
    tests
    .sort_values("Epsilon_Squared", ascending=False)
    .reset_index(drop=True)
)

tests["Display_Label"] = [
    FEATURE_LABELS.get(variable, variable)
    for variable in tests["Variable"]
]

effect_sizes = tests["Epsilon_Squared"].to_numpy()
significant = tests["Significant_FDR"].astype(bool).to_numpy()
labels = tests["Display_Label"].tolist()
y_positions = np.arange(len(tests))


# =============================================================================
# 6. COLOURS
# =============================================================================

SIG_COLOR = "#C65D18"
NS_COLOR = "#9DA4AA"
STEM_COLOR = "#AEB4B9"

SMALL_COLOR = "#7F8C8D"
MODERATE_COLOR = "#477A9F"
LARGE_COLOR = "#9E4B47"


# =============================================================================
# 7. CREATE LOLLIPOP / CLEVELAND PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(12.2, 9.0))


# -----------------------------------------------------------------------------
# 7.1 Lollipop stems
# -----------------------------------------------------------------------------

for y, effect, is_sig in zip(y_positions, effect_sizes, significant):

    stem_colour = SIG_COLOR if is_sig else STEM_COLOR

    ax.hlines(
        y=y,
        xmin=0,
        xmax=effect,
        color=stem_colour,
        linewidth=2.2,
        alpha=0.75,
        zorder=1
    )


# -----------------------------------------------------------------------------
# 7.2 End-point markers
# -----------------------------------------------------------------------------

for y, effect, is_sig in zip(y_positions, effect_sizes, significant):

    if is_sig:
        ax.scatter(
            effect,
            y,
            s=165,
            color=SIG_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=3
        )
    else:
        ax.scatter(
            effect,
            y,
            s=155,
            facecolor="white",
            edgecolor=NS_COLOR,
            linewidth=2.0,
            zorder=3
        )


# =============================================================================
# 8. EFFECT-SIZE VALUE LABELS
# =============================================================================

x_max = max(0.60, effect_sizes.max() * 1.15)
value_offset = x_max * 0.012

for y, effect in zip(y_positions, effect_sizes):
    ax.text(
        effect + value_offset,
        y,
        f"{effect:.2f}",
        ha="left",
        va="center",
        fontsize=22,
        color="#3F4448"
    )


# =============================================================================
# 9. EFFECT-SIZE REFERENCE THRESHOLDS
# =============================================================================

thresholds = [
    (0.01, "Small", SMALL_COLOR),
    (0.06, "Moderate", MODERATE_COLOR),
    (0.14, "Large", LARGE_COLOR),
]

for value, label, colour in thresholds:

    ax.axvline(
        value,
        color=colour,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        alpha=0.90,
        zorder=0
    )

    ax.text(
        value + 0.004,
        0.985,
        label,
        color=colour,
        fontsize=20,
        ha="left",
        va="top",
        transform=ax.get_xaxis_transform()
    )


# =============================================================================
# 10. AXES
# =============================================================================

ax.set_yticks(y_positions, labels)
ax.invert_yaxis()

ax.set_xlim(0, x_max)

ax.set_xlabel(
    r"Kruskal–Wallis effect size ($\varepsilon^2$)",
    labelpad=12
)

ax.set_ylabel(
    "Environmental variable",
    labelpad=14
)


# =============================================================================
# 11. GRID AND SPINES
# =============================================================================

ax.grid(
    axis="x",
    color="#D4D7DA",
    linewidth=0.7,
    alpha=0.50
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)


# =============================================================================
# 12. LEGEND INSIDE THE PLOT
# =============================================================================

legend_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=SIG_COLOR,
        markeredgecolor="white",
        markersize=14,
        label="FDR-significant (adjusted P < 0.05)"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor=NS_COLOR,
        markeredgewidth=1.8,
        markersize=14,
        label="Not FDR-significant"
    )
]

ax.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=False,
    handletextpad=0.6,
    borderaxespad=0.5
)


# =============================================================================
# 13. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.9)


# =============================================================================
# 14. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_3b_Environmental_Effect_Sizes.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_3b_Environmental_Effect_Sizes.pdf"
)

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 15. OUTPUT SUMMARY
# =============================================================================

print("\nFigure 3(b) files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")

print("\nVariables and effect sizes:")
print("-" * 68)

for _, row in tests.iterrows():

    status = (
        "FDR significant"
        if bool(row["Significant_FDR"])
        else "Not significant"
    )

    print(
        f"{row['Variable']:<18} "
        f"epsilon² = {row['Epsilon_Squared']:.3f}   "
        f"{status}"
    )


# In[29]:


"""Create six individual publication-quality confusion matrices
from saved LOYO predictions.

The plotting style is kept identical to the finalized single-model
confusion-matrix script. Only the model and colour map change.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

CM_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "04_Confusion_Matrix_Counts.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_Confusion_Matrices"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SIX FINAL MODELS
# =============================================================================

MODELS = [
    "CatBoost",
    "XGBoost",
    "HistGradientBoosting",
    "Equal Soft Voting",
    "TCN",
    "CNN-LSTM",
]


# Titles used in the figures
MODEL_TITLES = {
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "HistGradientBoosting": "HistGradientBoosting",
    "Equal Soft Voting": "Soft Voting",
    "TCN": "TCN",
    "CNN-LSTM": "CNN-LSTM",
}


# =============================================================================
# 3. FIGURE STYLE
# EXACTLY AS IN YOUR FINAL REFERENCE SCRIPT
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 24,

    "axes.labelsize": 28,

    "xtick.labelsize": 23,
    "ytick.labelsize": 23,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. DIFFERENT COLOUR MAP FOR EACH MODEL
# =============================================================================

MODEL_CMAPS = {

    # -------------------------------------------------------------------------
    # CatBoost — Blue
    # -------------------------------------------------------------------------
    "CatBoost": LinearSegmentedColormap.from_list(
        "CM_Blue",
        [
            "#F7FAFD",
            "#DCE8F3",
            "#A9C4DE",
            "#7299BE",
            "#4E79A7",
            "#2E5B87"
        ]
    ),

    # -------------------------------------------------------------------------
    # XGBoost — Orange
    # -------------------------------------------------------------------------
    "XGBoost": LinearSegmentedColormap.from_list(
        "CM_Orange",
        [
            "#FFF9F3",
            "#FDE7D0",
            "#F4C18F",
            "#E99A57",
            "#D8732E",
            "#A94A13"
        ]
    ),

    # -------------------------------------------------------------------------
    # HistGradientBoosting — Green
    # -------------------------------------------------------------------------
    "HistGradientBoosting": LinearSegmentedColormap.from_list(
        "CM_Green",
        [
            "#F5FBF8",
            "#D9EEE4",
            "#A9D7C1",
            "#72BC9B",
            "#419D77",
            "#24765A"
        ]
    ),

    # -------------------------------------------------------------------------
    # Soft Voting — Purple
    # -------------------------------------------------------------------------
    "Equal Soft Voting": LinearSegmentedColormap.from_list(
        "CM_Purple",
        [
            "#FAF8FC",
            "#E8E0F0",
            "#C9B8DD",
            "#A28CC4",
            "#7B64A6",
            "#5C4285"
        ]
    ),

    # -------------------------------------------------------------------------
    # TCN — Red
    # -------------------------------------------------------------------------
    "TCN": LinearSegmentedColormap.from_list(
        "CM_Red",
        [
            "#FFF8F7",
            "#F6DEDA",
            "#E7B3AA",
            "#D48173",
            "#BC5544",
            "#933527"
        ]
    ),

    # -------------------------------------------------------------------------
    # CNN-LSTM — Teal
    # -------------------------------------------------------------------------
    "CNN-LSTM": LinearSegmentedColormap.from_list(
        "CM_Teal",
        [
            "#F4FBFB",
            "#D4EEEE",
            "#9FD5D4",
            "#68B7B5",
            "#3A9290",
            "#226D6B"
        ]
    ),
}


# =============================================================================
# 5. LOAD SAVED CONFUSION COUNTS
# =============================================================================

data = pd.read_csv(CM_FILE)

required = {
    "Model",
    "True_State",
    "Predicted_State",
    "Count"
}

missing = required.difference(data.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# =============================================================================
# 6. STATE SETTINGS
# =============================================================================

STATES = [1, 2, 3]

state_labels = [
    "State 1",
    "State 2",
    "State 3"
]


# =============================================================================
# 7. CREATE ONE SEPARATE FIGURE FOR EACH MODEL
# =============================================================================

for MODEL in MODELS:


    # =========================================================================
    # 7.1 SELECT MODEL
    # =========================================================================

    model_df = data.loc[
        data["Model"] == MODEL
    ].copy()

    if model_df.empty:
        raise ValueError(
            f"Model '{MODEL}' was not found.\n"
            f"Available models:\n"
            f"{sorted(data['Model'].unique())}"
        )


    # =========================================================================
    # 7.2 BUILD 3 × 3 CONFUSION MATRIX
    # =========================================================================

    cm = (
        model_df
        .pivot(
            index="True_State",
            columns="Predicted_State",
            values="Count"
        )
        .reindex(
            index=STATES,
            columns=STATES,
            fill_value=0
        )
        .fillna(0)
        .to_numpy(dtype=int)
    )


    # =========================================================================
    # 7.3 ROW-NORMALIZED PERCENTAGES
    # =========================================================================

    row_totals = cm.sum(
        axis=1,
        keepdims=True
    )

    cm_percent = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_totals != 0
    ) * 100


    print(
        f"\n{MODEL} confusion matrix:"
    )

    print(cm)

    print(
        "\nRow-normalized percentages:"
    )

    print(
        np.round(
            cm_percent,
            1
        )
    )


    # =========================================================================
    # 7.4 CREATE FIGURE
    # EXACTLY SAME SIZE AS FINAL CATBOOST VERSION
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.6, 7.0)
    )


    # =========================================================================
    # 7.5 MATRIX
    # =========================================================================

    image = ax.imshow(
        cm_percent,
        cmap=MODEL_CMAPS[MODEL],
        vmin=0,
        vmax=100,
        interpolation="nearest"
    )


    # =========================================================================
    # 7.6 TITLE
    # =========================================================================

    ax.set_title(
        MODEL_TITLES[MODEL],
        fontsize=28,
        fontweight="normal",
        pad=14
    )


    # =========================================================================
    # 7.7 AXES
    # =========================================================================

    ax.set_xticks(
        np.arange(3),
        state_labels
    )

    ax.set_yticks(
        np.arange(3),
        state_labels
    )


    ax.set_xlabel(
        "Predicted state",
        labelpad=12
    )

    ax.set_ylabel(
        "True state",
        labelpad=12
    )


    # =========================================================================
    # 7.8 CELL ANNOTATIONS
    # =========================================================================

    for i in range(3):

        for j in range(3):

            percentage = cm_percent[i, j]
            count = cm[i, j]


            # White text for darker cells
            text_color = (
                "white"
                if percentage >= 50
                else "#222222"
            )


            # -----------------------------------------------------------------
            # Count
            # -----------------------------------------------------------------

            ax.text(
                j,
                i - 0.08,
                f"{count}",
                ha="center",
                va="center",
                fontsize=26,
                fontweight="bold",
                color=text_color
            )


            # -----------------------------------------------------------------
            # Percentage
            # -----------------------------------------------------------------

            ax.text(
                j,
                i + 0.17,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=19,
                color=text_color
            )


    # =========================================================================
    # 7.9 CELL GRID
    # =========================================================================

    ax.set_xticks(
        np.arange(-0.5, 3, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 3, 1),
        minor=True
    )


    ax.grid(
        which="minor",
        color="white",
        linewidth=2.0
    )


    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )


    # =========================================================================
    # 7.10 FULL OUTER BOX
    # =========================================================================

    for spine in ax.spines.values():

        spine.set_visible(True)

        spine.set_linewidth(1.2)

        spine.set_color(
            "#4A4A4A"
        )


    # =========================================================================
    # 7.11 COLORBAR
    # =========================================================================

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )


    colorbar.set_label(
        "Within-state prediction (%)",
        fontsize=23,
        labelpad=12
    )


    colorbar.ax.tick_params(
        labelsize=19
    )


    # =========================================================================
    # 7.12 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 7.13 SAVE INDIVIDUAL MODEL
    # =========================================================================

    safe_model = (
        MODEL
        .replace(" ", "_")
        .replace("-", "_")
    )


    tiff_file = (
        OUTPUT_DIR
        / f"Confusion_Matrix_{safe_model}.tiff"
    )

    pdf_file = (
        OUTPUT_DIR
        / f"Confusion_Matrix_{safe_model}.pdf"
    )


    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(
        pdf_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04
    )


    plt.show()

    plt.close(fig)


    print(
        f"\n{MODEL} confusion matrix saved:"
    )

    print(
        f"TIFF (1000 dpi): {tiff_file}"
    )

    print(
        f"PDF (vector)   : {pdf_file}"
    )


# =============================================================================
# 8. FINISHED
# =============================================================================

print(
    "\nAll six individual confusion matrices have been created."
)

print(
    f"Output folder: {OUTPUT_DIR}"
)


# In[30]:


"""Publication-quality multiclass ROC curve for CatBoost
using saved LOYO held-out probabilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. MODEL
# =============================================================================

MODEL = "CatBoost"

TRUE_COL = "True_State"

PROB_COLS = [
    "CatBoost_P_State_1",
    "CatBoost_P_State_2",
    "CatBoost_P_State_3",
]


# =============================================================================
# 3. FIGURE STYLE
# Similar visual weight to the finalized confusion matrix
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 24,

    "axes.labelsize": 28,

    "xtick.labelsize": 23,
    "ytick.labelsize": 23,

    "legend.fontsize": 19,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


required = [
    TRUE_COL,
    *PROB_COLS
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. TRUE STATES AND HELD-OUT PROBABILITIES
# =============================================================================

y_true = data[
    TRUE_COL
].astype(int).to_numpy()


probability = data[
    PROB_COLS
].astype(float).to_numpy()


if not np.isfinite(
    probability
).all():

    raise ValueError(
        "Non-finite CatBoost probabilities were found."
    )


# Check probability sums
probability_sum = probability.sum(
    axis=1
)

if not np.allclose(
    probability_sum,
    1.0,
    atol=1e-5
):

    print(
        "Warning: some class probabilities do not sum exactly to 1."
    )


# =============================================================================
# 6. ONE-VS-REST BINARY TARGETS
# =============================================================================

STATES = [
    1,
    2,
    3
]

y_binary = label_binarize(
    y_true,
    classes=STATES
)


# =============================================================================
# 7. STATE-SPECIFIC ROC CURVES
# =============================================================================

fpr = {}
tpr = {}
roc_auc = {}


for state_index, state in enumerate(STATES):

    fpr[state], tpr[state], _ = roc_curve(
        y_binary[:, state_index],
        probability[:, state_index]
    )

    roc_auc[state] = auc(
        fpr[state],
        tpr[state]
    )


# =============================================================================
# 8. MACRO-AVERAGE ROC CURVE
# =============================================================================

# Combine all unique FPR values
all_fpr = np.unique(
    np.concatenate(
        [
            fpr[state]
            for state in STATES
        ]
    )
)


# Interpolate the TPR of each state at the common FPR values
mean_tpr = np.zeros_like(
    all_fpr
)

for state in STATES:

    mean_tpr += np.interp(
        all_fpr,
        fpr[state],
        tpr[state]
    )


mean_tpr /= len(
    STATES
)


macro_auc_curve = auc(
    all_fpr,
    mean_tpr
)


# Direct macro-AUROC check
macro_auc_saved_definition = roc_auc_score(
    y_binary,
    probability,
    average="macro",
    multi_class="ovr"
)


# =============================================================================
# 9. PRINT AUC VALUES
# =============================================================================

print(
    "\nCatBoost LOYO ROC/AUC"
)

print(
    "=" * 45
)

for state in STATES:

    print(
        f"State {state} AUC       : "
        f"{roc_auc[state]:.3f}"
    )


print(
    f"Macro-average AUC : "
    f"{macro_auc_saved_definition:.3f}"
)

print(
    f"Macro curve AUC   : "
    f"{macro_auc_curve:.3f}"
)


# =============================================================================
# 10. COLOURS
# Use the same state colours as the ecological-state figures
# =============================================================================

STATE_COLORS = {

    1: "#D55E5E",   # State 1 — muted red

    2: "#2A9D8F",   # State 2 — teal

    3: "#7B6BB3",   # State 3 — soft purple
}


# =============================================================================
# 11. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.6, 7.0)
)


# =============================================================================
# 12. STATE-SPECIFIC ROC CURVES
# =============================================================================

for state in STATES:

    ax.plot(

        fpr[state],
        tpr[state],

        color=STATE_COLORS[state],

        linewidth=2.8,

        label=(
            f"State {state} "
            f"(AUC = {roc_auc[state]:.2f})"
        ),

        zorder=3
    )


# =============================================================================
# 13. MACRO-AVERAGE CURVE
# =============================================================================

ax.plot(

    all_fpr,
    mean_tpr,

    color="#222222",

    linewidth=3.2,

    linestyle="-",

    label=(
        "Macro-average "
        f"(AUC = {macro_auc_saved_definition:.2f})"
    ),

    zorder=4
)


# =============================================================================
# 14. RANDOM-CLASSIFIER REFERENCE
# =============================================================================

ax.plot(

    [0, 1],
    [0, 1],

    color="#8D9499",

    linewidth=1.7,

    linestyle=(0, (5, 5)),

    zorder=1
)


# =============================================================================
# 15. TITLE
# =============================================================================

ax.set_title(

    "CatBoost",

    fontsize=28,

    fontweight="normal",

    pad=14
)


# =============================================================================
# 16. AXES
# =============================================================================

ax.set_xlim(
    -0.01,
    1.01
)

ax.set_ylim(
    -0.01,
    1.01
)


ax.set_xlabel(
    "False positive rate",
    labelpad=12
)

ax.set_ylabel(
    "True positive rate",
    labelpad=12
)


ticks = np.arange(
    0,
    1.01,
    0.2
)

ax.set_xticks(
    ticks
)

ax.set_yticks(
    ticks
)


# =============================================================================
# 17. GRID
# =============================================================================

ax.grid(

    color="#D3D7DB",

    linewidth=0.65,

    alpha=0.50
)

ax.set_axisbelow(
    True
)


# =============================================================================
# 18. FULL OUTER BOX
# =============================================================================

for spine in ax.spines.values():

    spine.set_visible(
        True
    )

    spine.set_linewidth(
        1.2
    )

    spine.set_color(
        "#4A4A4A"
    )


# =============================================================================
# 19. LEGEND
# =============================================================================

ax.legend(

    loc="lower right",

    frameon=True,

    fancybox=False,

    facecolor="white",

    edgecolor="#7A7A7A",

    framealpha=0.94,

    borderpad=0.65,

    handlelength=2.2,

    handletextpad=0.65,

    labelspacing=0.50
)


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 21. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "ROC_AUC_CatBoost.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "ROC_AUC_CatBoost.pdf"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(fig)


# =============================================================================
# 22. OUTPUT SUMMARY
# =============================================================================

print(
    "\nCatBoost ROC files saved:"
)

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)


# In[35]:


"""Publication-quality multiclass ROC curve for CatBoost
using saved LOYO held-out probabilities.

Styled to resemble the provided reference ROC figure:
- step ROC curves
- no full box
- no grid
- clean left/bottom axes only
- compact legend with black border
- coloured legend boxes instead of long line handles
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. MODEL SETTINGS
# =============================================================================

MODEL = "CatBoost"

TRUE_COL = "True_State"

PROB_COLS = [
    "CatBoost_P_State_1",
    "CatBoost_P_State_2",
    "CatBoost_P_State_3",
]


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 22,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(PREDICTION_FILE)

required = [TRUE_COL, *PROB_COLS]
missing = [column for column in required if column not in data.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. TRUE LABELS AND PROBABILITIES
# =============================================================================

y_true = data[TRUE_COL].astype(int).to_numpy()
probability = data[PROB_COLS].astype(float).to_numpy()

if not np.isfinite(probability).all():
    raise ValueError("Non-finite CatBoost probabilities were found.")


# =============================================================================
# 6. BINARIZE TRUE LABELS
# =============================================================================

STATES = [1, 2, 3]

y_binary = label_binarize(
    y_true,
    classes=STATES
)


# =============================================================================
# 7. STATE-SPECIFIC ROC CURVES
# =============================================================================

fpr = {}
tpr = {}
roc_auc = {}

for state_index, state in enumerate(STATES):

    fpr[state], tpr[state], _ = roc_curve(
        y_binary[:, state_index],
        probability[:, state_index]
    )

    roc_auc[state] = auc(
        fpr[state],
        tpr[state]
    )


# =============================================================================
# 8. MACRO-AVERAGE ROC
# =============================================================================

all_fpr = np.unique(
    np.concatenate([fpr[state] for state in STATES])
)

mean_tpr = np.zeros_like(all_fpr)

for state in STATES:
    mean_tpr += np.interp(all_fpr, fpr[state], tpr[state])

mean_tpr /= len(STATES)

macro_auc_curve = auc(all_fpr, mean_tpr)

macro_auc_direct = roc_auc_score(
    y_binary,
    probability,
    average="macro",
    multi_class="ovr"
)


# =============================================================================
# 9. PRINT AUC VALUES
# =============================================================================

print("\nCatBoost LOYO ROC/AUC")
print("=" * 42)

for state in STATES:
    print(f"State {state} AUC       : {roc_auc[state]:.3f}")

print(f"Macro-average AUC : {macro_auc_direct:.3f}")
print(f"Macro curve AUC   : {macro_auc_curve:.3f}")


# =============================================================================
# 10. COLOURS
# =============================================================================

STATE_COLORS = {
    1: "#D55E5E",   # muted red
    2: "#4F9D92",   # teal-green
    3: "#7A6BB1",   # soft purple
}

MACRO_COLOR = "#222222"
DIAGONAL_COLOR = "#9A9A9A"


# =============================================================================
# 11. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(7.6, 7.0))


# =============================================================================
# 12. PLOT STEP ROC CURVES
# Slightly thicker than before
# =============================================================================

for state in STATES:

    ax.step(
        fpr[state],
        tpr[state],
        where="post",
        color=STATE_COLORS[state],
        linewidth=3.4,
        zorder=3
    )


# =============================================================================
# 13. MACRO-AVERAGE CURVE
# =============================================================================

ax.step(
    all_fpr,
    mean_tpr,
    where="post",
    color=MACRO_COLOR,
    linewidth=3.6,
    zorder=4
)


# =============================================================================
# 14. DIAGONAL REFERENCE LINE
# =============================================================================

ax.plot(
    [0, 1],
    [0, 1],
    color=DIAGONAL_COLOR,
    linewidth=1.8,
    linestyle=(0, (4, 6)),
    zorder=1
)


# =============================================================================
# 15. TITLE
# =============================================================================

ax.set_title(
    "CatBoost",
    fontsize=30,
    fontweight="normal",
    pad=12
)


# =============================================================================
# 16. AXES
# =============================================================================

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.04)

ax.set_xlabel(
    "False positive rate",
    labelpad=10
)

ax.set_ylabel(
    "True positive rate",
    labelpad=10
)

ticks = np.arange(0, 1.01, 0.25)

ax.set_xticks(ticks)
ax.set_yticks(ticks)


# =============================================================================
# 17. REMOVE GRID / FULL BOX
# =============================================================================

ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)

ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

ax.tick_params(
    axis="both",
    width=1.8,
    length=6,
    color="black"
)


# =============================================================================
# 18. LEGEND WITH COLOURED BOXES
# =============================================================================

legend_handles = [
    Patch(facecolor=STATE_COLORS[1], edgecolor="none",
          label=f"State 1 (AUC = {roc_auc[1]:.2f})"),
    Patch(facecolor=STATE_COLORS[2], edgecolor="none",
          label=f"State 2 (AUC = {roc_auc[2]:.2f})"),
    Patch(facecolor=STATE_COLORS[3], edgecolor="none",
          label=f"State 3 (AUC = {roc_auc[3]:.2f})"),
    Patch(facecolor=MACRO_COLOR, edgecolor="none",
          label=f"Macro-average (AUC = {macro_auc_direct:.2f})")
]

legend = ax.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
    borderpad=0.55,

    # Smaller coloured boxes
    handlelength=0.65,
    handleheight=0.50,
    handletextpad=0.50,

    labelspacing=0.45
)

legend.get_frame().set_linewidth(1.8)


# =============================================================================
# 19. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.8)


# =============================================================================
# 20. SAVE
# =============================================================================

tiff_file = OUTPUT_DIR / "ROC_AUC_CatBoost.tiff"
pdf_file = OUTPUT_DIR / "ROC_AUC_CatBoost.pdf"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 21. OUTPUT SUMMARY
# =============================================================================

print("\nCatBoost ROC files saved:")
print(f"TIFF (1000 dpi): {tiff_file}")
print(f"PDF (vector)   : {pdf_file}")


# In[36]:


"""Create six individual publication-quality multiclass ROC curves
from saved LOYO held-out probabilities.

Style is kept consistent with the finalized CatBoost ROC figure:
- step ROC curves
- no full box
- no grid
- clean left/bottom axes only
- compact legend with coloured boxes
- different colour combinations for each model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_ROC_AUC"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. MODELS
# =============================================================================

TRUE_COL = "True_State"
STATES = [1, 2, 3]

MODEL_CONFIGS = [

    {
        "model_name": "CatBoost",
        "title": "CatBoost",
        "prefix_candidates": ["CatBoost"],
        "state_colors": {
            1: "#D55E5E",   # muted red
            2: "#4F9D92",   # teal-green
            3: "#7A6BB1"    # soft purple
        }
    },

    {
        "model_name": "XGBoost",
        "title": "XGBoost",
        "prefix_candidates": ["XGBoost", "XGB"],
        "state_colors": {
            1: "#D17C2F",   # warm orange
            2: "#4C78A8",   # blue
            3: "#59A14F"    # green
        }
    },

    {
        "model_name": "HistGradientBoosting",
        "title": "HistGradientBoosting",
        "prefix_candidates": ["HistGradientBoosting", "HGB", "HistGB"],
        "state_colors": {
            1: "#C76D6D",   # dusty rose
            2: "#5C9E6E",   # muted green
            3: "#5E81AC"    # steel blue
        }
    },

    {
        "model_name": "Equal Soft Voting",
        "title": "Soft Voting",
        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ],
        "state_colors": {
            1: "#B85C8A",   # mauve
            2: "#4F8FBA",   # blue-cyan
            3: "#8A9A3B"    # olive
        }
    },

    {
        "model_name": "TCN",
        "title": "TCN",
        "prefix_candidates": ["TCN"],
        "state_colors": {
            1: "#C96A4A",   # terracotta
            2: "#4C9A8A",   # sea green
            3: "#8A6FBF"    # purple
        }
    },

    {
        "model_name": "CNN-LSTM",
        "title": "CNN-LSTM",
        "prefix_candidates": ["CNNLSTM", "CNN_LSTM", "CNN-LSTM"],
        "state_colors": {
            1: "#D96C6C",   # coral-red
            2: "#3F8F8C",   # teal
            3: "#6E78B7"    # indigo-blue
        }
    },
]

MACRO_COLOR = "#222222"
DIAGONAL_COLOR = "#9A9A9A"


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 22,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(PREDICTION_FILE)

if TRUE_COL not in data.columns:
    raise ValueError(
        f"'{TRUE_COL}' is missing from the prediction file.\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. HELPER FUNCTION TO FIND PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(all_columns, prefix_candidates):
    """
    Find the correct three probability columns for a model.
    Expected pattern is something like:
    Prefix_P_State_1, Prefix_P_State_2, Prefix_P_State_3
    """

    for prefix in prefix_candidates:
        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(col in all_columns for col in candidate_cols):
            return candidate_cols

    return None


# =============================================================================
# 6. COMMON TRUE LABELS
# =============================================================================

y_true = data[TRUE_COL].astype(int).to_numpy()

y_binary = label_binarize(
    y_true,
    classes=STATES
)


# =============================================================================
# 7. CREATE ONE ROC PLOT FOR EACH MODEL
# =============================================================================

for config in MODEL_CONFIGS:

    model_name = config["model_name"]
    title = config["title"]
    prefix_candidates = config["prefix_candidates"]
    STATE_COLORS = config["state_colors"]

    prob_cols = resolve_probability_columns(
        data.columns,
        prefix_candidates
    )

    if prob_cols is None:
        print("\n" + "=" * 70)
        print(f"Skipping {model_name}")
        print("Could not find probability columns for this model.")
        print(f"Tried prefixes: {prefix_candidates}")
        print("=" * 70)
        continue

    probability = data[prob_cols].astype(float).to_numpy()

    if not np.isfinite(probability).all():
        raise ValueError(
            f"Non-finite probabilities were found for {model_name}."
        )


    # =========================================================================
    # 7.1 STATE-SPECIFIC ROC CURVES
    # =========================================================================

    fpr = {}
    tpr = {}
    roc_auc = {}

    for state_index, state in enumerate(STATES):

        fpr[state], tpr[state], _ = roc_curve(
            y_binary[:, state_index],
            probability[:, state_index]
        )

        roc_auc[state] = auc(
            fpr[state],
            tpr[state]
        )


    # =========================================================================
    # 7.2 MACRO-AVERAGE ROC
    # =========================================================================

    all_fpr = np.unique(
        np.concatenate([fpr[state] for state in STATES])
    )

    mean_tpr = np.zeros_like(all_fpr)

    for state in STATES:
        mean_tpr += np.interp(all_fpr, fpr[state], tpr[state])

    mean_tpr /= len(STATES)

    macro_auc_curve = auc(all_fpr, mean_tpr)

    macro_auc_direct = roc_auc_score(
        y_binary,
        probability,
        average="macro",
        multi_class="ovr"
    )


    # =========================================================================
    # 7.3 PRINT SUMMARY
    # =========================================================================

    print("\n" + "=" * 50)
    print(f"{model_name} LOYO ROC/AUC")
    print("=" * 50)
    print(f"Probability columns: {prob_cols}")

    for state in STATES:
        print(f"State {state} AUC       : {roc_auc[state]:.3f}")

    print(f"Macro-average AUC : {macro_auc_direct:.3f}")
    print(f"Macro curve AUC   : {macro_auc_curve:.3f}")


    # =========================================================================
    # 7.4 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(figsize=(7.6, 7.0))


    # =========================================================================
    # 7.5 PLOT STEP ROC CURVES
    # =========================================================================

    for state in STATES:

        ax.step(
            fpr[state],
            tpr[state],
            where="post",
            color=STATE_COLORS[state],
            linewidth=3.4,
            zorder=3
        )


    # =========================================================================
    # 7.6 MACRO-AVERAGE CURVE
    # =========================================================================

    ax.step(
        all_fpr,
        mean_tpr,
        where="post",
        color=MACRO_COLOR,
        linewidth=3.6,
        zorder=4
    )


    # =========================================================================
    # 7.7 DIAGONAL REFERENCE LINE
    # =========================================================================

    ax.plot(
        [0, 1],
        [0, 1],
        color=DIAGONAL_COLOR,
        linewidth=1.8,
        linestyle=(0, (4, 6)),
        zorder=1
    )


    # =========================================================================
    # 7.8 TITLE
    # =========================================================================

    ax.set_title(
        title,
        fontsize=30,
        fontweight="normal",
        pad=12
    )


    # =========================================================================
    # 7.9 AXES
    # =========================================================================

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.04)

    ax.set_xlabel(
        "False positive rate",
        labelpad=10
    )

    ax.set_ylabel(
        "True positive rate",
        labelpad=10
    )

    ticks = np.arange(0, 1.01, 0.25)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


    # =========================================================================
    # 7.10 REMOVE GRID / FULL BOX
    # =========================================================================

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both",
        width=1.8,
        length=6,
        color="black"
    )


    # =========================================================================
    # 7.11 LEGEND WITH SMALL COLOURED BOXES
    # =========================================================================

    legend_handles = [
        Patch(
            facecolor=STATE_COLORS[1],
            edgecolor="none",
            label=f"State 1 (AUC = {roc_auc[1]:.2f})"
        ),
        Patch(
            facecolor=STATE_COLORS[2],
            edgecolor="none",
            label=f"State 2 (AUC = {roc_auc[2]:.2f})"
        ),
        Patch(
            facecolor=STATE_COLORS[3],
            edgecolor="none",
            label=f"State 3 (AUC = {roc_auc[3]:.2f})"
        ),
        Patch(
            facecolor=MACRO_COLOR,
            edgecolor="none",
            label=f"Macro-average (AUC = {macro_auc_direct:.2f})"
        )
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=0.55,

        handlelength=0.65,
        handleheight=0.50,
        handletextpad=0.50,

        labelspacing=0.45
    )

    legend.get_frame().set_linewidth(1.8)


    # =========================================================================
    # 7.12 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(pad=0.8)


    # =========================================================================
    # 7.13 SAVE
    # =========================================================================

    safe_title = (
        title
        .replace(" ", "_")
        .replace("-", "_")
    )

    tiff_file = OUTPUT_DIR / f"ROC_AUC_{safe_title}.tiff"
    pdf_file = OUTPUT_DIR / f"ROC_AUC_{safe_title}.pdf"

    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    fig.savefig(
        pdf_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04
    )

    plt.show()
    plt.close(fig)

    print(f"\n{title} ROC files saved:")
    print(f"TIFF (1000 dpi): {tiff_file}")
    print(f"PDF (vector)   : {pdf_file}")


# =============================================================================
# 8. FINISHED
# =============================================================================

print("\nAll available ROC/AUC figures have been created.")
print(f"Output folder: {OUTPUT_DIR}")


# In[1]:


"""Create six individual publication-quality Precision–Recall curves
from saved LOYO held-out probabilities.

Style matches the finalized ROC figures:
- step PR curves
- no full box
- no grid
- clean left/bottom axes only
- compact legend with small coloured boxes
- macro-average curve in black
- Average Precision (AP) reported in the legend
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_PR"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. MODELS
# =============================================================================

TRUE_COL = "True_State"

STATES = [
    1,
    2,
    3
]


MODEL_CONFIGS = [

    {
        "model_name": "CatBoost",
        "title": "CatBoost",

        "prefix_candidates": [
            "CatBoost"
        ],

        "state_colors": {
            1: "#D55E5E",
            2: "#4F9D92",
            3: "#7A6BB1"
        }
    },

    {
        "model_name": "XGBoost",
        "title": "XGBoost",

        "prefix_candidates": [
            "XGBoost",
            "XGB"
        ],

        "state_colors": {
            1: "#D17C2F",
            2: "#4C78A8",
            3: "#59A14F"
        }
    },

    {
        "model_name": "HistGradientBoosting",
        "title": "HistGradientBoosting",

        "prefix_candidates": [
            "HistGradientBoosting",
            "HGB",
            "HistGB"
        ],

        "state_colors": {
            1: "#C76D6D",
            2: "#5C9E6E",
            3: "#5E81AC"
        }
    },

    {
        "model_name": "Equal Soft Voting",
        "title": "Soft Voting",

        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ],

        "state_colors": {
            1: "#B85C8A",
            2: "#4F8FBA",
            3: "#8A9A3B"
        }
    },

    {
        "model_name": "TCN",
        "title": "TCN",

        "prefix_candidates": [
            "TCN"
        ],

        "state_colors": {
            1: "#C96A4A",
            2: "#4C9A8A",
            3: "#8A6FBF"
        }
    },

    {
        "model_name": "CNN-LSTM",
        "title": "CNN-LSTM",

        "prefix_candidates": [
            "CNNLSTM",
            "CNN_LSTM",
            "CNN-LSTM"
        ],

        "state_colors": {
            1: "#D96C6C",
            2: "#3F8F8C",
            3: "#6E78B7"
        }
    },
]


MACRO_COLOR = "#222222"


# =============================================================================
# 3. FIGURE STYLE
# Same as finalized ROC figure
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 22,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


if TRUE_COL not in data.columns:

    raise ValueError(
        f"'{TRUE_COL}' is missing from the prediction file.\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. FIND PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(
    all_columns,
    prefix_candidates
):

    for prefix in prefix_candidates:

        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(
            col in all_columns
            for col in candidate_cols
        ):

            return candidate_cols

    return None


# =============================================================================
# 6. TRUE LABELS
# =============================================================================

y_true = (
    data[TRUE_COL]
    .astype(int)
    .to_numpy()
)


y_binary = label_binarize(
    y_true,
    classes=STATES
)


# =============================================================================
# 7. CREATE ONE PR CURVE FOR EACH MODEL
# =============================================================================

for config in MODEL_CONFIGS:

    model_name = config["model_name"]
    title = config["title"]

    prefix_candidates = (
        config["prefix_candidates"]
    )

    STATE_COLORS = (
        config["state_colors"]
    )


    # =========================================================================
    # 7.1 FIND MODEL PROBABILITY COLUMNS
    # =========================================================================

    prob_cols = resolve_probability_columns(
        data.columns,
        prefix_candidates
    )


    if prob_cols is None:

        print("\n" + "=" * 70)

        print(
            f"Skipping {model_name}"
        )

        print(
            "Could not find probability columns."
        )

        print(
            f"Tried prefixes: {prefix_candidates}"
        )

        print("=" * 70)

        continue


    # =========================================================================
    # 7.2 PROBABILITIES
    # =========================================================================

    probability = (
        data[prob_cols]
        .astype(float)
        .to_numpy()
    )


    if not np.isfinite(
        probability
    ).all():

        raise ValueError(
            f"Non-finite probabilities "
            f"were found for {model_name}."
        )


    # =========================================================================
    # 7.3 STATE-SPECIFIC PR CURVES
    # =========================================================================

    precision = {}
    recall = {}
    average_precision = {}


    for state_index, state in enumerate(STATES):

        precision[state], recall[state], _ = (
            precision_recall_curve(
                y_binary[:, state_index],
                probability[:, state_index]
            )
        )


        average_precision[state] = (
            average_precision_score(
                y_binary[:, state_index],
                probability[:, state_index]
            )
        )


    # =========================================================================
    # 7.4 MACRO-AVERAGE AP
    # =========================================================================

    macro_ap = average_precision_score(
        y_binary,
        probability,
        average="macro"
    )


    # =========================================================================
    # 7.5 MACRO-AVERAGE PR CURVE
    # =========================================================================

    # Common recall grid from 0 to 1
    recall_grid = np.linspace(
        0,
        1,
        500
    )


    interpolated_precision = []


    for state in STATES:

        # sklearn returns recall from 1 -> 0.
        # Reverse it for interpolation on ascending recall.
        recall_ascending = (
            recall[state][::-1]
        )

        precision_ascending = (
            precision[state][::-1]
        )


        interp_precision = np.interp(
            recall_grid,
            recall_ascending,
            precision_ascending
        )


        interpolated_precision.append(
            interp_precision
        )


    macro_precision = np.mean(
        interpolated_precision,
        axis=0
    )


    # =========================================================================
    # 7.6 PRINT RESULTS
    # =========================================================================

    print("\n" + "=" * 50)

    print(
        f"{model_name} LOYO Precision–Recall"
    )

    print("=" * 50)

    print(
        f"Probability columns: {prob_cols}"
    )


    for state in STATES:

        print(
            f"State {state} AP        : "
            f"{average_precision[state]:.3f}"
        )


    print(
        f"Macro-average AP  : "
        f"{macro_ap:.3f}"
    )


    # =========================================================================
    # 7.7 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.6, 7.0)
    )


    # =========================================================================
    # 7.8 STATE-SPECIFIC STEP PR CURVES
    # =========================================================================

    for state in STATES:

        ax.step(
            recall[state],
            precision[state],

            where="post",

            color=STATE_COLORS[state],

            linewidth=3.4,

            zorder=3
        )


    # =========================================================================
    # 7.9 MACRO-AVERAGE CURVE
    # =========================================================================

    ax.plot(
        recall_grid,
        macro_precision,

        color=MACRO_COLOR,

        linewidth=3.6,

        zorder=4
    )


    # =========================================================================
    # 7.10 TITLE
    # =========================================================================

    ax.set_title(
        title,

        fontsize=30,

        fontweight="normal",

        pad=12
    )


    # =========================================================================
    # 7.11 AXES
    # =========================================================================

    ax.set_xlim(
        -0.01,
        1.01
    )

    ax.set_ylim(
        -0.01,
        1.04
    )


    ax.set_xlabel(
        "Recall",
        labelpad=10
    )


    ax.set_ylabel(
        "Precision",
        labelpad=10
    )


    ticks = np.arange(
        0,
        1.01,
        0.25
    )


    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )


    # =========================================================================
    # 7.12 REMOVE GRID / FULL BOX
    # =========================================================================

    ax.grid(False)


    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        2.0
    )

    ax.spines["bottom"].set_linewidth(
        2.0
    )


    ax.spines["left"].set_color(
        "black"
    )

    ax.spines["bottom"].set_color(
        "black"
    )


    ax.tick_params(
        axis="both",

        width=1.8,

        length=6,

        color="black"
    )


    # =========================================================================
    # 7.13 LEGEND WITH SMALL COLOURED BOXES
    # =========================================================================

    legend_handles = [

        Patch(
            facecolor=STATE_COLORS[1],
            edgecolor="none",

            label=(
                f"State 1 "
                f"(AP = {average_precision[1]:.2f})"
            )
        ),

        Patch(
            facecolor=STATE_COLORS[2],
            edgecolor="none",

            label=(
                f"State 2 "
                f"(AP = {average_precision[2]:.2f})"
            )
        ),

        Patch(
            facecolor=STATE_COLORS[3],
            edgecolor="none",

            label=(
                f"State 3 "
                f"(AP = {average_precision[3]:.2f})"
            )
        ),

        Patch(
            facecolor=MACRO_COLOR,
            edgecolor="none",

            label=(
                "Macro-average "
                f"(AP = {macro_ap:.2f})"
            )
        )
    ]


    legend = ax.legend(

        handles=legend_handles,

        loc="lower left",

        frameon=True,

        fancybox=False,

        framealpha=1.0,

        facecolor="white",

        edgecolor="black",

        borderpad=0.55,

        # Small coloured boxes
        handlelength=0.65,
        handleheight=0.50,
        handletextpad=0.50,

        labelspacing=0.45
    )


    legend.get_frame().set_linewidth(
        1.8
    )


    # =========================================================================
    # 7.14 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 7.15 SAVE
    # =========================================================================

    safe_title = (
        title
        .replace(" ", "_")
        .replace("-", "_")
    )


    tiff_file = (
        OUTPUT_DIR
        / f"PR_Curve_{safe_title}.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"PR_Curve_{safe_title}.pdf"
    )


    fig.savefig(
        tiff_file,

        dpi=1000,

        format="tiff",

        bbox_inches="tight",

        pad_inches=0.04,

        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(
        pdf_file,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0.04
    )


    plt.show()

    plt.close(fig)


    print(
        f"\n{title} PR files saved:"
    )

    print(
        f"TIFF (1000 dpi): {tiff_file}"
    )

    print(
        f"PDF (vector)   : {pdf_file}"
    )


# =============================================================================
# 8. FINISHED
# =============================================================================

print(
    "\nAll available Precision–Recall figures have been created."
)

print(
    f"Output folder: {OUTPUT_DIR}"
)


# In[7]:


"""Create six individual publication-quality Decision Curve Analysis (DCA)
figures using saved LOYO held-out probabilities.

Multiclass DCA is implemented one-vs-rest for each ecological state.

Style follows the finalized CatBoost DCA figure:
- threshold range = 0.05–0.80
- no full box
- no grid
- clean left/bottom axes only
- thick coloured state-specific model curves
- thin state-specific treat-all reference curves
- treat-none horizontal reference
- state legend in upper right
- treat-all / treat-none legend in lower left
- same font sizes and figure dimensions for every model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_DCA"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. BASIC SETTINGS
# =============================================================================

TRUE_COL = "True_State"

STATES = [
    1,
    2,
    3
]


# =============================================================================
# 3. MODEL SETTINGS
# =============================================================================

MODEL_CONFIGS = [

    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------
    {
        "model_name": "CatBoost",

        "title": "CatBoost",

        "prefix_candidates": [
            "CatBoost"
        ],

        "state_colors": {
            1: "#D55E5E",
            2: "#4F9D92",
            3: "#7A6BB1"
        }
    },


    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------
    {
        "model_name": "XGBoost",

        "title": "XGBoost",

        "prefix_candidates": [
            "XGBoost",
            "XGB"
        ],

        "state_colors": {
            1: "#D17C2F",
            2: "#4C78A8",
            3: "#59A14F"
        }
    },


    # -------------------------------------------------------------------------
    # HistGradientBoosting
    # -------------------------------------------------------------------------
    {
        "model_name": "HistGradientBoosting",

        "title": "HistGradientBoosting",

        "prefix_candidates": [
            "HistGradientBoosting",
            "HGB",
            "HistGB"
        ],

        "state_colors": {
            1: "#C76D6D",
            2: "#5C9E6E",
            3: "#5E81AC"
        }
    },


    # -------------------------------------------------------------------------
    # Soft Voting
    # -------------------------------------------------------------------------
    {
        "model_name": "Equal Soft Voting",

        "title": "Soft Voting",

        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ],

        "state_colors": {
            1: "#B85C8A",
            2: "#4F8FBA",
            3: "#8A9A3B"
        }
    },


    # -------------------------------------------------------------------------
    # TCN
    # -------------------------------------------------------------------------
    {
        "model_name": "TCN",

        "title": "TCN",

        "prefix_candidates": [
            "TCN"
        ],

        "state_colors": {
            1: "#C96A4A",
            2: "#4C9A8A",
            3: "#8A6FBF"
        }
    },


    # -------------------------------------------------------------------------
    # CNN-LSTM
    # -------------------------------------------------------------------------
    {
        "model_name": "CNN-LSTM",

        "title": "CNN-LSTM",

        "prefix_candidates": [
            "CNNLSTM",
            "CNN_LSTM",
            "CNN-LSTM"
        ],

        "state_colors": {
            1: "#D96C6C",
            2: "#3F8F8C",
            3: "#6E78B7"
        }
    },
]


# =============================================================================
# 4. THRESHOLD RANGE
# =============================================================================

THRESHOLDS = np.linspace(
    0.05,
    0.80,
    151
)


# =============================================================================
# 5. FIGURE STYLE
# EXACTLY MATCH FINAL CATBOOST DCA STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 21,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 6. OTHER COLOURS
# =============================================================================

TREAT_NONE_COLOR = "#222222"


# =============================================================================
# 7. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


if TRUE_COL not in data.columns:

    raise ValueError(
        f"'{TRUE_COL}' is missing from the prediction file.\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 8. HELPER: FIND PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(
    all_columns,
    prefix_candidates
):

    """
    Finds probability columns such as:

    CatBoost_P_State_1
    CatBoost_P_State_2
    CatBoost_P_State_3

    Multiple candidate prefixes are allowed because saved naming may differ
    slightly among models.
    """

    for prefix in prefix_candidates:

        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(
            col in all_columns
            for col in candidate_cols
        ):

            return candidate_cols

    return None


# =============================================================================
# 9. DECISION-CURVE FUNCTION
# =============================================================================

def calculate_net_benefit(
    y_binary,
    predicted_probability,
    thresholds
):

    """
    Standard binary DCA net benefit:

        NB = TP/N - FP/N × pt/(1-pt)

    Multiclass analysis is handled one-vs-rest.
    """

    net_benefit = []

    n = len(
        y_binary
    )


    for threshold in thresholds:

        predicted_positive = (
            predicted_probability
            >= threshold
        )


        tp = np.sum(

            predicted_positive

            &

            (y_binary == 1)
        )


        fp = np.sum(

            predicted_positive

            &

            (y_binary == 0)
        )


        odds = (
            threshold
            / (1.0 - threshold)
        )


        nb = (
            tp / n
            -
            (fp / n) * odds
        )


        net_benefit.append(
            nb
        )


    return np.asarray(
        net_benefit
    )


# =============================================================================
# 10. TRUE STATES
# =============================================================================

y_true = (
    data[TRUE_COL]
    .astype(int)
    .to_numpy()
)


# =============================================================================
# 11. CREATE DCA FOR EACH MODEL
# =============================================================================

for config in MODEL_CONFIGS:


    # =========================================================================
    # 11.1 MODEL INFORMATION
    # =========================================================================

    model_name = (
        config["model_name"]
    )

    title = (
        config["title"]
    )

    prefix_candidates = (
        config["prefix_candidates"]
    )

    STATE_COLORS = (
        config["state_colors"]
    )


    # =========================================================================
    # 11.2 FIND PROBABILITY COLUMNS
    # =========================================================================

    PROB_COLS = resolve_probability_columns(
        data.columns,
        prefix_candidates
    )


    if PROB_COLS is None:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Could not find probability columns for: {model_name}"
        )

        print(
            f"Tried prefixes: {prefix_candidates}"
        )

        print(
            "\nAvailable probability-like columns:"
        )

        print(
            [
                column
                for column in data.columns
                if "_P_State_" in column
            ]
        )

        print(
            "=" * 70
        )

        continue


    # =========================================================================
    # 11.3 LOAD MODEL PROBABILITIES
    # =========================================================================

    probability = (
        data[PROB_COLS]
        .astype(float)
        .to_numpy()
    )


    if not np.isfinite(
        probability
    ).all():

        raise ValueError(
            f"Non-finite probabilities were found for {model_name}."
        )


    # =========================================================================
    # 11.4 CALCULATE DCA
    # =========================================================================

    net_benefit = {}

    treat_all = {}

    prevalence = {}


    for state_index, state in enumerate(STATES):


        # ---------------------------------------------------------------------
        # Binary target: state vs all other states
        # ---------------------------------------------------------------------

        y_binary = (
            y_true == state
        ).astype(int)


        # ---------------------------------------------------------------------
        # Prevalence
        # ---------------------------------------------------------------------

        prevalence[state] = (
            y_binary.mean()
        )


        # ---------------------------------------------------------------------
        # Model net benefit
        # ---------------------------------------------------------------------

        net_benefit[state] = (
            calculate_net_benefit(

                y_binary,

                probability[:, state_index],

                THRESHOLDS
            )
        )


        # ---------------------------------------------------------------------
        # Treat-all reference
        # ---------------------------------------------------------------------

        odds = (
            THRESHOLDS
            / (1.0 - THRESHOLDS)
        )


        treat_all[state] = (

            prevalence[state]

            -

            (1.0 - prevalence[state])

            * odds
        )


    # =========================================================================
    # 11.5 PRINT RESULTS
    # =========================================================================

    print(
        "\n" + "=" * 55
    )

    print(
        f"{model_name} LOYO Decision Curve Analysis"
    )

    print(
        "=" * 55
    )

    print(
        f"Probability columns: {PROB_COLS}"
    )


    for state in STATES:

        print(
            f"State {state} prevalence : "
            f"{prevalence[state]:.3f}"
        )


    # =========================================================================
    # 11.6 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.6, 7.0)
    )


    # =========================================================================
    # 11.7 MODEL NET-BENEFIT CURVES
    # =========================================================================

    for state in STATES:

        ax.plot(

            THRESHOLDS,

            net_benefit[state],

            color=STATE_COLORS[state],

            linewidth=3.4,

            zorder=4
        )


    # =========================================================================
    # 11.8 STATE-SPECIFIC TREAT-ALL CURVES
    # =========================================================================

    for state in STATES:

        ax.plot(

            THRESHOLDS,

            treat_all[state],

            color=STATE_COLORS[state],

            linewidth=1.5,

            linestyle=(0, (4, 5)),

            alpha=0.55,

            zorder=2
        )


    # =========================================================================
    # 11.9 TREAT-NONE REFERENCE
    # =========================================================================

    ax.axhline(

        y=0,

        color=TREAT_NONE_COLOR,

        linewidth=1.8,

        linestyle=(0, (6, 5)),

        zorder=1
    )


    # =========================================================================
    # 11.10 TITLE
    # =========================================================================

    ax.set_title(

        title,

        fontsize=30,

        fontweight="normal",

        pad=12
    )


    # =========================================================================
    # 11.11 X-AXIS
    # =========================================================================

    ax.set_xlim(
        0.05,
        0.80
    )


    ax.set_xticks(
        [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80
        ]
    )


    # =========================================================================
    # 11.12 Y-AXIS LIMIT
    # Same rule as final CatBoost script
    # =========================================================================

    all_values = np.concatenate(
        [
            net_benefit[1],
            net_benefit[2],
            net_benefit[3],

            treat_all[1],
            treat_all[2],
            treat_all[3],

            np.array([0.0])
        ]
    )


    visible_values = all_values[
        np.isfinite(all_values)
    ]


    y_lower = max(

        -0.30,

        np.nanmin(
            visible_values
        ) - 0.03
    )


    y_upper = min(

        0.45,

        np.nanmax(
            visible_values
        ) + 0.03
    )


    ax.set_ylim(
        y_lower,
        y_upper
    )


    # =========================================================================
    # 11.13 AXIS LABELS
    # =========================================================================

    ax.set_xlabel(

        "Threshold probability",

        labelpad=10
    )


    ax.set_ylabel(

        "Net benefit",

        labelpad=10
    )


    # =========================================================================
    # 11.14 NO GRID / NO FULL BOX
    # =========================================================================

    ax.grid(
        False
    )


    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        2.0
    )

    ax.spines["bottom"].set_linewidth(
        2.0
    )


    ax.spines["left"].set_color(
        "black"
    )

    ax.spines["bottom"].set_color(
        "black"
    )


    ax.tick_params(

        axis="both",

        width=1.8,

        length=6,

        color="black"
    )


    # =========================================================================
    # 11.15 STATE LEGEND — UPPER RIGHT
    # =========================================================================

    state_handles = [

        Patch(
            facecolor=STATE_COLORS[1],
            edgecolor="none",
            label="State 1"
        ),

        Patch(
            facecolor=STATE_COLORS[2],
            edgecolor="none",
            label="State 2"
        ),

        Patch(
            facecolor=STATE_COLORS[3],
            edgecolor="none",
            label="State 3"
        )
    ]


    state_legend = ax.legend(

        handles=state_handles,

        loc="upper right",

        frameon=True,

        fancybox=False,

        framealpha=1.0,

        facecolor="white",

        edgecolor="black",

        borderpad=0.45,

        handlelength=0.65,

        handleheight=0.50,

        handletextpad=0.50,

        labelspacing=0.35
    )


    state_legend.get_frame().set_linewidth(
        1.6
    )


    # =========================================================================
    # 11.16 STRATEGY LEGEND — LOWER LEFT
    # =========================================================================

    strategy_handles = [

        Line2D(
            [0],
            [0],

            color="#777777",

            linewidth=1.6,

            linestyle=(0, (4, 5)),

            label="Treat all"
        ),

        Line2D(
            [0],
            [0],

            color=TREAT_NONE_COLOR,

            linewidth=1.8,

            linestyle=(0, (6, 5)),

            label="Treat none"
        )
    ]


    strategy_legend = ax.legend(

        handles=strategy_handles,

        loc="lower left",

        frameon=True,

        fancybox=False,

        framealpha=1.0,

        facecolor="white",

        edgecolor="black",

        borderpad=0.45,

        handlelength=1.3,

        handletextpad=0.50,

        labelspacing=0.35
    )


    strategy_legend.get_frame().set_linewidth(
        1.6
    )


    # Keep state legend visible
    ax.add_artist(
        state_legend
    )


    # =========================================================================
    # 11.17 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 11.18 SAVE
    # =========================================================================

    safe_title = (
        title
        .replace(" ", "_")
        .replace("-", "_")
    )


    tiff_file = (
        OUTPUT_DIR
        / f"DCA_{safe_title}.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"DCA_{safe_title}.pdf"
    )


    fig.savefig(

        tiff_file,

        dpi=1000,

        format="tiff",

        bbox_inches="tight",

        pad_inches=0.04,

        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(

        pdf_file,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0.04
    )


    plt.show()

    plt.close(fig)


    print(
        f"\n{title} DCA files saved:"
    )

    print(
        f"TIFF (1000 dpi): {tiff_file}"
    )

    print(
        f"PDF (vector)   : {pdf_file}"
    )


# =============================================================================
# 12. FINISHED
# =============================================================================

print(
    "\nAll available six-model DCA figures have been created."
)

print(
    f"Output folder: {OUTPUT_DIR}"
)


# In[16]:


"""Create six individual publication-quality multiclass calibration curves
using saved LOYO held-out probabilities.

Multiclass calibration is evaluated one-vs-rest for each ecological state.

Final settings:
- N_BINS = 5
- BIN_STRATEGY = "quantile"
- no 'Ideal calibration' text
- same visual style as ROC / PR / DCA figures
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_Calibration_Curves"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. BASIC SETTINGS
# =============================================================================

TRUE_COL = "True_State"

STATES = [1, 2, 3]

N_BINS = 5
BIN_STRATEGY = "quantile"


# =============================================================================
# 3. MODEL SETTINGS
# =============================================================================

MODEL_CONFIGS = [

    {
        "model_name": "CatBoost",
        "title": "CatBoost",
        "prefix_candidates": ["CatBoost"],
        "state_colors": {
            1: "#D55E5E",
            2: "#4F9D92",
            3: "#7A6BB1"
        }
    },

    {
        "model_name": "XGBoost",
        "title": "XGBoost",
        "prefix_candidates": ["XGBoost", "XGB"],
        "state_colors": {
            1: "#D17C2F",
            2: "#4C78A8",
            3: "#59A14F"
        }
    },

    {
        "model_name": "HistGradientBoosting",
        "title": "HistGradientBoosting",
        "prefix_candidates": ["HistGradientBoosting", "HGB", "HistGB"],
        "state_colors": {
            1: "#C76D6D",
            2: "#5C9E6E",
            3: "#5E81AC"
        }
    },

    {
        "model_name": "Equal Soft Voting",
        "title": "Soft Voting",
        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ],
        "state_colors": {
            1: "#B85C8A",
            2: "#4F8FBA",
            3: "#8A9A3B"
        }
    },

    {
        "model_name": "TCN",
        "title": "TCN",
        "prefix_candidates": ["TCN"],
        "state_colors": {
            1: "#C96A4A",
            2: "#4C9A8A",
            3: "#8A6FBF"
        }
    },

    {
        "model_name": "CNN-LSTM",
        "title": "CNN-LSTM",
        "prefix_candidates": ["CNNLSTM", "CNN_LSTM", "CNN-LSTM"],
        "state_colors": {
            1: "#D96C6C",
            2: "#3F8F8C",
            3: "#6E78B7"
        }
    },
]


# =============================================================================
# 4. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 21,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 5. OTHER COLOURS
# =============================================================================

IDEAL_COLOR = "#8F969B"


# =============================================================================
# 6. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)

if TRUE_COL not in data.columns:
    raise ValueError(
        f"'{TRUE_COL}' is missing from the prediction file.\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 7. HELPER: FIND PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(all_columns, prefix_candidates):
    """
    Finds probability columns such as:
    Model_P_State_1, Model_P_State_2, Model_P_State_3
    """

    for prefix in prefix_candidates:

        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(col in all_columns for col in candidate_cols):
            return candidate_cols

    return None


# =============================================================================
# 8. TRUE STATES
# =============================================================================

y_true = (
    data[TRUE_COL]
    .astype(int)
    .to_numpy()
)


# =============================================================================
# 9. CREATE CALIBRATION CURVE FOR EACH MODEL
# =============================================================================

for config in MODEL_CONFIGS:

    model_name = config["model_name"]
    title = config["title"]
    prefix_candidates = config["prefix_candidates"]
    STATE_COLORS = config["state_colors"]

    PROB_COLS = resolve_probability_columns(
        data.columns,
        prefix_candidates
    )

    if PROB_COLS is None:

        print("\n" + "=" * 70)
        print(f"Could not find probability columns for: {model_name}")
        print(f"Tried prefixes: {prefix_candidates}")
        print("\nAvailable probability-like columns:")
        print([column for column in data.columns if "_P_State_" in column])
        print("=" * 70)
        continue


    # =========================================================================
    # 9.1 LOAD MODEL PROBABILITIES
    # =========================================================================

    probability = (
        data[PROB_COLS]
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(probability).all():
        raise ValueError(
            f"Non-finite probabilities were found for {model_name}."
        )

    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(
            f"Probabilities for {model_name} must lie between 0 and 1."
        )


    # =========================================================================
    # 9.2 CALCULATE STATE-SPECIFIC CALIBRATION CURVES
    # =========================================================================

    fraction_positive = {}
    mean_predicted = {}
    brier_score = {}

    for state_index, state in enumerate(STATES):

        y_binary = (
            y_true == state
        ).astype(int)

        fraction_positive[state], mean_predicted[state] = calibration_curve(
            y_binary,
            probability[:, state_index],
            n_bins=N_BINS,
            strategy=BIN_STRATEGY
        )

        brier_score[state] = brier_score_loss(
            y_binary,
            probability[:, state_index]
        )


    # =========================================================================
    # 9.3 PRINT RESULTS
    # =========================================================================

    print("\n" + "=" * 52)
    print(f"{model_name} LOYO Calibration")
    print("=" * 52)

    print(f"Probability columns: {PROB_COLS}")

    for state in STATES:
        print(
            f"State {state} Brier score : "
            f"{brier_score[state]:.4f}"
        )


    # =========================================================================
    # 9.4 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.6, 7.0)
    )


    # =========================================================================
    # 9.5 IDEAL CALIBRATION LINE
    # =========================================================================

    ax.plot(
        [0, 1],
        [0, 1],
        color=IDEAL_COLOR,
        linewidth=1.8,
        linestyle=(0, (5, 6)),
        zorder=1
    )


    # =========================================================================
    # 9.6 STATE-SPECIFIC CALIBRATION CURVES
    # =========================================================================

    for state in STATES:

        ax.plot(
            mean_predicted[state],
            fraction_positive[state],

            color=STATE_COLORS[state],

            linewidth=3.2,

            marker="o",
            markersize=9,

            markerfacecolor=STATE_COLORS[state],
            markeredgecolor="white",
            markeredgewidth=1.0,

            zorder=3
        )


    # =========================================================================
    # 9.7 TITLE
    # =========================================================================

    ax.set_title(
        title,
        fontsize=30,
        fontweight="normal",
        pad=12
    )


    # =========================================================================
    # 9.8 AXES
    # =========================================================================

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    ax.set_xlabel(
        "Mean predicted probability",
        labelpad=10
    )

    ax.set_ylabel(
        "Observed frequency",
        labelpad=10
    )

    ticks = np.arange(0, 1.01, 0.25)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


    # =========================================================================
    # 9.9 REMOVE GRID / FULL BOX
    # =========================================================================

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both",
        width=1.8,
        length=6,
        color="black"
    )


    # =========================================================================
    # 9.10 LEGEND
    # =========================================================================

    legend_handles = [

        Patch(
            facecolor=STATE_COLORS[1],
            edgecolor="none",
            label=f"State 1 (Brier = {brier_score[1]:.3f})"
        ),

        Patch(
            facecolor=STATE_COLORS[2],
            edgecolor="none",
            label=f"State 2 (Brier = {brier_score[2]:.3f})"
        ),

        Patch(
            facecolor=STATE_COLORS[3],
            edgecolor="none",
            label=f"State 3 (Brier = {brier_score[3]:.3f})"
        )
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",

        frameon=True,
        fancybox=False,
        framealpha=1.0,

        facecolor="white",
        edgecolor="black",

        borderpad=0.50,

        handlelength=0.65,
        handleheight=0.50,
        handletextpad=0.50,

        labelspacing=0.40
    )

    legend.get_frame().set_linewidth(1.6)


    # =========================================================================
    # 9.11 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 9.12 SAVE
    # =========================================================================

    safe_title = (
        title
        .replace(" ", "_")
        .replace("-", "_")
    )

    tiff_file = OUTPUT_DIR / f"Calibration_Curve_{safe_title}.tiff"
    pdf_file = OUTPUT_DIR / f"Calibration_Curve_{safe_title}.pdf"

    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    fig.savefig(
        pdf_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04
    )

    plt.show()
    plt.close(fig)

    print(f"\n{title} calibration files saved:")
    print(f"TIFF (1000 dpi): {tiff_file}")
    print(f"PDF (vector)   : {pdf_file}")


# =============================================================================
# 10. FINISHED
# =============================================================================

print("\nAll six-model calibration-curve figures have been created.")
print(f"Output folder: {OUTPUT_DIR}")


# In[30]:


"""Create six individual publication-quality threshold trade-off plots.

Shows one-vs-rest sensitivity and specificity across probability thresholds
using saved LOYO held-out probabilities.

For each ecological state, the balanced threshold is defined as the threshold
at which |sensitivity - specificity| is minimized.

Final presentation:
- y-axis = Rate
- sensitivity = solid
- specificity = dashed
- balance-point circles retained
- balanced thresholds reported in compact state legend
- no threshold text written over curves
- state legend in lower-left / centre
- sensitivity-specificity legend in lower-right / centre
- same dimensions and font sizes for every model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
    / "Individual_Tradeoff"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. BASIC SETTINGS
# =============================================================================

TRUE_COL = "True_State"

STATES = [
    1,
    2,
    3
]


# =============================================================================
# 3. MODEL SETTINGS
# =============================================================================

MODEL_CONFIGS = [

    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------
    {
        "model_name": "CatBoost",
        "title": "CatBoost",

        "prefix_candidates": [
            "CatBoost"
        ],

        "state_colors": {
            1: "#D55E5E",
            2: "#4F9D92",
            3: "#7A6BB1"
        }
    },


    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------
    {
        "model_name": "XGBoost",
        "title": "XGBoost",

        "prefix_candidates": [
            "XGBoost",
            "XGB"
        ],

        "state_colors": {
            1: "#D17C2F",
            2: "#4C78A8",
            3: "#59A14F"
        }
    },


    # -------------------------------------------------------------------------
    # HistGradientBoosting
    # -------------------------------------------------------------------------
    {
        "model_name": "HistGradientBoosting",
        "title": "HistGradientBoosting",

        "prefix_candidates": [
            "HistGradientBoosting",
            "HGB",
            "HistGB"
        ],

        "state_colors": {
            1: "#C76D6D",
            2: "#5C9E6E",
            3: "#5E81AC"
        }
    },


    # -------------------------------------------------------------------------
    # Soft Voting
    # -------------------------------------------------------------------------
    {
        "model_name": "Equal Soft Voting",
        "title": "Soft Voting",

        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ],

        "state_colors": {
            1: "#B85C8A",
            2: "#4F8FBA",
            3: "#8A9A3B"
        }
    },


    # -------------------------------------------------------------------------
    # TCN
    # -------------------------------------------------------------------------
    {
        "model_name": "TCN",
        "title": "TCN",

        "prefix_candidates": [
            "TCN"
        ],

        "state_colors": {
            1: "#C96A4A",
            2: "#4C9A8A",
            3: "#8A6FBF"
        }
    },


    # -------------------------------------------------------------------------
    # CNN-LSTM
    # -------------------------------------------------------------------------
    {
        "model_name": "CNN-LSTM",
        "title": "CNN-LSTM",

        "prefix_candidates": [
            "CNNLSTM",
            "CNN_LSTM",
            "CNN-LSTM"
        ],

        "state_colors": {
            1: "#D96C6C",
            2: "#3F8F8C",
            3: "#6E78B7"
        }
    },
]


# =============================================================================
# 4. THRESHOLD SETTINGS
# =============================================================================

THRESHOLDS = np.linspace(
    0.01,
    0.99,
    199
)


# =============================================================================
# 5. FIGURE STYLE
# Exactly follows the finalized CatBoost trade-off figure
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "axes.labelsize": 32,
    "axes.titlesize": 32,

    "xtick.labelsize": 26,
    "ytick.labelsize": 26,

    "legend.fontsize": 20,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 6. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


if TRUE_COL not in data.columns:

    raise ValueError(
        f"'{TRUE_COL}' is missing from the prediction file.\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 7. HELPER: FIND MODEL PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(
    all_columns,
    prefix_candidates
):

    """
    Find the three saved probability columns for each model.

    Expected general structure:
        Model_P_State_1
        Model_P_State_2
        Model_P_State_3
    """

    for prefix in prefix_candidates:

        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(
            column in all_columns
            for column in candidate_cols
        ):

            return candidate_cols

    return None


# =============================================================================
# 8. TRUE STATES
# =============================================================================

y_true = (
    data[TRUE_COL]
    .astype(int)
    .to_numpy()
)


# =============================================================================
# 9. CREATE ONE TRADE-OFF FIGURE FOR EACH MODEL
# =============================================================================

for config in MODEL_CONFIGS:


    # =========================================================================
    # 9.1 MODEL INFORMATION
    # =========================================================================

    model_name = (
        config["model_name"]
    )

    title = (
        config["title"]
    )

    prefix_candidates = (
        config["prefix_candidates"]
    )

    STATE_COLORS = (
        config["state_colors"]
    )


    # =========================================================================
    # 9.2 FIND PROBABILITY COLUMNS
    # =========================================================================

    PROB_COLS = resolve_probability_columns(
        data.columns,
        prefix_candidates
    )


    if PROB_COLS is None:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Could not find probability columns for: {model_name}"
        )

        print(
            f"Tried prefixes: {prefix_candidates}"
        )

        print(
            "\nAvailable probability-like columns:"
        )

        print(
            [
                column
                for column in data.columns
                if "_P_State_" in column
            ]
        )

        print(
            "=" * 70
        )

        continue


    # =========================================================================
    # 9.3 LOAD MODEL PROBABILITIES
    # =========================================================================

    probability = (
        data[PROB_COLS]
        .astype(float)
        .to_numpy()
    )


    if not np.isfinite(
        probability
    ).all():

        raise ValueError(
            f"Non-finite probabilities were found for {model_name}."
        )


    # =========================================================================
    # 9.4 CALCULATE SENSITIVITY AND SPECIFICITY
    # =========================================================================

    sensitivity = {}
    specificity = {}


    for state_index, state in enumerate(STATES):


        # ---------------------------------------------------------------------
        # One-vs-rest target
        # ---------------------------------------------------------------------

        y_binary = (
            y_true == state
        ).astype(int)


        sensitivity[state] = []
        specificity[state] = []


        # ---------------------------------------------------------------------
        # Evaluate all thresholds
        # ---------------------------------------------------------------------

        for threshold in THRESHOLDS:


            predicted_positive = (
                probability[:, state_index]
                >= threshold
            ).astype(int)


            tp = np.sum(
                (predicted_positive == 1)
                &
                (y_binary == 1)
            )


            fn = np.sum(
                (predicted_positive == 0)
                &
                (y_binary == 1)
            )


            tn = np.sum(
                (predicted_positive == 0)
                &
                (y_binary == 0)
            )


            fp = np.sum(
                (predicted_positive == 1)
                &
                (y_binary == 0)
            )


            sens = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else np.nan
            )


            spec = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else np.nan
            )


            sensitivity[state].append(
                sens
            )


            specificity[state].append(
                spec
            )


        sensitivity[state] = np.asarray(
            sensitivity[state]
        )


        specificity[state] = np.asarray(
            specificity[state]
        )


    # =========================================================================
    # 9.5 FIND BALANCED THRESHOLD FOR EACH STATE
    # =========================================================================

    balanced_threshold = {}

    balanced_sensitivity = {}

    balanced_specificity = {}

    balanced_rate = {}


    for state in STATES:


        difference = np.abs(
            sensitivity[state]
            -
            specificity[state]
        )


        valid_indices = np.where(
            np.isfinite(difference)
        )[0]


        best_index = valid_indices[
            np.argmin(
                difference[valid_indices]
            )
        ]


        balanced_threshold[state] = (
            THRESHOLDS[best_index]
        )


        balanced_sensitivity[state] = (
            sensitivity[state][best_index]
        )


        balanced_specificity[state] = (
            specificity[state][best_index]
        )


        # Midpoint used only for plotting balance marker
        balanced_rate[state] = (

            balanced_sensitivity[state]
            +
            balanced_specificity[state]

        ) / 2.0


    # =========================================================================
    # 9.6 PRINT RESULTS
    # =========================================================================

    print(
        "\n" + "=" * 58
    )

    print(
        f"{model_name} threshold trade-off"
    )

    print(
        "=" * 58
    )

    print(
        f"Probability columns: {PROB_COLS}"
    )


    for state in STATES:

        print(
            f"State {state}: "
            f"threshold = {balanced_threshold[state]:.2f}, "
            f"sensitivity = {balanced_sensitivity[state]:.3f}, "
            f"specificity = {balanced_specificity[state]:.3f}"
        )


    # =========================================================================
    # 9.7 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.6, 7.0)
    )


    # =========================================================================
    # 9.8 PLOT SENSITIVITY / SPECIFICITY
    # =========================================================================

    for state in STATES:


        # ---------------------------------------------------------------------
        # Sensitivity — solid
        # ---------------------------------------------------------------------

        ax.plot(

            THRESHOLDS,

            sensitivity[state],

            color=STATE_COLORS[state],

            linewidth=3.3,

            linestyle="-",

            zorder=3
        )


        # ---------------------------------------------------------------------
        # Specificity — dashed
        # ---------------------------------------------------------------------

        ax.plot(

            THRESHOLDS,

            specificity[state],

            color=STATE_COLORS[state],

            linewidth=2.7,

            linestyle=(0, (6, 4)),

            alpha=0.92,

            zorder=2
        )


    # =========================================================================
    # 9.9 BALANCE-POINT MARKERS
    # =========================================================================

    for state in STATES:

        ax.scatter(

            balanced_threshold[state],

            balanced_rate[state],

            s=120,

            color=STATE_COLORS[state],

            edgecolor="white",

            linewidth=1.5,

            zorder=6
        )


    # =========================================================================
    # 9.10 TITLE
    # =========================================================================

    ax.set_title(

        title,

        fontsize=30,

        fontweight="normal",

        pad=12
    )


    # =========================================================================
    # 9.11 AXES
    # =========================================================================

    ax.set_xlim(
        0.0,
        1.0
    )


    ax.set_ylim(
        -0.01,
        1.01
    )


    ax.set_xlabel(

        "Probability threshold",

        labelpad=10
    )


    ax.set_ylabel(

        "Rate",

        labelpad=10
    )


    ticks = np.arange(
        0,
        1.01,
        0.20
    )


    ax.set_xticks(
        ticks
    )


    ax.set_yticks(
        ticks
    )


    # =========================================================================
    # 9.12 REMOVE GRID / FULL BOX
    # =========================================================================

    ax.grid(
        False
    )


    ax.spines["top"].set_visible(
        False
    )


    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        2.0
    )


    ax.spines["bottom"].set_linewidth(
        2.0
    )


    ax.spines["left"].set_color(
        "black"
    )


    ax.spines["bottom"].set_color(
        "black"
    )


    ax.tick_params(

        axis="both",

        width=1.8,

        length=6,

        color="black"
    )


    # =========================================================================
    # 9.13 COMPACT STATE / THRESHOLD LEGEND
    # =========================================================================

    state_handles = [

        Patch(
            facecolor=STATE_COLORS[1],
            edgecolor="none",
            label=(
                f"State 1   "
                f"{balanced_threshold[1]:.2f}"
            )
        ),

        Patch(
            facecolor=STATE_COLORS[2],
            edgecolor="none",
            label=(
                f"State 2   "
                f"{balanced_threshold[2]:.2f}"
            )
        ),

        Patch(
            facecolor=STATE_COLORS[3],
            edgecolor="none",
            label=(
                f"State 3   "
                f"{balanced_threshold[3]:.2f}"
            )
        )
    ]


    state_legend = ax.legend(

        handles=state_handles,

        title="Threshold",

        loc="lower left",

        bbox_to_anchor=(
            0.11,
            0.03
        ),

        frameon=True,

        fancybox=False,

        framealpha=1.0,

        facecolor="white",

        edgecolor="black",

        borderpad=0.40,

        handlelength=0.55,

        handleheight=0.45,

        handletextpad=0.45,

        labelspacing=0.30,

        fontsize=19
    )


    plt.setp(

        state_legend.get_title(),

        fontsize=18,

        fontweight="normal"
    )


    state_legend.get_frame().set_linewidth(
        1.6
    )


    # =========================================================================
    # 9.14 SENSITIVITY / SPECIFICITY LEGEND
    # =========================================================================

    metric_handles = [

        Line2D(

            [0],
            [0],

            color="#333333",

            linewidth=3.0,

            linestyle="-",

            label="Sensitivity"
        ),


        Line2D(

            [0],
            [0],

            color="#333333",

            linewidth=2.6,

            linestyle=(0, (6, 4)),

            label="Specificity"
        )
    ]


    metric_legend = ax.legend(

        handles=metric_handles,

        loc="lower right",

        bbox_to_anchor=(
            0.86,
            0.03
        ),

        frameon=True,

        fancybox=False,

        framealpha=1.0,

        facecolor="white",

        edgecolor="black",

        borderpad=0.45,

        handlelength=1.35,

        handletextpad=0.50,

        labelspacing=0.35
    )


    metric_legend.get_frame().set_linewidth(
        1.6
    )


    # Keep both legends visible
    ax.add_artist(
        state_legend
    )


    # =========================================================================
    # 9.15 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 9.16 SAVE
    # =========================================================================

    safe_title = (
        title
        .replace(" ", "_")
        .replace("-", "_")
    )


    tiff_file = (
        OUTPUT_DIR
        / f"Threshold_Tradeoff_{safe_title}.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"Threshold_Tradeoff_{safe_title}.pdf"
    )


    fig.savefig(

        tiff_file,

        dpi=1000,

        format="tiff",

        bbox_inches="tight",

        pad_inches=0.04,

        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(

        pdf_file,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0.04
    )


    plt.show()

    plt.close(fig)


    print(
        f"\n{title} trade-off files saved:"
    )

    print(
        f"TIFF (1000 dpi): {tiff_file}"
    )

    print(
        f"PDF (vector)   : {pdf_file}"
    )


# =============================================================================
# 10. FINISHED
# =============================================================================

print(
    "\nAll available six-model threshold trade-off figures have been created."
)

print(
    f"Output folder: {OUTPUT_DIR}"
)


# In[31]:


"""Publication-quality grouped bar plot comparing the six final models.

Metrics are calculated directly from the saved LOYO held-out predictions:

- Accuracy
- Balanced accuracy
- Macro-F1
- Macro-AUROC
- Macro-AUPRC

No model is retrained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. MODEL SETTINGS
# =============================================================================

TRUE_COL = "True_State"

STATES = [1, 2, 3]


MODEL_CONFIGS = [

    {
        "title": "CatBoost",
        "prefix_candidates": ["CatBoost"]
    },

    {
        "title": "XGBoost",
        "prefix_candidates": ["XGBoost", "XGB"]
    },

    {
        "title": "HistGradient\nBoosting",
        "prefix_candidates": [
            "HistGradientBoosting",
            "HGB",
            "HistGB"
        ]
    },

    {
        "title": "Soft\nVoting",
        "prefix_candidates": [
            "EqualSoftVoting",
            "Equal_Soft_Voting",
            "SoftVoting",
            "Soft_Voting",
            "Voting"
        ]
    },

    {
        "title": "TCN",
        "prefix_candidates": ["TCN"]
    },

    {
        "title": "CNN-LSTM",
        "prefix_candidates": [
            "CNNLSTM",
            "CNN_LSTM",
            "CNN-LSTM"
        ]
    },
]


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 26,

    "axes.labelsize": 30,

    "xtick.labelsize": 22,
    "ytick.labelsize": 24,

    "legend.fontsize": 20,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. METRIC COLOURS
# =============================================================================

METRIC_COLORS = {

    "Accuracy": "#4C78A8",

    "Balanced accuracy": "#59A14F",

    "Macro-F1": "#E07B39",

    "Macro-AUROC": "#B06AA3",

    "Macro-AUPRC": "#5AA7A7"
}


# =============================================================================
# 5. LOAD SAVED LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


if TRUE_COL not in data.columns:

    raise ValueError(
        f"'{TRUE_COL}' is missing.\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


y_true = (
    data[TRUE_COL]
    .astype(int)
    .to_numpy()
)


y_binary = label_binarize(
    y_true,
    classes=STATES
)


# =============================================================================
# 6. HELPER — FIND MODEL PROBABILITY COLUMNS
# =============================================================================

def resolve_probability_columns(
    all_columns,
    prefix_candidates
):

    for prefix in prefix_candidates:

        candidate_cols = [
            f"{prefix}_P_State_1",
            f"{prefix}_P_State_2",
            f"{prefix}_P_State_3",
        ]

        if all(
            col in all_columns
            for col in candidate_cols
        ):

            return candidate_cols

    return None


# =============================================================================
# 7. CALCULATE MODEL METRICS
# =============================================================================

records = []


for config in MODEL_CONFIGS:

    title = config["title"]

    probability_columns = resolve_probability_columns(
        data.columns,
        config["prefix_candidates"]
    )


    if probability_columns is None:

        print(
            f"\nSkipping {title}: "
            "probability columns were not found."
        )

        continue


    probability = (
        data[probability_columns]
        .astype(float)
        .to_numpy()
    )


    if not np.isfinite(probability).all():

        raise ValueError(
            f"Non-finite probabilities found for {title}."
        )


    # -------------------------------------------------------------------------
    # Multiclass predicted state = class with highest probability
    # -------------------------------------------------------------------------

    y_pred = (
        np.argmax(
            probability,
            axis=1
        )
        + 1
    )


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )


    macro_auroc = roc_auc_score(
        y_binary,
        probability,
        average="macro",
        multi_class="ovr"
    )


    macro_auprc = average_precision_score(
        y_binary,
        probability,
        average="macro"
    )


    records.append({

        "Model": title,

        "Accuracy": accuracy,

        "Balanced accuracy": balanced_accuracy,

        "Macro-F1": macro_f1,

        "Macro-AUROC": macro_auroc,

        "Macro-AUPRC": macro_auprc
    })


# =============================================================================
# 8. RESULT TABLE
# =============================================================================

results = pd.DataFrame(
    records
)


if results.empty:

    raise ValueError(
        "No model probability columns were found."
    )


print(
    "\nFinal model-performance comparison"
)

print(
    "=" * 72
)

print(
    results.round(3).to_string(index=False)
)


# Save numerical results
results.to_csv(
    ROOT
    / "Results"
    / "03_Final_Models"
    / "13_Final_Model_Metric_Comparison.csv",
    index=False
)


# =============================================================================
# 9. GROUPED BAR SETTINGS
# =============================================================================

METRICS = [

    "Accuracy",

    "Balanced accuracy",

    "Macro-F1",

    "Macro-AUROC",

    "Macro-AUPRC"
]


x = np.arange(
    len(results)
)


# Total width occupied by each group
GROUP_WIDTH = 0.80

BAR_WIDTH = (
    GROUP_WIDTH
    / len(METRICS)
)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14.5, 7.6)
)


# =============================================================================
# 11. DRAW GROUPED BARS
# =============================================================================

for metric_index, metric in enumerate(METRICS):


    offset = (
        metric_index
        -
        (len(METRICS) - 1) / 2
    ) * BAR_WIDTH


    bars = ax.bar(

        x + offset,

        results[metric],

        width=BAR_WIDTH * 0.92,

        color=METRIC_COLORS[metric],

        edgecolor="white",

        linewidth=0.8,

        label=metric,

        zorder=3
    )


# =============================================================================
# 12. AXES
# =============================================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(
    results["Model"]
)


ax.set_ylabel(
    "Score",
    labelpad=12
)


ax.set_ylim(
    0,
    1.05
)


ax.set_yticks(
    np.arange(
        0,
        1.01,
        0.20
    )
)


# =============================================================================
# 13. LIGHT HORIZONTAL GRID
# =============================================================================

ax.grid(

    axis="y",

    color="#D3D7DB",

    linewidth=0.65,

    alpha=0.50,

    zorder=0
)


ax.set_axisbelow(
    True
)


# =============================================================================
# 14. CLEAN SPINES
# =============================================================================

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)


ax.spines["left"].set_linewidth(
    1.5
)

ax.spines["bottom"].set_linewidth(
    1.5
)


ax.tick_params(
    axis="both",
    width=1.4,
    length=5
)


# =============================================================================
# 15. LEGEND
# =============================================================================

ax.legend(

    loc="lower center",

    bbox_to_anchor=(
        0.5,
        1.015
    ),

    ncol=5,

    frameon=False,

    columnspacing=1.0,

    handlelength=1.0,

    handletextpad=0.40
)


# =============================================================================
# 16. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 17. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Final_Model_Performance_Grouped_Bar.tiff"
)


pdf_file = (
    OUTPUT_DIR
    / "Final_Model_Performance_Grouped_Bar.pdf"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(fig)


# =============================================================================
# 18. OUTPUT SUMMARY
# =============================================================================

print(
    "\nGrouped model-performance figure saved:"
)

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)


# In[39]:


"""Figure 6(a): Global CatBoost SHAP importance.

Uses the saved final overall SHAP-importance results.

Expected saved columns:
- Rank
- Feature
- Mean_Absolute_SHAP
- Relative_Importance_Percent

No model is retrained and SHAP values are not recalculated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

SHAP_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "01_Overall_SHAP_Importance.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_06"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 28,

    "axes.labelsize": 30,
    "axes.titlesize": 30,

    "xtick.labelsize": 24,
    "ytick.labelsize": 25,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. PUBLICATION-QUALITY VARIABLE LABELS
# =============================================================================

FEATURE_LABELS = {

    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "SST": "SST",
    "SSS": "SSS",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO",
}


# =============================================================================
# 4. LOAD SAVED GLOBAL SHAP IMPORTANCE
# =============================================================================

shap_df = pd.read_csv(
    SHAP_FILE
)


# =============================================================================
# 5. VALIDATE SAVED COLUMNS
# =============================================================================

required_columns = [
    "Rank",
    "Feature",
    "Mean_Absolute_SHAP",
    "Relative_Importance_Percent"
]

missing = [
    column
    for column in required_columns
    if column not in shap_df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(shap_df.columns)}"
    )


# =============================================================================
# 6. PREPARE PLOT DATA
# =============================================================================

plot_df = shap_df[
    [
        "Rank",
        "Feature",
        "Mean_Absolute_SHAP",
        "Relative_Importance_Percent"
    ]
].copy()


plot_df["Mean_Absolute_SHAP"] = pd.to_numeric(
    plot_df["Mean_Absolute_SHAP"],
    errors="coerce"
)

plot_df["Relative_Importance_Percent"] = pd.to_numeric(
    plot_df["Relative_Importance_Percent"],
    errors="coerce"
)


plot_df = plot_df.dropna(
    subset=[
        "Feature",
        "Mean_Absolute_SHAP"
    ]
)


# =============================================================================
# 7. SORT FEATURES
# =============================================================================

plot_df = (
    plot_df
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )
    .reset_index(drop=True)
)


# =============================================================================
# 8. DISPLAY LABELS
# =============================================================================

plot_df["Display_Label"] = [

    FEATURE_LABELS.get(
        feature,
        feature
    )

    for feature in plot_df["Feature"]
]


# =============================================================================
# 9. PRINT FINAL IMPORTANCE TABLE
# =============================================================================

print(
    "\nGlobal CatBoost SHAP importance:"
)

print(
    "=" * 70
)

print(
    plot_df[
        [
            "Feature",
            "Mean_Absolute_SHAP",
            "Relative_Importance_Percent"
        ]
    ]
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(9.8, 8.5)
)


# =============================================================================
# 11. BAR COLORS
# =============================================================================

BAR_COLORS = [
    "#4E79A7", "#59A14F", "#E15759", "#F28E2B",
    "#76B7B2", "#B07AA1", "#EDC948", "#9C755F",
    "#BAB0AC", "#86BCB6", "#FF9DA7", "#8CD17D",
    "#B6992D", "#499894"
]

bar_colors = BAR_COLORS[:len(plot_df)]


# =============================================================================
# 12. HORIZONTAL BAR PLOT
# =============================================================================

bars = ax.barh(

    np.arange(
        len(plot_df)
    ),

    plot_df["Mean_Absolute_SHAP"],

    height=0.68,

    color=bar_colors,

    edgecolor="white",

    linewidth=0.8,

    zorder=3
)


# =============================================================================
# 13. TITLE
# =============================================================================

ax.set_title(

    "Overall SHAP importance",

    fontsize=30,

    fontweight="normal",

    pad=14
)


# =============================================================================
# 14. Y-AXIS FEATURE NAMES
# =============================================================================

ax.set_yticks(
    np.arange(
        len(plot_df)
    )
)

ax.set_yticklabels(
    plot_df["Display_Label"]
)


# =============================================================================
# 15. X AXIS
# =============================================================================

ax.set_xlabel(
    "Mean |SHAP value|",
    labelpad=10
)


# =============================================================================
# 16. VALUE LABELS
# =============================================================================

maximum = plot_df[
    "Mean_Absolute_SHAP"
].max()


for bar, shap_value in zip(
    bars,
    plot_df["Mean_Absolute_SHAP"]
):

    ax.text(

        shap_value + maximum * 0.018,

        bar.get_y()
        + bar.get_height() / 2,

        f"{shap_value:.3f}",

        ha="left",
        va="center",

        fontsize=23,

        fontweight="normal",

        color="#333333"
    )


# =============================================================================
# 17. X LIMIT
# =============================================================================

ax.set_xlim(
    0,
    maximum * 1.21
)


# =============================================================================
# 18. LIGHT VERTICAL GRID
# =============================================================================

ax.grid(

    axis="x",

    color="#D3D7DB",

    linewidth=0.70,

    alpha=0.50
)

ax.set_axisbelow(
    True
)


# =============================================================================
# 19. CLEAN AXES
# =============================================================================

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

ax.spines["left"].set_linewidth(
    1.2
)

ax.spines["bottom"].set_linewidth(
    1.2
)

ax.tick_params(
    axis="both",
    width=1.2,
    length=5
)


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 21. OUTPUT FILES
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Figure_6a_Global_SHAP_Importance.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Figure_6a_Global_SHAP_Importance.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "Figure_6a_Global_SHAP_Importance.svg"
)


# =============================================================================
# 22. SAVE TIFF
# =============================================================================

fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


# =============================================================================
# 23. SAVE PDF
# =============================================================================

fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


# =============================================================================
# 24. SAVE SVG
# =============================================================================

fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


# =============================================================================
# 25. SHOW / CLOSE
# =============================================================================

plt.show()

plt.close(fig)


# =============================================================================
# 26. OUTPUT SUMMARY
# =============================================================================

print(
    "\nFigure 6(a) — Global CatBoost SHAP importance saved:"
)

print(
    f"TIFF (1000 dpi): {tiff_file}"
)

print(
    f"PDF (vector)   : {pdf_file}"
)

print(
    f"SVG (vector)   : {svg_file}"
)


# In[42]:


"""Figure 6(b–d): Native CatBoost SHAP beeswarm plots.

Uses the saved raw long-format SHAP results:

    05_Raw_SHAP_Long_Format.csv

Actual saved columns:
- Months
- Observed_State
- Explained_State
- State_Name
- Feature
- Feature_Value
- SHAP_Value

IMPORTANT
---------
- CatBoost is NOT retrained.
- SHAP values are NOT recalculated.
- Explained_State identifies the CatBoost output being explained.
- Months is used as the observation identifier.
- SHAP's native beeswarm algorithm is used for point packing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

SHAP_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "05_Raw_SHAP_Long_Format.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_06"
    / "State_Beeswarm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FIGURE STYLE
# Same scale as the finalized global SHAP bar plot
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 28,

    "axes.labelsize": 30,
    "axes.titlesize": 30,

    "xtick.labelsize": 24,
    "ytick.labelsize": 25,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. PUBLICATION-QUALITY FEATURE LABELS
# =============================================================================

FEATURE_LABELS = {

    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "SST": "SST",
    "SSS": "SSS",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO",
}


# =============================================================================
# 4. BLUE → PINK SHAP COLOUR MAP
# Similar to the reference figure
# =============================================================================

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "SHAP_Blue_Pink",
    [
        "#2F80ED",   # low values: blue
        "#7B6FD0",   # middle: violet
        "#F12F63"    # high values: pink
    ]
)


# =============================================================================
# 5. LOAD SAVED RAW SHAP DATA
# =============================================================================

data = pd.read_csv(
    SHAP_FILE
)


# =============================================================================
# 6. VALIDATE EXACT SAVED COLUMNS
# =============================================================================

required_columns = [
    "Months",
    "Observed_State",
    "Explained_State",
    "State_Name",
    "Feature",
    "Feature_Value",
    "SHAP_Value"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


print("\nColumns found in raw SHAP file:")
print(list(data.columns))


# =============================================================================
# 7. CLEAN TYPES
# =============================================================================

data["Months"] = pd.to_datetime(
    data["Months"],
    errors="coerce"
)


data["SHAP_Value"] = pd.to_numeric(
    data["SHAP_Value"],
    errors="coerce"
)


data["Feature_Value"] = pd.to_numeric(
    data["Feature_Value"],
    errors="coerce"
)


data = data.dropna(
    subset=[
        "Months",
        "Explained_State",
        "Feature",
        "SHAP_Value",
        "Feature_Value"
    ]
)


# =============================================================================
# 8. NORMALIZE EXPLAINED_STATE TO 1, 2, 3
# =============================================================================

def extract_state_number(value):

    text = str(value).strip().lower()

    # -------------------------------------------------------------------------
    # Numeric state
    # -------------------------------------------------------------------------

    try:

        number = int(float(text))

        if number in [1, 2, 3]:
            return number

    except Exception:
        pass


    # -------------------------------------------------------------------------
    # Text labels such as:
    # State 1
    # Class 1
    # state_1
    # -------------------------------------------------------------------------

    for state in [1, 2, 3]:

        if str(state) in text:
            return state


    return np.nan


data["Explained_State_Number"] = (
    data["Explained_State"]
    .apply(extract_state_number)
)


data = data.dropna(
    subset=["Explained_State_Number"]
)


data["Explained_State_Number"] = (
    data["Explained_State_Number"]
    .astype(int)
)


print("\nExplained-state rows:")
print(
    data["Explained_State_Number"]
    .value_counts()
    .sort_index()
)


# =============================================================================
# 9. CHECK FOR DUPLICATE MONTH × FEATURE × EXPLAINED_STATE ENTRIES
# =============================================================================

duplicate_mask = data.duplicated(
    subset=[
        "Months",
        "Explained_State_Number",
        "Feature"
    ],
    keep=False
)


if duplicate_mask.any():

    duplicates = data.loc[
        duplicate_mask,
        [
            "Months",
            "Explained_State_Number",
            "Feature"
        ]
    ]

    raise ValueError(
        "\nDuplicate Month × Explained_State × Feature rows were found.\n"
        "The long SHAP table should contain one SHAP value per "
        "month-feature-state combination.\n\n"
        f"First duplicated rows:\n{duplicates.head(20)}"
    )


# =============================================================================
# 10. STATES
# =============================================================================

STATES = [
    1,
    2,
    3
]


PANEL_LETTERS = {
    1: "b",
    2: "c",
    3: "d"
}


# =============================================================================
# 11. CREATE NATIVE SHAP BEESWARM FOR EACH STATE
# =============================================================================

for state in STATES:


    # =========================================================================
    # 11.1 FILTER THE SHAP OUTPUT BEING EXPLAINED
    # =========================================================================

    state_df = data.loc[
        data["Explained_State_Number"] == state
    ].copy()


    if state_df.empty:

        print(
            f"\nNo SHAP rows were found for State {state}."
        )

        continue


    # =========================================================================
    # 11.2 BUILD SAMPLE × FEATURE SHAP MATRIX
    # =========================================================================

    shap_matrix = (
        state_df
        .pivot(
            index="Months",
            columns="Feature",
            values="SHAP_Value"
        )
        .sort_index()
    )


    # =========================================================================
    # 11.3 BUILD MATCHING SAMPLE × FEATURE VALUE MATRIX
    # =========================================================================

    feature_matrix = (
        state_df
        .pivot(
            index="Months",
            columns="Feature",
            values="Feature_Value"
        )
        .sort_index()
    )


    # =========================================================================
    # 11.4 ALIGN THE TWO MATRICES EXACTLY
    # =========================================================================

    common_months = (
        shap_matrix.index
        .intersection(
            feature_matrix.index
        )
    )


    common_features = [
        feature
        for feature in shap_matrix.columns
        if feature in feature_matrix.columns
    ]


    shap_matrix = shap_matrix.loc[
        common_months,
        common_features
    ]


    feature_matrix = feature_matrix.loc[
        common_months,
        common_features
    ]


    # =========================================================================
    # 11.5 REQUIRE COMPLETE SAMPLE × FEATURE MATRICES
    # =========================================================================

    incomplete_rows = (
        shap_matrix.isna().any(axis=1)
        |
        feature_matrix.isna().any(axis=1)
    )


    if incomplete_rows.any():

        print(
            f"\nState {state}: "
            f"dropping {int(incomplete_rows.sum())} incomplete months."
        )


        shap_matrix = shap_matrix.loc[
            ~incomplete_rows
        ]


        feature_matrix = feature_matrix.loc[
            ~incomplete_rows
        ]


    # =========================================================================
    # 11.6 VERIFY ALIGNMENT
    # =========================================================================

    if shap_matrix.shape != feature_matrix.shape:

        raise ValueError(
            f"\nState {state}: SHAP and feature-value matrices do not match.\n"
            f"SHAP matrix    : {shap_matrix.shape}\n"
            f"Feature matrix : {feature_matrix.shape}"
        )


    if shap_matrix.empty:

        raise ValueError(
            f"State {state}: no complete observations remain."
        )


    print(
        f"\nState {state} matrix shape: "
        f"{shap_matrix.shape}"
    )


    # =========================================================================
    # 11.7 STATE-SPECIFIC IMPORTANCE RANKING
    # =========================================================================

    state_importance = (
        shap_matrix
        .abs()
        .mean(axis=0)
        .sort_values(
            ascending=False
        )
    )


    feature_order = (
        state_importance
        .index
        .tolist()
    )


    # Explicitly arrange matrices in importance order.
    # SHAP can still sort internally, but this guarantees deterministic input.

    shap_matrix = shap_matrix[
        feature_order
    ]


    feature_matrix = feature_matrix[
        feature_order
    ]


    # =========================================================================
    # 11.8 PUBLICATION DISPLAY NAMES
    # =========================================================================

    display_names = [
        FEATURE_LABELS.get(
            feature,
            feature
        )
        for feature in feature_order
    ]


    # =========================================================================
    # 11.9 CREATE NATIVE SHAP BEESWARM
    # =========================================================================

    plt.figure(
        figsize=(9.8, 8.5)
    )


    shap.summary_plot(

        shap_values=shap_matrix.to_numpy(),

        features=feature_matrix.to_numpy(),

        feature_names=display_names,

        plot_type="dot",

        max_display=len(
            feature_order
        ),

        sort=True,

        color_bar=True,

        cmap=SHAP_CMAP,

        alpha=0.90,

        show=False,

        plot_size=(
            9.8,
            8.5
        )
    )


    # =========================================================================
    # 11.10 GET FIGURE / AXES CREATED BY SHAP
    # =========================================================================

    fig = plt.gcf()

    axes = fig.axes


    if len(axes) == 0:

        raise RuntimeError(
            f"SHAP did not create a plotting axis for State {state}."
        )


    main_ax = axes[0]


    colorbar_ax = (
        axes[1]
        if len(axes) > 1
        else None
    )


    # =========================================================================
    # 11.11 TITLE
    # =========================================================================

    main_ax.set_title(
        f"State {state}",
        fontsize=30,
        fontweight="normal",
        pad=12
    )


    # =========================================================================
    # 11.12 X-AXIS
    # =========================================================================

    main_ax.set_xlabel(
        "SHAP value",
        fontsize=30,
        labelpad=10
    )


    # =========================================================================
    # 11.13 Y-AXIS TITLE
    #
    # State 1 -> keep
    # State 2 -> remove
    # State 3 -> keep
    # =========================================================================

    if state == 2:

        main_ax.set_ylabel("")

    else:

        main_ax.set_ylabel(
            "Environmental variable",
            fontsize=30,
            labelpad=10
        )


    # =========================================================================
    # 11.14 TICK FONT SIZES
    # =========================================================================

    main_ax.tick_params(
        axis="x",
        labelsize=24,
        width=1.2,
        length=5
    )


    main_ax.tick_params(
        axis="y",
        labelsize=25,
        width=1.2,
        length=4
    )


    # =========================================================================
    # 11.15 ZERO REFERENCE LINE
    # =========================================================================

    main_ax.axvline(
        x=0,
        color="#777777",
        linewidth=1.15,
        zorder=0
    )


    # =========================================================================
    # 11.16 HORIZONTAL DOTTED GUIDES
    # Similar to reference SHAP beeswarm
    # =========================================================================

    main_ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.65,
        linestyle=(0, (1, 3)),
        alpha=0.75
    )


    main_ax.grid(
        False,
        axis="x"
    )


    main_ax.set_axisbelow(
        True
    )


    # =========================================================================
    # 11.17 CLEAN SPINES
    # =========================================================================

    main_ax.spines["top"].set_visible(
        False
    )

    main_ax.spines["right"].set_visible(
        False
    )


    main_ax.spines["left"].set_linewidth(
        1.2
    )

    main_ax.spines["bottom"].set_linewidth(
        1.2
    )


    # =========================================================================
    # 11.18 COLORBAR
    #
    # State 1:
    #   keep High/Low
    #   REMOVE Feature value title
    #
    # State 2:
    #   keep High/Low
    #   KEEP Feature value title
    #
    # State 3:
    #   keep High/Low
    #   REMOVE Feature value title
    # =========================================================================

    if colorbar_ax is not None:


        colorbar_ax.tick_params(
            labelsize=20,
            length=0
        )


        # ---------------------------------------------------------------------
        # Make High / Low easier to read
        # ---------------------------------------------------------------------

        for tick_label in colorbar_ax.get_yticklabels():

            tick_label.set_fontsize(
                20
            )


        # ---------------------------------------------------------------------
        # Colorbar title rule
        # ---------------------------------------------------------------------

        if state == 2:

            colorbar_ax.set_ylabel(
                "Feature value",
                fontsize=23,
                labelpad=10
            )

        else:

            colorbar_ax.set_ylabel("")


    # =========================================================================
    # 11.19 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 11.20 OUTPUT FILES
    # =========================================================================

    panel_letter = PANEL_LETTERS[
        state
    ]


    tiff_file = (
        OUTPUT_DIR
        / f"Figure_6{panel_letter}_State_{state}_SHAP_Beeswarm.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"Figure_6{panel_letter}_State_{state}_SHAP_Beeswarm.pdf"
    )


    svg_file = (
        OUTPUT_DIR
        / f"Figure_6{panel_letter}_State_{state}_SHAP_Beeswarm.svg"
    )


    # =========================================================================
    # 11.21 SAVE TIFF
    # =========================================================================

    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    # =========================================================================
    # 11.22 SAVE PDF
    # =========================================================================

    fig.savefig(
        pdf_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04
    )


    # =========================================================================
    # 11.23 SAVE SVG
    # =========================================================================

    fig.savefig(
        svg_file,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.04
    )


    # =========================================================================
    # 11.24 SHOW
    # =========================================================================

    plt.show()

    plt.close(
        fig
    )


    # =========================================================================
    # 11.25 PRINT STATE-SPECIFIC RANKING
    # =========================================================================

    print(
        f"\nState {state} mean |SHAP| ranking:"
    )

    print(
        "=" * 55
    )

    print(
        state_importance
        .round(4)
        .to_string()
    )


    print(
        f"\nState {state} beeswarm saved:"
    )

    print(
        f"TIFF : {tiff_file}"
    )

    print(
        f"PDF  : {pdf_file}"
    )

    print(
        f"SVG  : {svg_file}"
    )


# =============================================================================
# 12. FINISHED
# =============================================================================

print(
    "\nAll three native SHAP beeswarm plots have been created."
)


# In[48]:


"""Generate SHAP dependence plots for ALL final environmental predictors.

For every environmental variable:
- State 1, State 2 and State 3 are shown together.
- Raw SHAP observations are displayed.
- A LOWESS response curve is drawn for each ecological state.
- No main title.
- Legend is placed in the upper-right corner WITHOUT a box.
- Large fonts are used for later multi-panel assembly.

Saved SHAP values are used directly.
CatBoost is NOT retrained and SHAP is NOT recalculated.
"""

from pathlib import Path

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

RAW_SHAP_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "05_Raw_SHAP_Long_Format.csv"
)

IMPORTANCE_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "01_Overall_SHAP_Importance.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_07"
    / "Dependence_All_Features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FIGURE STYLE
# Increased further for multi-panel use
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 38,

    "axes.labelsize": 44,
    "axes.titlesize": 40,

    "xtick.labelsize": 37,
    "ytick.labelsize": 37,

    "legend.fontsize": 34,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. STATE COLOURS
# =============================================================================

STATE_COLORS = {

    1: "#D55E5E",   # State 1
    2: "#4F9D92",   # State 2
    3: "#7A6BB1"    # State 3
}


# =============================================================================
# 4. PUBLICATION DISPLAY LABELS
# =============================================================================

FEATURE_LABELS = {

    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "SST": "SST",
    "SSS": "SSS",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "MHWmean": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHWmax": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHWcum": r"$\mathrm{MHW}_{\mathrm{cum}}$",

    "WPI": "WPI",
    "NINO_3.4": "Niño 3.4",
    "PDO": "PDO"
}


# =============================================================================
# 5. LOAD SAVED SHAP DATA
# =============================================================================

data = pd.read_csv(
    RAW_SHAP_FILE
)

importance = pd.read_csv(
    IMPORTANCE_FILE
)


# =============================================================================
# 6. VALIDATE RAW SHAP FILE
# =============================================================================

required_raw = [
    "Explained_State",
    "Feature",
    "Feature_Value",
    "SHAP_Value"
]

missing_raw = [
    column
    for column in required_raw
    if column not in data.columns
]

if missing_raw:

    raise ValueError(
        f"Missing raw SHAP columns: {missing_raw}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 7. VALIDATE IMPORTANCE FILE
# =============================================================================

required_importance = [
    "Feature",
    "Mean_Absolute_SHAP"
]

missing_importance = [
    column
    for column in required_importance
    if column not in importance.columns
]

if missing_importance:

    raise ValueError(
        f"Missing SHAP importance columns: {missing_importance}\n\n"
        f"Available columns:\n{list(importance.columns)}"
    )


# =============================================================================
# 8. CLEAN NUMERIC COLUMNS
# =============================================================================

data["Feature_Value"] = pd.to_numeric(
    data["Feature_Value"],
    errors="coerce"
)

data["SHAP_Value"] = pd.to_numeric(
    data["SHAP_Value"],
    errors="coerce"
)

importance["Mean_Absolute_SHAP"] = pd.to_numeric(
    importance["Mean_Absolute_SHAP"],
    errors="coerce"
)


# =============================================================================
# 9. NORMALIZE EXPLAINED STATE
# =============================================================================

def extract_state_number(value):

    text = str(value).strip().lower()

    try:

        numeric = int(float(text))

        if numeric in [1, 2, 3]:
            return numeric

    except Exception:
        pass


    for state in [1, 2, 3]:

        if str(state) in text:
            return state


    return np.nan


data["State_Number"] = (
    data["Explained_State"]
    .apply(extract_state_number)
)


data = data.dropna(
    subset=[
        "Feature",
        "Feature_Value",
        "SHAP_Value",
        "State_Number"
    ]
)


data["State_Number"] = (
    data["State_Number"]
    .astype(int)
)


# =============================================================================
# 10. GET ALL ENVIRONMENTAL VARIABLES
# Ordered by overall SHAP importance
# =============================================================================

feature_order = (

    importance

    .dropna(
        subset=[
            "Feature",
            "Mean_Absolute_SHAP"
        ]
    )

    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    ["Feature"]

    .tolist()
)


raw_features = set(
    data["Feature"].unique()
)


feature_order = [
    feature
    for feature in feature_order
    if feature in raw_features
]


print(
    "\nEnvironmental variables to plot:"
)

print(
    "=" * 55
)


for number, feature in enumerate(
    feature_order,
    start=1
):

    print(
        f"{number:>2}. {feature}"
    )


print(
    f"\nTotal variables: {len(feature_order)}"
)


# =============================================================================
# 11. SAFE FILE-NAME FUNCTION
# =============================================================================

def safe_filename(text):

    text = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(text)
    )

    return text.strip("_")


# =============================================================================
# 12. CREATE DEPENDENCE PLOTS
# =============================================================================

summary_rows = []


for feature_number, feature in enumerate(
    feature_order,
    start=1
):


    # =========================================================================
    # 12.1 SELECT FEATURE
    # =========================================================================

    feature_df = data.loc[
        data["Feature"] == feature
    ].copy()


    if feature_df.empty:

        print(
            f"\nSkipping {feature}: no SHAP observations."
        )

        continue


    # =========================================================================
    # 12.2 DIAGNOSTIC SUMMARY
    # =========================================================================

    diagnostic = (

        feature_df

        .groupby(
            "State_Number"
        )["SHAP_Value"]

        .agg(
            [
                "count",
                "min",
                "max",
                "mean",
                "std"
            ]
        )
    )


    print(
        "\n" + "=" * 72
    )

    print(
        f"{feature_number}. {feature}"
    )

    print(
        "=" * 72
    )

    print(
        diagnostic.round(4)
    )


    for state in [1, 2, 3]:

        if state in diagnostic.index:

            summary_rows.append({

                "Feature": feature,

                "State": state,

                "Count":
                    diagnostic.loc[state, "count"],

                "Minimum_SHAP":
                    diagnostic.loc[state, "min"],

                "Maximum_SHAP":
                    diagnostic.loc[state, "max"],

                "Mean_SHAP":
                    diagnostic.loc[state, "mean"],

                "SD_SHAP":
                    diagnostic.loc[state, "std"]
            })


    # =========================================================================
    # 12.3 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10.2, 8.8)
    )


    # =========================================================================
    # 12.4 SHAP = 0 REFERENCE
    # =========================================================================

    ax.axhline(

        y=0,

        color="#777777",

        linewidth=1.9,

        linestyle=(0, (5, 5)),

        zorder=1
    )


    # =========================================================================
    # 12.5 THREE ECOLOGICAL STATES
    # =========================================================================

    for state in [1, 2, 3]:


        state_df = feature_df.loc[
            feature_df["State_Number"] == state
        ].copy()


        if state_df.empty:
            continue


        state_df = state_df.sort_values(
            "Feature_Value"
        )


        x = (
            state_df["Feature_Value"]
            .to_numpy(float)
        )


        y = (
            state_df["SHAP_Value"]
            .to_numpy(float)
        )


        # ---------------------------------------------------------------------
        # Raw SHAP observations
        # ---------------------------------------------------------------------

        ax.scatter(

            x,
            y,

            s=62,

            color=STATE_COLORS[state],

            alpha=0.27,

            edgecolors="none",

            zorder=2
        )


        # ---------------------------------------------------------------------
        # LOWESS trend
        # ---------------------------------------------------------------------

        unique_x = np.unique(
            x[np.isfinite(x)]
        )


        if len(unique_x) >= 5:


            smooth = lowess(

                endog=y,

                exog=x,

                frac=0.25,

                it=1,

                return_sorted=True
            )


            ax.plot(

                smooth[:, 0],

                smooth[:, 1],

                color=STATE_COLORS[state],

                linewidth=5.0,

                label=f"State {state}",

                zorder=4
            )


        else:

            ax.plot(

                [],

                [],

                color=STATE_COLORS[state],

                linewidth=5.0,

                label=f"State {state}"
            )


    # =========================================================================
    # 12.6 AXIS LABELS
    # =========================================================================

    display_label = FEATURE_LABELS.get(
        feature,
        feature
    )


    ax.set_xlabel(

        display_label,

        fontsize=44,

        labelpad=15
    )


    ax.set_ylabel(

        "SHAP value",

        fontsize=44,

        labelpad=15
    )


    # =========================================================================
    # 12.7 NO MAIN TITLE
    # =========================================================================

    # No title intentionally.


    # =========================================================================
    # 12.8 LEGEND — UPPER RIGHT, NO BOX
    # =========================================================================

    ax.legend(

        loc="upper right",

        fontsize=34,

        frameon=False,

        handlelength=1.35,

        handletextpad=0.60,

        labelspacing=0.42
    )


    # =========================================================================
    # 12.9 CLEAN AXES
    # =========================================================================

    ax.grid(
        False
    )


    ax.spines["top"].set_visible(
        False
    )


    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        1.8
    )


    ax.spines["bottom"].set_linewidth(
        1.8
    )


    ax.tick_params(

        axis="both",

        labelsize=37,

        width=1.7,

        length=7
    )


    # =========================================================================
    # 12.10 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 12.11 OUTPUT NAMES
    # =========================================================================

    safe_feature = safe_filename(
        feature
    )


    prefix = (
        f"{feature_number:02d}_{safe_feature}"
    )


    tiff_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.pdf"
    )


    svg_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.svg"
    )


    # =========================================================================
    # 12.12 SAVE TIFF
    # =========================================================================

    fig.savefig(

        tiff_file,

        dpi=1000,

        format="tiff",

        bbox_inches="tight",

        pad_inches=0.04,

        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    # =========================================================================
    # 12.13 SAVE PDF
    # =========================================================================

    fig.savefig(

        pdf_file,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0.04
    )


    # =========================================================================
    # 12.14 SAVE SVG
    # =========================================================================

    fig.savefig(

        svg_file,

        format="svg",

        bbox_inches="tight",

        pad_inches=0.04
    )


    # =========================================================================
    # 12.15 SHOW / CLOSE
    # =========================================================================

    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# 13. SAVE SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "07_SHAP_Dependence_State_Summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# 14. FINISHED
# =============================================================================

print(
    "\n" + "=" * 72
)

print(
    "ALL SHAP DEPENDENCE PLOTS COMPLETED"
)

print(
    "=" * 72
)


print(
    f"\nNumber of environmental variables plotted: "
    f"{len(feature_order)}"
)


print(
    f"\nFigures saved to:\n{OUTPUT_DIR}"
)


print(
    f"\nSummary saved to:\n{summary_file}"
)


# In[50]:


"""Final SHAP dependence plots for six selected environmental predictors.

Selected predictors:
1. MLD
2. SST
3. SSS
4. PAR
5. pCO2
6. Niño 3.4

Final presentation:
- State 1, State 2 and State 3 shown together
- Raw SHAP observations + LOWESS curves
- legend ONLY on MLD
- no main titles
- exactly four major x-axis ticks per plot
- Y-AXIS AUTOSCALED independently for each environmental variable
- similar y-tick density without forcing the same range
- identical figure size and typography
- TIFF, PDF and SVG output

Saved SHAP values are used directly.
CatBoost is NOT retrained and SHAP is NOT recalculated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

RAW_SHAP_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP"
    / "05_Raw_SHAP_Long_Format.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_07"
    / "Final_Six_Dependence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FINAL SIX FEATURES
# =============================================================================

FEATURES = [
    "MLD",
    "SST",
    "SSS",
    "PAR",
    "SPCo2",
    "NINO_3.4",
]


# =============================================================================
# 3. PUBLICATION DISPLAY LABELS
# =============================================================================

FEATURE_LABELS = {

    "MLD": "MLD",
    "SST": "SST",
    "SSS": "SSS",
    "PAR": "PAR",

    "SPCo2": r"$p\mathrm{CO}_2$",
    "spCO2": r"$p\mathrm{CO}_2$",

    "NINO_3.4": "Niño 3.4",
}


# =============================================================================
# 4. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 38,

    "axes.labelsize": 44,

    "xtick.labelsize": 37,
    "ytick.labelsize": 37,

    "legend.fontsize": 34,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 5. STATE COLOURS
# =============================================================================

STATE_COLORS = {

    1: "#D55E5E",   # State 1
    2: "#4F9D92",   # State 2
    3: "#7A6BB1",   # State 3
}


# =============================================================================
# 6. LOAD SAVED SHAP DATA
# =============================================================================

data = pd.read_csv(
    RAW_SHAP_FILE
)


# =============================================================================
# 7. VALIDATE COLUMNS
# =============================================================================

required = [
    "Explained_State",
    "Feature",
    "Feature_Value",
    "SHAP_Value"
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 8. CLEAN NUMERIC COLUMNS
# =============================================================================

data["Feature_Value"] = pd.to_numeric(
    data["Feature_Value"],
    errors="coerce"
)

data["SHAP_Value"] = pd.to_numeric(
    data["SHAP_Value"],
    errors="coerce"
)


# =============================================================================
# 9. NORMALIZE EXPLAINED STATE
# =============================================================================

def extract_state_number(value):

    text = str(value).strip().lower()

    try:

        number = int(float(text))

        if number in [1, 2, 3]:
            return number

    except Exception:
        pass


    for state in [1, 2, 3]:

        if str(state) in text:
            return state


    return np.nan


data["State_Number"] = (
    data["Explained_State"]
    .apply(extract_state_number)
)


data = data.dropna(
    subset=[
        "Feature",
        "Feature_Value",
        "SHAP_Value",
        "State_Number"
    ]
)


data["State_Number"] = (
    data["State_Number"]
    .astype(int)
)


# =============================================================================
# 10. CHECK SIX FEATURES
# =============================================================================

available_features = set(
    data["Feature"].unique()
)


missing_features = [
    feature
    for feature in FEATURES
    if feature not in available_features
]


if missing_features:

    raise ValueError(
        f"Selected features missing from SHAP file: {missing_features}\n\n"
        f"Available features:\n{sorted(available_features)}"
    )


# =============================================================================
# 11. FOUR X-TICK FUNCTION
# =============================================================================

def make_four_x_ticks(values):

    values = np.asarray(
        values,
        dtype=float
    )

    xmin = np.nanmin(values)
    xmax = np.nanmax(values)

    span = xmax - xmin


    # Small visual padding
    pad = span * 0.025

    if np.isclose(span, 0):
        pad = 0.5


    plot_min = xmin - pad
    plot_max = xmax + pad


    # Exactly four x-axis ticks
    ticks = np.linspace(
        xmin,
        xmax,
        4
    )


    # ---------------------------------------------------------
    # Sensible formatting
    # ---------------------------------------------------------

    if span >= 10:

        ticks = np.round(
            ticks,
            0
        )

        decimals = 0


    elif span >= 2:

        ticks = np.round(
            ticks,
            1
        )

        decimals = 1


    else:

        ticks = np.round(
            ticks,
            2
        )

        decimals = 2


    return (
        plot_min,
        plot_max,
        ticks,
        decimals
    )


# =============================================================================
# 12. FEATURE-SPECIFIC Y AUTOSCALE
# =============================================================================

def make_y_limits(values):

    """
    Autoscale SHAP y-axis separately for each feature.

    The observed SHAP range determines the limits.
    A small margin is added above and below.

    Zero is always included.
    """

    values = np.asarray(
        values,
        dtype=float
    )


    ymin = np.nanmin(values)
    ymax = np.nanmax(values)


    # Always include SHAP = 0
    ymin = min(
        ymin,
        0
    )

    ymax = max(
        ymax,
        0
    )


    span = ymax - ymin


    if np.isclose(
        span,
        0
    ):

        span = 1.0


    # 8% visual breathing room
    padding = span * 0.08


    return (
        ymin - padding,
        ymax + padding
    )


# =============================================================================
# 13. OUTPUT FILE PREFIXES
# =============================================================================

FILE_NAMES = {

    "MLD": "01_MLD",
    "SST": "02_SST",
    "SSS": "03_SSS",
    "PAR": "04_PAR",
    "SPCo2": "05_pCO2",
    "NINO_3.4": "06_Nino_3_4",
}


# =============================================================================
# 14. CREATE FINAL SIX DEPENDENCE PLOTS
# =============================================================================

for feature in FEATURES:


    # =========================================================================
    # 14.1 FEATURE DATA
    # =========================================================================

    feature_df = data.loc[
        data["Feature"] == feature
    ].copy()


    # =========================================================================
    # 14.2 X AXIS SETTINGS
    # =========================================================================

    (
        x_min,
        x_max,
        x_ticks,
        x_decimals

    ) = make_four_x_ticks(
        feature_df["Feature_Value"]
    )


    # =========================================================================
    # 14.3 FEATURE-SPECIFIC Y AUTOSCALE
    # =========================================================================

    y_min, y_max = make_y_limits(
        feature_df["SHAP_Value"]
    )


    # =========================================================================
    # 14.4 CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10.2, 8.8)
    )


    # =========================================================================
    # 14.5 SHAP = 0 REFERENCE
    # =========================================================================

    ax.axhline(

        y=0,

        color="#777777",

        linewidth=1.9,

        linestyle=(0, (5, 5)),

        zorder=1
    )


    # =========================================================================
    # 14.6 THREE ECOLOGICAL STATES
    # =========================================================================

    for state in [1, 2, 3]:


        state_df = feature_df.loc[
            feature_df["State_Number"] == state
        ].copy()


        if state_df.empty:
            continue


        state_df = state_df.sort_values(
            "Feature_Value"
        )


        x = (
            state_df["Feature_Value"]
            .to_numpy(float)
        )


        y = (
            state_df["SHAP_Value"]
            .to_numpy(float)
        )


        # ---------------------------------------------------------------------
        # RAW SHAP OBSERVATIONS
        # ---------------------------------------------------------------------

        ax.scatter(

            x,
            y,

            s=62,

            color=STATE_COLORS[state],

            alpha=0.27,

            edgecolors="none",

            zorder=2
        )


        # ---------------------------------------------------------------------
        # LOWESS TREND
        # ---------------------------------------------------------------------

        unique_x = np.unique(
            x[np.isfinite(x)]
        )


        if len(unique_x) >= 5:


            smooth = lowess(

                endog=y,

                exog=x,

                frac=0.25,

                it=1,

                return_sorted=True
            )


            ax.plot(

                smooth[:, 0],

                smooth[:, 1],

                color=STATE_COLORS[state],

                linewidth=5.0,

                label=f"State {state}",

                zorder=4
            )


        else:


            ax.plot(

                [],

                [],

                color=STATE_COLORS[state],

                linewidth=5.0,

                label=f"State {state}"
            )


    # =========================================================================
    # 14.7 X AXIS
    # =========================================================================

    display_label = FEATURE_LABELS.get(
        feature,
        feature
    )


    ax.set_xlabel(

        display_label,

        fontsize=44,

        labelpad=15
    )


    ax.set_xlim(
        x_min,
        x_max
    )


    ax.set_xticks(
        x_ticks
    )


    # -------------------------------------------------------------------------
    # X-axis formatting
    # -------------------------------------------------------------------------

    if x_decimals == 0:

        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%.0f")
        )


    elif x_decimals == 1:

        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )


    else:

        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%.2f")
        )


    # =========================================================================
    # 14.8 Y AXIS — AUTOSCALED FOR EACH FEATURE
    # =========================================================================

    ax.set_ylabel(

        "SHAP value",

        fontsize=44,

        labelpad=15
    )


    ax.set_ylim(
        y_min,
        y_max
    )


    # Similar tick density across panels,
    # but NOT the same numerical range.
    ax.yaxis.set_major_locator(

        MaxNLocator(
            nbins=5
        )
    )


    # =========================================================================
    # 14.9 LEGEND
    #
    # ONLY MLD
    # =========================================================================

    if feature == "MLD":

        ax.legend(

            loc="upper right",

            fontsize=34,

            frameon=False,

            handlelength=1.35,

            handletextpad=0.60,

            labelspacing=0.42
        )


    # =========================================================================
    # 14.10 NO MAIN TITLE
    # =========================================================================

    # Intentionally no title.


    # =========================================================================
    # 14.11 CLEAN AXES
    # =========================================================================

    ax.grid(
        False
    )


    ax.spines["top"].set_visible(
        False
    )


    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        1.8
    )


    ax.spines["bottom"].set_linewidth(
        1.8
    )


    ax.tick_params(

        axis="both",

        labelsize=37,

        width=1.7,

        length=7
    )


    # =========================================================================
    # 14.12 FINAL LAYOUT
    # =========================================================================

    fig.tight_layout(
        pad=0.8
    )


    # =========================================================================
    # 14.13 OUTPUT FILES
    # =========================================================================

    prefix = FILE_NAMES[
        feature
    ]


    tiff_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.tiff"
    )


    pdf_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.pdf"
    )


    svg_file = (
        OUTPUT_DIR
        / f"{prefix}_SHAP_Dependence.svg"
    )


    # =========================================================================
    # 14.14 SAVE TIFF
    # =========================================================================

    fig.savefig(

        tiff_file,

        dpi=1000,

        format="tiff",

        bbox_inches="tight",

        pad_inches=0.04,

        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    # =========================================================================
    # 14.15 SAVE PDF
    # =========================================================================

    fig.savefig(

        pdf_file,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0.04
    )


    # =========================================================================
    # 14.16 SAVE SVG
    # =========================================================================

    fig.savefig(

        svg_file,

        format="svg",

        bbox_inches="tight",

        pad_inches=0.04
    )


    # =========================================================================
    # 14.17 SHOW / CLOSE
    # =========================================================================

    plt.show()

    plt.close(
        fig
    )


    print(
        f"\n{feature} dependence plot saved."
    )

    print(
        f"Y range: {y_min:.3f} to {y_max:.3f}"
    )


# =============================================================================
# 15. FINISHED
# =============================================================================

print(
    "\n" + "=" * 65
)

print(
    "FINAL SIX SHAP DEPENDENCE PLOTS COMPLETED"
)

print(
    "=" * 65
)


print(
    f"\nOutput folder:\n{OUTPUT_DIR}"
)


# In[62]:


"""Create two separate temporal ecological-state panels.

Panel 1: True ecological state
Panel 2: Predicted ecological state

Uses saved LOYO predictions only.
No model is retrained.

Both panels use exactly the same figure dimensions as the
final LOYO balanced-accuracy panel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_08"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",

    "font.size": 24,
    "axes.labelsize": 30,
    "axes.titlesize": 30,

    "xtick.labelsize": 21,
    "ytick.labelsize": 23,

    "legend.fontsize": 19,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. FIXED PANEL SIZE
#
# Same width and height as the finalized accuracy line plot.
# =============================================================================

FIGURE_SIZE = (13.2, 4.6)


# =============================================================================
# 4. COLOURS
# =============================================================================

STATE_COLORS = {
    1: "#D55E5E",   # muted red
    2: "#4F9D92",   # teal
    3: "#7A6BB1"    # purple
}

STEP_LINE_COLOR = "#404040"


# =============================================================================
# 5. LOAD DATA
# =============================================================================

data = pd.read_csv(PREDICTION_FILE)

DATE_COL = "Months"
TRUE_COL = "True_State"

if DATE_COL not in data.columns:
    raise ValueError(f"'{DATE_COL}' not found in file.")

if TRUE_COL not in data.columns:
    raise ValueError(f"'{TRUE_COL}' not found in file.")


# =============================================================================
# 6. GET CATBOOST PREDICTION
# =============================================================================

if "CatBoost_Prediction" in data.columns:
    PRED_COL = "CatBoost_Prediction"

else:

    PROB_COLS = [
        "CatBoost_P_State_1",
        "CatBoost_P_State_2",
        "CatBoost_P_State_3"
    ]

    missing_prob = [
        col
        for col in PROB_COLS
        if col not in data.columns
    ]

    if missing_prob:
        raise ValueError(
            "Could not find CatBoost prediction column or "
            "CatBoost probability columns.\n"
            f"Missing columns: {missing_prob}"
        )

    PRED_COL = None


# =============================================================================
# 7. CLEAN DATA
# =============================================================================

data[DATE_COL] = pd.to_datetime(
    data[DATE_COL],
    errors="coerce"
)

required_for_drop = [
    DATE_COL,
    TRUE_COL
]

if PRED_COL is not None:
    required_for_drop.append(PRED_COL)

data = data.dropna(
    subset=required_for_drop
).copy()

data = (
    data
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)


if PRED_COL is None:

    probs = (
        data[PROB_COLS]
        .astype(float)
        .to_numpy()
    )

    data["CatBoost_Prediction"] = (
        np.argmax(
            probs,
            axis=1
        )
        + 1
    )

    PRED_COL = "CatBoost_Prediction"


data[TRUE_COL] = (
    data[TRUE_COL]
    .astype(int)
)

data[PRED_COL] = (
    data[PRED_COL]
    .astype(int)
)


# =============================================================================
# 8. EXTRACT SERIES
# =============================================================================

dates = data[DATE_COL]

true_state = (
    data[TRUE_COL]
    .to_numpy()
)

pred_state = (
    data[PRED_COL]
    .to_numpy()
)


accuracy = np.mean(
    true_state == pred_state
)


print(
    f"\nNumber of monthly observations : {len(data)}"
)

print(
    f"CatBoost temporal accuracy     : {accuracy:.3f}"
)


# =============================================================================
# 9. LEGEND HANDLES
# =============================================================================

legend_handles = [

    Patch(
        facecolor=STATE_COLORS[1],
        edgecolor="none",
        label="State 1"
    ),

    Patch(
        facecolor=STATE_COLORS[2],
        edgecolor="none",
        label="State 2"
    ),

    Patch(
        facecolor=STATE_COLORS[3],
        edgecolor="none",
        label="State 3"
    )
]


# =============================================================================
# 10. HELPER FUNCTION
# =============================================================================

def draw_single_panel(
    x_dates,
    y_states,
    title_text,
    output_stub,
    show_legend=False,
    show_accuracy=False,
    hide_upper_year_labels=False
):

    # =========================================================================
    # SAME FIGURE SIZE FOR BOTH PANELS
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE
    )


    # -------------------------------------------------------------------------
    # Step line
    # -------------------------------------------------------------------------

    ax.step(
        x_dates,
        y_states,
        where="mid",
        color=STEP_LINE_COLOR,
        linewidth=2.3,
        zorder=2
    )


    # -------------------------------------------------------------------------
    # State-coloured markers
    # -------------------------------------------------------------------------

    for state in [1, 2, 3]:

        mask = (
            y_states == state
        )

        ax.scatter(
            x_dates[mask],
            y_states[mask],
            s=30,
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3
        )


    # -------------------------------------------------------------------------
    # Titles and labels
    # -------------------------------------------------------------------------

    ax.set_title(
        title_text,
        pad=10,
        fontweight="normal"
    )


    ax.set_ylabel("")


    if hide_upper_year_labels:

        ax.set_xlabel("")

    else:

        ax.set_xlabel(
            "Year",
            labelpad=10
        )


    ax.set_ylim(
        0.6,
        3.4
    )


    ax.set_yticks(
        [1, 2, 3]
    )


    ax.set_yticklabels(
        [
            "State 1",
            "State 2",
            "State 3"
        ]
    )


    # -------------------------------------------------------------------------
    # X-axis formatting
    # -------------------------------------------------------------------------

    ax.xaxis.set_major_locator(
        mdates.YearLocator(2)
    )


    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )


    ax.set_xlim(
        x_dates.min(),
        x_dates.max()
    )


    if hide_upper_year_labels:

        ax.tick_params(
            axis="x",
            labelbottom=False
        )


    # -------------------------------------------------------------------------
    # Style
    # -------------------------------------------------------------------------

    ax.grid(
        axis="y",
        color="#D3D7DB",
        linewidth=0.7,
        alpha=0.72
    )


    ax.set_axisbelow(
        True
    )


    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


    ax.spines["left"].set_linewidth(
        1.15
    )

    ax.spines["bottom"].set_linewidth(
        1.15
    )


    ax.tick_params(
        axis="both",
        width=1.0,
        length=4
    )


    # -------------------------------------------------------------------------
    # Legend for upper panel only
    # -------------------------------------------------------------------------

    if show_legend:

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.995, 1.10),
            ncol=3,
            frameon=False,
            columnspacing=1.0,
            handlelength=0.9,
            handletextpad=0.35,
            borderaxespad=0.1
        )


    # -------------------------------------------------------------------------
    # Accuracy text for lower panel only
    # -------------------------------------------------------------------------

    if show_accuracy:

        ax.text(
            0.995,
            1.07,
            f"Temporal accuracy = {accuracy:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=18,
            color="#4A4A4A"
        )


    # =========================================================================
    # FIXED AXIS POSITION
    #
    # This is intentionally identical for True and Predicted so that the
    # actual plotting rectangles have the same width and height.
    # =========================================================================

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.22,
        top=0.80
    )


    # =========================================================================
    # SAVE
    #
    # IMPORTANT:
    # Do NOT use bbox_inches="tight" here.
    # Tight cropping would give the two panels different final dimensions
    # because one contains a Year label and accuracy text.
    # =========================================================================

    tiff_file = (
        OUTPUT_DIR
        / f"{output_stub}.tiff"
    )

    pdf_file = (
        OUTPUT_DIR
        / f"{output_stub}.pdf"
    )

    svg_file = (
        OUTPUT_DIR
        / f"{output_stub}.svg"
    )


    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(
        pdf_file,
        format="pdf"
    )


    fig.savefig(
        svg_file,
        format="svg"
    )


    plt.show()

    plt.close(fig)


    print(
        f"\nSaved panel: {title_text}"
    )

    print(
        f"TIFF : {tiff_file}"
    )

    print(
        f"PDF  : {pdf_file}"
    )

    print(
        f"SVG  : {svg_file}"
    )


# =============================================================================
# 11. CREATE TRUE PANEL
# =============================================================================

draw_single_panel(
    x_dates=dates,
    y_states=true_state,
    title_text="True",
    output_stub="Figure_08a_True",
    show_legend=True,
    show_accuracy=False,
    hide_upper_year_labels=True
)


# =============================================================================
# 12. CREATE PREDICTED PANEL
# =============================================================================

draw_single_panel(
    x_dates=dates,
    y_states=pred_state,
    title_text="Predicted",
    output_stub="Figure_08b_Predicted",
    show_legend=False,
    show_accuracy=True,
    hide_upper_year_labels=False
)


# In[64]:


"""Refined LOYO balanced-accuracy panel only.

Optimized for combining later with the True and Predicted ecological-state panels:
- compact height
- y-axis starts at 0.30
- no x-axis labels or x-axis title
- visible x-axis tick marks (labels hidden)
- compact one-line legend at the top
- no main title
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

RESULT_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "LOYO_Annual_Model_Performance.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_08"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. MODEL SETTINGS
# =============================================================================

MODELS = {
    "CatBoost": "CatBoost_Prediction",
    "XGBoost": "XGBoost_Prediction",
    "HistGradientBoosting": "HistGradientBoosting_Prediction",
    "Soft Voting": "Equal_Soft_Voting_Prediction",
    "TCN": "TCN_Prediction",
    "CNN-LSTM": "CNN_LSTM_Prediction",
}

LEGEND_LABELS = {
    "CatBoost": "CatBoost",
    "XGBoost": "XGB",
    "HistGradientBoosting": "HGB",
    "Soft Voting": "Soft Vote",
    "TCN": "TCN",
    "CNN-LSTM": "CNN-LSTM",
}

MODEL_COLORS = {
    "CatBoost": "#3B6FB6",
    "XGBoost": "#D97732",
    "HistGradientBoosting": "#5A9B6F",
    "Soft Voting": "#7B63A8",
    "TCN": "#C16056",
    "CNN-LSTM": "#4D8F90",
}


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.labelweight": "normal",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,   # increased a bit
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 4. LOAD DATA
# =============================================================================

data = pd.read_csv(PREDICTION_FILE)

YEAR_COL = "Test_Year"
TRUE_COL = "True_State"

required = [YEAR_COL, TRUE_COL, *MODELS.values()]
missing = [col for col in required if col not in data.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. CLEAN DATA
# =============================================================================

data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TRUE_COL] = pd.to_numeric(data[TRUE_COL], errors="coerce")

for pred_col in MODELS.values():
    data[pred_col] = pd.to_numeric(data[pred_col], errors="coerce")

data = data.dropna(subset=[YEAR_COL, TRUE_COL, *MODELS.values()]).copy()

data[YEAR_COL] = data[YEAR_COL].astype(int)
data[TRUE_COL] = data[TRUE_COL].astype(int)

for pred_col in MODELS.values():
    data[pred_col] = data[pred_col].astype(int)


# =============================================================================
# 6. CALCULATE YEARLY PERFORMANCE
# =============================================================================

records = []

for year, year_df in data.groupby(YEAR_COL, sort=True):

    y_true = year_df[TRUE_COL].to_numpy()

    for model, pred_col in MODELS.items():

        y_pred = year_df[pred_col].to_numpy()

        records.append({
            "Year": year,
            "Model": model,
            "N": len(year_df),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0)
        })

annual = pd.DataFrame(records)
annual.to_csv(RESULT_FILE, index=False)


# =============================================================================
# 7. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(13.2, 4.6))


# =============================================================================
# 8. PLOT LINES
# =============================================================================

for model in MODELS.keys():

    subset = annual.loc[
        annual["Model"] == model
    ].sort_values("Year")

    ax.plot(
        subset["Year"],
        subset["Balanced_Accuracy"],
        color=MODEL_COLORS[model],
        linewidth=2.3,
        marker="o",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        alpha=0.97,
        label=LEGEND_LABELS[model],
        zorder=3
    )


# =============================================================================
# 9. AXES
# =============================================================================

years = np.sort(annual["Year"].unique())

# Keep aligned tick positions
tick_years = years[1::2]   # 2004, 2006, 2008, ...

ax.set_xticks(tick_years)
ax.set_xlim(years.min() - 0.25, years.max() + 0.25)

# Show tick marks, but hide text labels
ax.tick_params(
    axis="x",
    labelbottom=False,
    bottom=True,
    length=5,
    width=1.0
)

ax.set_ylabel("Balanced accuracy", labelpad=10)

ax.set_ylim(0.30, 1.03)
ax.set_yticks([0.4, 0.6, 0.8, 1.0])


# =============================================================================
# 10. GRID / SPINES
# =============================================================================

ax.grid(
    axis="y",
    color="#D3D7DB",
    linewidth=0.65,
    alpha=0.55
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.tick_params(axis="y", width=1.0, length=5)


# =============================================================================
# 11. COMPACT ONE-LINE LEGEND
# =============================================================================

legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=6,
    frameon=False,
    handlelength=1.6,
    handletextpad=0.45,
    columnspacing=1.0,
    borderaxespad=0.0
)

for line in legend.get_lines():
    line.set_linewidth(2.1)


# =============================================================================
# 12. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.55)


# =============================================================================
# 13. SAVE
# =============================================================================

tiff_file = OUTPUT_DIR / "Figure_08c_LOYO_Balanced_Accuracy_Panel.tiff"
pdf_file = OUTPUT_DIR / "Figure_08c_LOYO_Balanced_Accuracy_Panel.pdf"
svg_file = OUTPUT_DIR / "Figure_08c_LOYO_Balanced_Accuracy_Panel.svg"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.03,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03
)

fig.savefig(
    svg_file,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()
plt.close(fig)


# =============================================================================
# 14. OUTPUT SUMMARY
# =============================================================================

print("\nRefined LOYO balanced-accuracy panel saved:")
print(f"TIFF : {tiff_file}")
print(f"PDF  : {pdf_file}")
print(f"SVG  : {svg_file}")
print(f"\nAnnual performance table:\n{RESULT_FILE}")


# In[65]:


"""Monthly ecological-state transition matrix.

Uses the TRUE ecological-state sequence from saved LOYO data.

For each pair of consecutive calendar months:
    State at month t  ->  State at month t+1

Outputs:
- transition counts
- row-normalized transition probabilities
- publication-quality 3 × 3 heatmap

No model is retrained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

DATA_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "01_State_Discovery"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_09"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 28,

    "axes.labelsize": 32,

    "xtick.labelsize": 27,
    "ytick.labelsize": 27,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. LOAD TRUE ECOLOGICAL STATES
# =============================================================================

data = pd.read_csv(
    DATA_FILE
)

required = [
    "Months",
    "True_State"
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 4. CLEAN / SORT MONTHLY SERIES
# =============================================================================

data["Months"] = pd.to_datetime(
    data["Months"],
    errors="coerce"
)

data["True_State"] = pd.to_numeric(
    data["True_State"],
    errors="coerce"
)


data = data.dropna(
    subset=[
        "Months",
        "True_State"
    ]
).copy()


data["True_State"] = (
    data["True_State"]
    .astype(int)
)


data = (
    data
    .sort_values("Months")
    .drop_duplicates(
        subset="Months",
        keep="first"
    )
    .reset_index(drop=True)
)


# =============================================================================
# 5. CONSTRUCT MONTH-TO-MONTH TRANSITIONS
# =============================================================================

transition_rows = []


for i in range(
    len(data) - 1
):

    current_month = (
        data.loc[i, "Months"]
        .to_period("M")
    )

    next_month = (
        data.loc[i + 1, "Months"]
        .to_period("M")
    )


    # -------------------------------------------------------------------------
    # Only count TRUE consecutive months.
    #
    # This prevents a missing month from generating a false direct transition.
    # -------------------------------------------------------------------------

    if (
        next_month
        ==
        current_month + 1
    ):

        transition_rows.append({

            "From_State":
                int(
                    data.loc[
                        i,
                        "True_State"
                    ]
                ),

            "To_State":
                int(
                    data.loc[
                        i + 1,
                        "True_State"
                    ]
                ),

            "From_Month":
                data.loc[
                    i,
                    "Months"
                ],

            "To_Month":
                data.loc[
                    i + 1,
                    "Months"
                ]
        })


transitions = pd.DataFrame(
    transition_rows
)


if transitions.empty:

    raise ValueError(
        "No consecutive-month transitions were found."
    )


# =============================================================================
# 6. BUILD 3 × 3 TRANSITION COUNT MATRIX
# =============================================================================

STATES = [
    1,
    2,
    3
]


count_matrix = (

    pd.crosstab(

        transitions[
            "From_State"
        ],

        transitions[
            "To_State"
        ]
    )

    .reindex(
        index=STATES,
        columns=STATES,
        fill_value=0
    )
)


# =============================================================================
# 7. ROW-NORMALIZED TRANSITION PROBABILITIES
# =============================================================================

row_totals = (
    count_matrix
    .sum(axis=1)
)


prob_matrix = (
    count_matrix
    .div(
        row_totals,
        axis=0
    )
    .fillna(0)
)


# =============================================================================
# 8. PRINT RESULTS
# =============================================================================

print(
    "\nNumber of valid consecutive-month transitions:",
    len(transitions)
)


print(
    "\nTransition counts:"
)

print(
    count_matrix
)


print(
    "\nTransition probabilities:"
)

print(
    prob_matrix.round(3)
)


# =============================================================================
# 9. SAVE TABLES
# =============================================================================

count_file = (
    RESULT_DIR
    / "State_Transition_Counts.csv"
)

prob_file = (
    RESULT_DIR
    / "State_Transition_Probabilities.csv"
)


count_matrix.to_csv(
    count_file
)

prob_matrix.to_csv(
    prob_file
)


# =============================================================================
# 10. CUSTOM COLOUR MAP
# =============================================================================

transition_cmap = LinearSegmentedColormap.from_list(
    "TransitionMap",
    [
        "#F7FAFC",
        "#DDECEF",
        "#A8CDD0",
        "#69A9A6",
        "#347F7A",
        "#19524E"
    ]
)


# =============================================================================
# 11. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.2, 7.2)
)


image = ax.imshow(

    prob_matrix.to_numpy(),

    cmap=transition_cmap,

    vmin=0,
    vmax=1,

    interpolation="nearest"
)


# =============================================================================
# 12. AXES
# =============================================================================

state_labels = [
    "State 1",
    "State 2",
    "State 3"
]


ax.set_xticks(
    np.arange(3),
    state_labels
)

ax.set_yticks(
    np.arange(3),
    state_labels
)


ax.set_xlabel(
    "State at month $t+1$",
    labelpad=12
)

ax.set_ylabel(
    "State at month $t$",
    labelpad=12
)


# =============================================================================
# 13. CELL ANNOTATIONS
#
# Main number = transition probability
# Smaller number = transition count
# =============================================================================

for i in range(3):

    for j in range(3):


        probability = (
            prob_matrix.iloc[
                i,
                j
            ]
        )

        count = (
            count_matrix.iloc[
                i,
                j
            ]
        )


        text_color = (
            "white"
            if probability >= 0.50
            else "#222222"
        )


        # ---------------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------------

        ax.text(

            j,
            i - 0.08,

            f"{probability:.2f}",

            ha="center",
            va="center",

            fontsize=27,

            fontweight="bold",

            color=text_color
        )


        # ---------------------------------------------------------------------
        # Count
        # ---------------------------------------------------------------------

        ax.text(

            j,
            i + 0.20,

            f"(n={count})",

            ha="center",
            va="center",

            fontsize=19,

            fontweight="normal",

            color=text_color
        )


# =============================================================================
# 14. CELL SEPARATION
# =============================================================================

ax.set_xticks(
    np.arange(-0.5, 3, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, 3, 1),
    minor=True
)


ax.grid(

    which="minor",

    color="white",

    linewidth=2.2
)


ax.tick_params(

    which="minor",

    bottom=False,
    left=False
)


# =============================================================================
# 15. OUTER BOX
# =============================================================================

for spine in ax.spines.values():

    spine.set_visible(
        True
    )

    spine.set_linewidth(
        1.2
    )

    spine.set_color(
        "#4A4A4A"
    )


# =============================================================================
# 16. COLORBAR
# =============================================================================

colorbar = fig.colorbar(

    image,

    ax=ax,

    fraction=0.046,

    pad=0.045
)


colorbar.set_label(

    "Transition probability",

    fontsize=27,

    labelpad=12
)


colorbar.ax.tick_params(
    labelsize=22
)


# =============================================================================
# 17. NO MAIN TITLE
# =============================================================================

# Deliberately omitted.
# Axis labels already make the figure self-explanatory.


# =============================================================================
# 18. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 19. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "State_Transition_Probability_Matrix.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "State_Transition_Probability_Matrix.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "State_Transition_Probability_Matrix.svg"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(
    fig
)


# =============================================================================
# 20. SUMMARY
# =============================================================================

print(
    "\nState-transition figure saved:"
)

print(
    f"TIFF : {tiff_file}"
)

print(
    f"PDF  : {pdf_file}"
)

print(
    f"SVG  : {svg_file}"
)

print(
    f"\nCount table:\n{count_file}"
)

print(
    f"\nProbability table:\n{prob_file}"
)


# In[3]:


"""Final publication-quality CatBoost environmental interaction network.

Uses the SAVED CatBoost native interaction ranking.

No model retraining.
No interaction recalculation.

Visual encoding
---------------
Node        = environmental predictor
Node size   = total displayed interaction strength
Node colour = predictor identity
Edge width  = CatBoost pairwise interaction strength

Only predictors participating in the strongest interactions are shown.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

INTERACTION_FILE = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP_Interactions"
    / "CatBoost_Feature_Interaction_Ranking.csv"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "05_CatBoost_SHAP_Interactions"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_10"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. NETWORK SETTINGS
# =============================================================================

# Your current result looks well balanced with 12.
TOP_EDGES = 12

# Keep numerical interaction values off the main figure.
SHOW_EDGE_VALUES = False

# If changed to True, label only strongest N edges.
N_EDGE_LABELS = 5


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 30,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "mathtext.fontset": "custom",
    "mathtext.rm": "Calibri",
    "mathtext.it": "Calibri:italic",
    "mathtext.bf": "Calibri:bold",
})


# =============================================================================
# 4. PUBLICATION FEATURE LABELS
# =============================================================================

FEATURE_LABELS = {

    "SST": "SST",
    "SSS": "SSS",
    "MLD": "MLD",
    "PAR": "PAR",
    "SSH": "SSH",

    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",
    "SPCo2": r"$p\mathrm{CO}_2$",

    "PDO": "PDO",
    "NINO_3.4": "Niño 3.4",
    "WPI": "WPI",

    "MHW_MeanInt": r"$\mathrm{MHW}_{\mathrm{mean}}$",
    "MHW_MaxInt": r"$\mathrm{MHW}_{\mathrm{max}}$",
    "MHW_CumInt": r"$\mathrm{MHW}_{\mathrm{cum}}$",
}


# =============================================================================
# 5. INDIVIDUAL FEATURE COLOURS
#
# Multicolour style consistent with the SHAP importance bar plot.
# Colours are deliberately muted rather than highly saturated.
# =============================================================================

FEATURE_COLORS = {

    "MLD":        "#4C9499",   # teal
    "SST":        "#C3A12A",   # mustard
    "SSS":        "#82C66A",   # light green
    "PAR":        "#F08F99",   # soft pink

    "SPCo2":      "#83BBB4",   # pale teal
    "NINO_3.4":   "#B4AAA5",   # warm grey
    "SSH":        "#A87C65",   # brown
    "NO3":        "#E9C83A",   # golden yellow

    "PDO":        "#A46F9D",   # mauve
    "PO4":        "#F28E2B",   # orange
    "WPI":        "#59A14F",   # green

    "MHW_CumInt": "#76B7B2",
    "MHW_MaxInt": "#E15759",
    "MHW_MeanInt":"#4E79A7",
}


# =============================================================================
# 6. LOAD SAVED INTERACTION TABLE
# =============================================================================

interactions = pd.read_csv(
    INTERACTION_FILE
)


required = [
    "Feature_1",
    "Feature_2",
    "Interaction_Strength"
]


missing = [
    column
    for column in required
    if column not in interactions.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(interactions.columns)}"
    )


interactions["Interaction_Strength"] = pd.to_numeric(
    interactions["Interaction_Strength"],
    errors="coerce"
)


interactions = (
    interactions
    .dropna(
        subset=[
            "Feature_1",
            "Feature_2",
            "Interaction_Strength"
        ]
    )
    .sort_values(
        "Interaction_Strength",
        ascending=False
    )
    .reset_index(drop=True)
)


# =============================================================================
# 7. SELECT TOP INTERACTIONS
# =============================================================================

top = (
    interactions
    .head(TOP_EDGES)
    .copy()
)


print(
    f"\nTop {TOP_EDGES} interactions used in final network:"
)

print(
    "=" * 72
)

print(
    top[
        [
            "Feature_1",
            "Feature_2",
            "Interaction_Strength"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 8. BUILD NETWORK
# =============================================================================

G = nx.Graph()


for _, row in top.iterrows():

    G.add_edge(

        row["Feature_1"],

        row["Feature_2"],

        weight=float(
            row["Interaction_Strength"]
        )
    )


# =============================================================================
# 9. CALCULATE INTERACTION-HUB STRENGTH
#
# Larger node = greater total interaction strength among displayed edges.
# =============================================================================

node_strength = {}


for node in G.nodes():

    node_strength[node] = sum(

        G[node][neighbor]["weight"]

        for neighbor in G.neighbors(node)
    )


node_strength_values = np.array(
    list(node_strength.values()),
    dtype=float
)


strength_min = node_strength_values.min()
strength_max = node_strength_values.max()


# =============================================================================
# 10. SCALE NODE SIZE
# =============================================================================

node_sizes = {}


for node, strength in node_strength.items():


    if np.isclose(
        strength_min,
        strength_max
    ):

        scaled = 0.5


    else:

        scaled = (

            strength - strength_min

        ) / (

            strength_max - strength_min

        )


    node_sizes[node] = (

        1250

        +

        2500 * scaled
    )


# =============================================================================
# 11. PREPARE NETWORK LAYOUT
#
# Strong interactions are assigned shorter network distance.
# =============================================================================

max_weight = max(

    nx.get_edge_attributes(
        G,
        "weight"
    ).values()
)


for u, v, edge_data in G.edges(
    data=True
):

    edge_data["distance"] = (

        max_weight

        /

        edge_data["weight"]
    )


# =============================================================================
# 12. KAMADA-KAWAI LAYOUT
#
# This is the layout that produced the clean result you liked.
# =============================================================================

pos = nx.kamada_kawai_layout(

    G,

    weight="distance",

    scale=1.0
)


# Slightly open the network
for node in pos:

    pos[node] = (
        pos[node] * 1.25
    )


# =============================================================================
# 13. EDGE WIDTHS
# =============================================================================

edge_strengths = np.array(

    [
        G[u][v]["weight"]

        for u, v in G.edges()
    ],

    dtype=float
)


edge_min = edge_strengths.min()
edge_max = edge_strengths.max()


if np.isclose(
    edge_min,
    edge_max
):

    normalized_edges = np.ones_like(
        edge_strengths
    )


else:

    normalized_edges = (

        edge_strengths - edge_min

    ) / (

        edge_max - edge_min

    )


edge_widths = (

    1.4

    +

    5.8 * normalized_edges
)


edge_alphas = (

    0.28

    +

    0.52 * normalized_edges
)


# =============================================================================
# 14. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(

    figsize=(11.8, 9.2)
)


# Pure white background for manuscript use
fig.patch.set_facecolor(
    "white"
)

ax.set_facecolor(
    "white"
)


# =============================================================================
# 15. DRAW EDGES
# =============================================================================

for (
    edge,
    width,
    alpha
) in zip(

    G.edges(),

    edge_widths,

    edge_alphas
):


    nx.draw_networkx_edges(

        G,

        pos,

        edgelist=[
            edge
        ],

        width=float(
            width
        ),

        edge_color="#839097",

        alpha=float(
            alpha
        ),

        ax=ax
    )


# =============================================================================
# 16. DRAW INDIVIDUALLY COLOURED NODES
# =============================================================================

nodes = list(
    G.nodes()
)


for node in nodes:


    nx.draw_networkx_nodes(

        G,

        pos,

        nodelist=[
            node
        ],

        node_size=[
            node_sizes[node]
        ],

        node_color=[
            FEATURE_COLORS.get(
                node,
                "#8C8C8C"
            )
        ],

        edgecolors="white",

        linewidths=2.5,

        alpha=0.97,

        ax=ax
    )


# =============================================================================
# 17. CALCULATE LABEL POSITIONS
#
# Labels are pushed OUTWARD from network centre.
# =============================================================================

all_positions = np.array(
    list(pos.values())
)


centre = np.mean(
    all_positions,
    axis=0
)


label_pos = {}


for node, xy in pos.items():


    x, y = xy


    direction = (

        np.array(
            [x, y]
        )

        -

        centre
    )


    distance = np.linalg.norm(
        direction
    )


    if distance < 1e-9:

        direction = np.array(
            [0.0, 1.0]
        )

    else:

        direction = (

            direction

            /

            distance
        )


    # -------------------------------------------------------------
    # Stronger interaction hubs receive slightly larger offset.
    # -------------------------------------------------------------

    if np.isclose(
        strength_min,
        strength_max
    ):

        relative_strength = 0.5

    else:

        relative_strength = (

            node_strength[node]

            -

            strength_min

        ) / (

            strength_max

            -

            strength_min
        )


    offset = (

        0.090

        +

        0.045 * relative_strength
    )


    label_pos[node] = (

        x
        +
        direction[0] * offset,

        y
        +
        direction[1] * offset
    )


# =============================================================================
# 18. DRAW FEATURE LABELS
# =============================================================================

for node in nodes:


    x_node, y_node = pos[node]

    x_text, y_text = label_pos[node]


    # -------------------------------------------------------------
    # Horizontal alignment
    # -------------------------------------------------------------

    if x_text > x_node + 0.015:

        ha = "left"

    elif x_text < x_node - 0.015:

        ha = "right"

    else:

        ha = "center"


    # -------------------------------------------------------------
    # Vertical alignment
    # -------------------------------------------------------------

    if y_text > y_node + 0.015:

        va = "bottom"

    elif y_text < y_node - 0.015:

        va = "top"

    else:

        va = "center"


    ax.text(

        x_text,

        y_text,

        FEATURE_LABELS.get(
            node,
            node
        ),

        fontsize=27,

        fontfamily="Calibri",

        fontweight="normal",

        color="#202020",

        ha=ha,

        va=va,

        zorder=7
    )


# =============================================================================
# 19. OPTIONAL EDGE VALUES
#
# Keep False for the clean manuscript version.
# =============================================================================

if SHOW_EDGE_VALUES:


    strongest_for_labels = (

        top

        .head(
            N_EDGE_LABELS
        )
    )


    edge_labels = {}


    for _, row in strongest_for_labels.iterrows():


        edge_labels[
            (
                row["Feature_1"],
                row["Feature_2"]
            )
        ] = (

            f"{row['Interaction_Strength']:.2f}"
        )


    nx.draw_networkx_edge_labels(

        G,

        pos,

        edge_labels=edge_labels,

        font_size=17,

        font_family="Calibri",

        font_color="#505050",

        rotate=False,

        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.15
        },

        ax=ax
    )


# =============================================================================
# 20. REMOVE NORMAL AXES
# =============================================================================

ax.set_axis_off()


# =============================================================================
# 21. FIGURE MARGINS
# =============================================================================

x_values = np.array(
    [
        position[0]
        for position in pos.values()
    ]
)


y_values = np.array(
    [
        position[1]
        for position in pos.values()
    ]
)


x_span = (
    x_values.max()
    -
    x_values.min()
)


y_span = (
    y_values.max()
    -
    y_values.min()
)


ax.set_xlim(

    x_values.min()
    -
    0.25 * x_span,

    x_values.max()
    +
    0.25 * x_span
)


ax.set_ylim(

    y_values.min()
    -
    0.25 * y_span,

    y_values.max()
    +
    0.25 * y_span
)


# =============================================================================
# 22. NO MAIN TITLE
# =============================================================================

# Intentionally omitted.


# =============================================================================
# 23. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.55
)


# =============================================================================
# 24. SAVE FIGURE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "CatBoost_Environmental_Interaction_Network_Final.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "CatBoost_Environmental_Interaction_Network_Final.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "CatBoost_Environmental_Interaction_Network_Final.svg"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.05,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.05
)


fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.05
)


plt.show()

plt.close(
    fig
)


# =============================================================================
# 25. SAVE NODE / HUB SUMMARY
# =============================================================================

node_summary = pd.DataFrame(

    [
        {

            "Feature":
                node,

            "Displayed_Total_Interaction_Strength":
                node_strength[node],

            "Displayed_Degree":
                G.degree(node)

        }

        for node in nodes
    ]
)


node_summary = (

    node_summary

    .sort_values(

        "Displayed_Total_Interaction_Strength",

        ascending=False
    )

    .reset_index(
        drop=True
    )
)


node_summary["Hub_Rank"] = (

    np.arange(
        1,
        len(node_summary) + 1
    )
)


node_file = (

    RESULT_DIR
    / "Final_Interaction_Network_Hub_Strength.csv"
)


node_summary.to_csv(

    node_file,

    index=False
)


# =============================================================================
# 26. PRINT RESULTS
# =============================================================================

print(
    "\nInteraction-network hub ranking:"
)

print(
    "=" * 72
)

print(

    node_summary[
        [
            "Hub_Rank",
            "Feature",
            "Displayed_Degree",
            "Displayed_Total_Interaction_Strength"
        ]
    ]

    .round(3)

    .to_string(
        index=False
    )
)


print(
    "\nFinal interaction network saved:"
)

print(
    f"TIFF : {tiff_file}"
)

print(
    f"PDF  : {pdf_file}"
)

print(
    f"SVG  : {svg_file}"
)

print(
    f"\nHub table:\n{node_file}"
)


# In[4]:


"""Publication-quality violin + box plot of LOYO balanced accuracy.

Each distribution contains the balanced accuracy obtained for
each held-out year.

Models:
- CatBoost
- XGBoost
- HistGradientBoosting
- Soft Voting
- TCN
- CNN-LSTM

The figure combines:
1. violin distribution
2. boxplot
3. individual held-out-year observations

No model is retrained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

RESULT_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "LOYO_Balanced_Accuracy_Distribution.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_08"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. MODEL SETTINGS
# =============================================================================

MODELS = {
    "CatBoost":
        "CatBoost_Prediction",

    "XGBoost":
        "XGBoost_Prediction",

    "HistGradientBoosting":
        "HistGradientBoosting_Prediction",

    "Soft Voting":
        "Equal_Soft_Voting_Prediction",

    "TCN":
        "TCN_Prediction",

    "CNN-LSTM":
        "CNN_LSTM_Prediction"
}


# Shorter publication labels
DISPLAY_LABELS = {
    "CatBoost": "CatBoost",
    "XGBoost": "XGB",
    "HistGradientBoosting": "HGB",
    "Soft Voting": "Soft Vote",
    "TCN": "TCN",
    "CNN-LSTM": "CNN-LSTM"
}


# =============================================================================
# 3. MODEL COLOURS
# =============================================================================

MODEL_COLORS = {
    "CatBoost": "#4E79A7",
    "XGBoost": "#F28E2B",
    "HistGradientBoosting": "#59A14F",
    "Soft Voting": "#8064A2",
    "TCN": "#C65F54",
    "CNN-LSTM": "#4E8C8D"
}


# =============================================================================
# 4. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 26,

    "axes.labelsize": 30,

    "xtick.labelsize": 23,
    "ytick.labelsize": 24,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 5. LOAD LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


YEAR_COL = "Test_Year"
TRUE_COL = "True_State"


required = [
    YEAR_COL,
    TRUE_COL,
    *MODELS.values()
]


missing = [
    column
    for column in required
    if column not in data.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 6. CLEAN DATA
# =============================================================================

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="coerce"
)

data[TRUE_COL] = pd.to_numeric(
    data[TRUE_COL],
    errors="coerce"
)


for pred_col in MODELS.values():

    data[pred_col] = pd.to_numeric(
        data[pred_col],
        errors="coerce"
    )


data = data.dropna(
    subset=[
        YEAR_COL,
        TRUE_COL,
        *MODELS.values()
    ]
).copy()


data[YEAR_COL] = (
    data[YEAR_COL]
    .astype(int)
)

data[TRUE_COL] = (
    data[TRUE_COL]
    .astype(int)
)


for pred_col in MODELS.values():

    data[pred_col] = (
        data[pred_col]
        .astype(int)
    )


# =============================================================================
# 7. CALCULATE BALANCED ACCURACY FOR EACH HELD-OUT YEAR
# =============================================================================

records = []


for year, year_df in data.groupby(
    YEAR_COL,
    sort=True
):

    y_true = (
        year_df[TRUE_COL]
        .to_numpy()
    )


    for model, pred_col in MODELS.items():

        y_pred = (
            year_df[pred_col]
            .to_numpy()
        )


        score = balanced_accuracy_score(
            y_true,
            y_pred
        )


        records.append({

            "Year": year,

            "Model": model,

            "Balanced_Accuracy": score
        })


performance = pd.DataFrame(
    records
)


performance.to_csv(
    RESULT_FILE,
    index=False
)


# =============================================================================
# 8. PRINT SUMMARY
# =============================================================================

summary = (

    performance

    .groupby("Model")[
        "Balanced_Accuracy"
    ]

    .agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max"
        ]
    )
)


# Keep desired model order
summary = summary.reindex(
    MODELS.keys()
)


print(
    "\nLOYO balanced-accuracy distributions"
)

print(
    "=" * 70
)

print(
    summary.round(3)
)


# =============================================================================
# 9. PREPARE DATA FOR PLOTTING
# =============================================================================

model_order = list(
    MODELS.keys()
)


violin_data = [

    performance.loc[
        performance["Model"] == model,
        "Balanced_Accuracy"
    ].to_numpy()

    for model in model_order
]


positions = np.arange(
    1,
    len(model_order) + 1
)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12.8, 7.2)
)


# =============================================================================
# 11. VIOLIN PLOTS
# =============================================================================

violins = ax.violinplot(

    violin_data,

    positions=positions,

    widths=0.82,

    showmeans=False,

    showmedians=False,

    showextrema=False,

    bw_method=0.35
)


# Individual colours
for body, model in zip(
    violins["bodies"],
    model_order
):

    body.set_facecolor(
        MODEL_COLORS[model]
    )

    body.set_edgecolor(
        MODEL_COLORS[model]
    )

    body.set_alpha(
        0.72
    )

    body.set_linewidth(
        1.2
    )


# =============================================================================
# 12. INTERNAL BOXPLOTS
# =============================================================================

box = ax.boxplot(

    violin_data,

    positions=positions,

    widths=0.22,

    patch_artist=True,

    showfliers=False,

    whis=1.5,

    medianprops={
        "color": "black",
        "linewidth": 2.0
    },

    whiskerprops={
        "color": "#303030",
        "linewidth": 1.3
    },

    capprops={
        "color": "#303030",
        "linewidth": 1.3
    }
)


# Box colours
for patch, model in zip(
    box["boxes"],
    model_order
):

    patch.set_facecolor(
        MODEL_COLORS[model]
    )

    patch.set_alpha(
        0.88
    )

    patch.set_edgecolor(
        "black"
    )

    patch.set_linewidth(
        1.2
    )


# =============================================================================
# 13. INDIVIDUAL YEARLY OBSERVATIONS
#
# Deterministic jitter so figure is reproducible.
# =============================================================================

rng = np.random.default_rng(
    42
)


for x_position, model, scores in zip(
    positions,
    model_order,
    violin_data
):

    jitter = rng.normal(
        loc=0,
        scale=0.045,
        size=len(scores)
    )


    ax.scatter(

        np.full(
            len(scores),
            x_position
        )
        +
        jitter,

        scores,

        s=28,

        facecolor="#202020",

        edgecolor="white",

        linewidth=0.45,

        alpha=0.72,

        zorder=5
    )


# =============================================================================
# 14. OPTIONAL MEAN MARKER
#
# Small white diamond = model mean
# =============================================================================

for x_position, model, scores in zip(
    positions,
    model_order,
    violin_data
):

    mean_score = np.mean(
        scores
    )


    ax.scatter(

        x_position,

        mean_score,

        marker="D",

        s=58,

        facecolor="white",

        edgecolor="black",

        linewidth=1.25,

        zorder=7
    )


# =============================================================================
# 15. X AXIS
# =============================================================================

ax.set_xticks(
    positions
)


ax.set_xticklabels(

    [
        DISPLAY_LABELS[model]
        for model in model_order
    ]
)


ax.set_xlabel(
    ""
)


# =============================================================================
# 16. Y AXIS
# =============================================================================

ax.set_ylabel(
    "Balanced accuracy",
    labelpad=12
)


# Our observed annual values reach roughly 0.37–1.00.
# This range gives some breathing space while retaining the full distributions.

ax.set_ylim(
    0.30,
    1.03
)


ax.set_yticks(
    np.arange(
        0.3,
        1.01,
        0.1
    )
)


# =============================================================================
# 17. GRID
# =============================================================================

ax.grid(

    axis="y",

    color="#D3D7DB",

    linewidth=0.7,

    alpha=0.60
)


ax.set_axisbelow(
    True
)


# =============================================================================
# 18. CLEAN SPINES
# =============================================================================

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)


ax.spines["left"].set_linewidth(
    1.25
)

ax.spines["bottom"].set_linewidth(
    1.25
)


ax.tick_params(

    axis="both",

    width=1.1,

    length=5
)


# =============================================================================
# 19. NO MAIN TITLE
# =============================================================================

# Intentionally omitted for easy panel assembly.


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.8
)


# =============================================================================
# 21. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Box.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Box.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Box.svg"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(
    fig
)


# =============================================================================
# 22. OUTPUT
# =============================================================================

print(
    "\nBalanced-accuracy violin plot saved:"
)

print(
    f"TIFF : {tiff_file}"
)

print(
    f"PDF  : {pdf_file}"
)

print(
    f"SVG  : {svg_file}"
)

print(
    f"\nYear-level scores:\n{RESULT_FILE}"
)


# In[5]:


"""Final polished violin + box plot of LOYO balanced accuracy.

Each model distribution contains balanced accuracy obtained
from individual held-out years.

Visual elements
---------------
Violin        : year-to-year performance distribution
Box           : interquartile range
Black line    : median
Dark points   : individual held-out years
White diamond : mean balanced accuracy

No model is retrained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

PREDICTION_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

RESULT_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "LOYO_Balanced_Accuracy_Distribution.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_08"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. MODEL SETTINGS
# =============================================================================

MODELS = {
    "CatBoost": "CatBoost_Prediction",
    "XGBoost": "XGBoost_Prediction",
    "HistGradientBoosting": "HistGradientBoosting_Prediction",
    "Soft Voting": "Equal_Soft_Voting_Prediction",
    "TCN": "TCN_Prediction",
    "CNN-LSTM": "CNN_LSTM_Prediction"
}


# Short publication labels
DISPLAY_LABELS = {
    "CatBoost": "CatBoost",
    "XGBoost": "XGB",
    "HistGradientBoosting": "HGB",
    "Soft Voting": "Soft Vote",
    "TCN": "TCN",
    "CNN-LSTM": "CNN-LSTM"
}


# =============================================================================
# 3. MODEL COLOURS
# =============================================================================

MODEL_COLORS = {
    "CatBoost": "#4E79A7",
    "XGBoost": "#F28E2B",
    "HistGradientBoosting": "#59A14F",
    "Soft Voting": "#8064A2",
    "TCN": "#C65F54",
    "CNN-LSTM": "#4E8C8D"
}


# =============================================================================
# 4. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 27,

    "axes.labelsize": 31,

    "xtick.labelsize": 25,
    "ytick.labelsize": 25,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 5. LOAD LOYO PREDICTIONS
# =============================================================================

data = pd.read_csv(
    PREDICTION_FILE
)


YEAR_COL = "Test_Year"
TRUE_COL = "True_State"


required = [
    YEAR_COL,
    TRUE_COL,
    *MODELS.values()
]


missing = [
    column
    for column in required
    if column not in data.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 6. CLEAN DATA
# =============================================================================

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="coerce"
)

data[TRUE_COL] = pd.to_numeric(
    data[TRUE_COL],
    errors="coerce"
)


for pred_col in MODELS.values():

    data[pred_col] = pd.to_numeric(
        data[pred_col],
        errors="coerce"
    )


data = data.dropna(
    subset=[
        YEAR_COL,
        TRUE_COL,
        *MODELS.values()
    ]
).copy()


data[YEAR_COL] = data[YEAR_COL].astype(int)
data[TRUE_COL] = data[TRUE_COL].astype(int)


for pred_col in MODELS.values():

    data[pred_col] = data[pred_col].astype(int)


# =============================================================================
# 7. CALCULATE YEAR-SPECIFIC BALANCED ACCURACY
# =============================================================================

records = []


for year, year_df in data.groupby(
    YEAR_COL,
    sort=True
):

    y_true = year_df[TRUE_COL].to_numpy()


    for model, pred_col in MODELS.items():

        y_pred = year_df[pred_col].to_numpy()


        score = balanced_accuracy_score(
            y_true,
            y_pred
        )


        records.append({

            "Year": year,

            "Model": model,

            "Balanced_Accuracy": score
        })


performance = pd.DataFrame(
    records
)


performance.to_csv(
    RESULT_FILE,
    index=False
)


# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================

model_order = list(
    MODELS.keys()
)


summary = (

    performance

    .groupby("Model")[
        "Balanced_Accuracy"
    ]

    .agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max"
        ]
    )

    .reindex(
        model_order
    )
)


print(
    "\nLOYO balanced-accuracy distributions"
)

print(
    "=" * 70
)

print(
    summary.round(3)
)


# =============================================================================
# 9. PREPARE PLOTTING DATA
# =============================================================================

violin_data = [

    performance.loc[
        performance["Model"] == model,
        "Balanced_Accuracy"
    ].to_numpy()

    for model in model_order
]


positions = np.arange(
    1,
    len(model_order) + 1
)


# =============================================================================
# 10. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12.8, 7.2)
)


# =============================================================================
# 11. VIOLIN DISTRIBUTIONS
#
# Slightly narrower and lighter than previous version.
# =============================================================================

violins = ax.violinplot(

    violin_data,

    positions=positions,

    widths=0.70,

    showmeans=False,

    showmedians=False,

    showextrema=False,

    bw_method=0.35
)


for body, model in zip(
    violins["bodies"],
    model_order
):

    body.set_facecolor(
        MODEL_COLORS[model]
    )

    body.set_edgecolor(
        MODEL_COLORS[model]
    )

    body.set_alpha(
        0.60
    )

    body.set_linewidth(
        1.15
    )


# =============================================================================
# 12. INTERNAL BOXPLOTS
# =============================================================================

box = ax.boxplot(

    violin_data,

    positions=positions,

    widths=0.18,

    patch_artist=True,

    showfliers=False,

    whis=1.5,

    medianprops={
        "color": "#111111",
        "linewidth": 2.4
    },

    whiskerprops={
        "color": "#303030",
        "linewidth": 1.25
    },

    capprops={
        "color": "#303030",
        "linewidth": 1.25
    }
)


for patch, model in zip(
    box["boxes"],
    model_order
):

    patch.set_facecolor(
        MODEL_COLORS[model]
    )

    patch.set_alpha(
        0.80
    )

    patch.set_edgecolor(
        "#202020"
    )

    patch.set_linewidth(
        1.25
    )


# =============================================================================
# 13. INDIVIDUAL HELD-OUT YEARS
#
# Smaller and lighter so they do not dominate the distribution.
# =============================================================================

rng = np.random.default_rng(
    42
)


for x_position, scores in zip(
    positions,
    violin_data
):

    jitter = rng.normal(
        loc=0,
        scale=0.035,
        size=len(scores)
    )


    ax.scatter(

        np.full(
            len(scores),
            x_position
        )
        +
        jitter,

        scores,

        s=22,

        facecolor="#303030",

        edgecolor="white",

        linewidth=0.35,

        alpha=0.62,

        zorder=5
    )


# =============================================================================
# 14. MEAN MARKER
#
# White diamond = mean.
# =============================================================================

for x_position, scores in zip(
    positions,
    violin_data
):

    mean_score = np.mean(
        scores
    )


    ax.scatter(

        x_position,

        mean_score,

        marker="D",

        s=55,

        facecolor="white",

        edgecolor="#202020",

        linewidth=1.35,

        zorder=7
    )


# =============================================================================
# 15. X AXIS
# =============================================================================

ax.set_xticks(
    positions
)


ax.set_xticklabels(

    [
        DISPLAY_LABELS[model]
        for model in model_order
    ]
)


ax.set_xlabel(
    ""
)


# =============================================================================
# 16. Y AXIS
# =============================================================================

ax.set_ylabel(
    "Balanced accuracy",
    labelpad=12
)


ax.set_ylim(
    0.30,
    1.03
)


ax.set_yticks(
    np.arange(
        0.3,
        1.01,
        0.1
    )
)


# =============================================================================
# 17. GRID
#
# Only subtle horizontal grid lines.
# =============================================================================

ax.grid(

    axis="y",

    color="#D7DADD",

    linewidth=0.65,

    alpha=0.55
)


ax.set_axisbelow(
    True
)


# =============================================================================
# 18. CLEAN AXES
# =============================================================================

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)


ax.spines["left"].set_linewidth(
    1.25
)

ax.spines["bottom"].set_linewidth(
    1.25
)


ax.spines["left"].set_color(
    "#202020"
)

ax.spines["bottom"].set_color(
    "#202020"
)


ax.tick_params(

    axis="both",

    width=1.15,

    length=5,

    color="#202020"
)


# =============================================================================
# 19. NO TITLE / NO LEGEND
# =============================================================================

# No title because this will be used as a manuscript panel.
#
# No colour legend is required because the model names are directly
# shown beneath their corresponding distributions.


# =============================================================================
# 20. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.65
)


# =============================================================================
# 21. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Final.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Final.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "LOYO_Balanced_Accuracy_Violin_Final.svg"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(
    fig
)


# =============================================================================
# 22. OUTPUT SUMMARY
# =============================================================================

print(
    "\nFinal polished balanced-accuracy violin figure saved:"
)

print(
    f"TIFF : {tiff_file}"
)

print(
    f"PDF  : {pdf_file}"
)

print(
    f"SVG  : {svg_file}"
)

print(
    f"\nYear-level scores:\n{RESULT_FILE}"
)


# In[6]:


"""LOYO accuracy distributions across held-out years
as a polished violin + box plot."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PRED_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 22,

    "axes.labelsize": 30,
    "axes.titlesize": 28,

    "xtick.labelsize": 22,
    "ytick.labelsize": 22,

    "legend.fontsize": 18,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. SETTINGS
# =============================================================================

TRUE_COL = "True_State"
YEAR_COL = "Test_Year"

MODEL_INFO = [
    ("CatBoost",              "CatBoost_Prediction",              "CatBoost",  "#4E79A7"),
    ("XGBoost",               "XGBoost_Prediction",               "XGB",       "#F28E2B"),
    ("HistGradientBoosting",  "HistGradientBoosting_Prediction",  "HGB",       "#59A14F"),
    ("Equal Soft Voting",     "Equal_Soft_Voting_Prediction",     "Soft Vote", "#9C6FB6"),
    ("TCN",                   "TCN_Prediction",                   "TCN",       "#D37267"),
    ("CNN-LSTM",              "CNN_LSTM_Prediction",              "CNN-LSTM",  "#5B9EA0"),
]


# =============================================================================
# 4. LOAD DATA
# =============================================================================

data = pd.read_csv(PRED_FILE)

required_base = [TRUE_COL, YEAR_COL]
missing_base = [col for col in required_base if col not in data.columns]

if missing_base:
    raise ValueError(
        f"Missing required base columns: {missing_base}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )

prediction_cols = [item[1] for item in MODEL_INFO]
missing_preds = [col for col in prediction_cols if col not in data.columns]

if missing_preds:
    raise ValueError(
        f"Missing required prediction columns: {missing_preds}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. CLEAN DATA
# =============================================================================

data = data.copy()
data = data.dropna(subset=[TRUE_COL, YEAR_COL] + prediction_cols)

data[TRUE_COL] = data[TRUE_COL].astype(int)
data[YEAR_COL] = data[YEAR_COL].astype(int)

for _, pred_col, _, _ in MODEL_INFO:
    data[pred_col] = data[pred_col].astype(int)


# =============================================================================
# 6. COMPUTE LOYO ACCURACY FOR EACH HELD-OUT YEAR
# =============================================================================

records = []

for model_name, pred_col, short_label, color in MODEL_INFO:

    for year, group in data.groupby(YEAR_COL):

        y_true = group[TRUE_COL].to_numpy()
        y_pred = group[pred_col].to_numpy()

        accuracy = np.mean(y_true == y_pred)

        records.append({
            "Model": model_name,
            "Short_Label": short_label,
            "Color": color,
            "Year": int(year),
            "Accuracy": float(accuracy)
        })

metrics_df = pd.DataFrame(records)


# =============================================================================
# 7. SUMMARY TABLE
# =============================================================================

summary = (
    metrics_df
    .groupby("Model")["Accuracy"]
    .agg(["mean", "std", "median", "min", "max"])
    .reindex([item[0] for item in MODEL_INFO])
)

print("\nLOYO accuracy distributions")
print("=" * 70)
print(summary.round(3))


# =============================================================================
# 8. PREPARE PLOTTING DATA
# =============================================================================

plot_arrays = []
x_labels = []
colors = []

for model_name, pred_col, short_label, color in MODEL_INFO:

    arr = (
        metrics_df.loc[metrics_df["Model"] == model_name, "Accuracy"]
        .to_numpy()
    )

    plot_arrays.append(arr)
    x_labels.append(short_label)
    colors.append(color)


# =============================================================================
# 9. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(12.8, 9.6))

positions = np.arange(1, len(plot_arrays) + 1)


# =============================================================================
# 10. VIOLIN PLOTS
# =============================================================================

violin = ax.violinplot(
    plot_arrays,
    positions=positions,
    widths=0.82,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, color in zip(violin["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.68)
    body.set_linewidth(1.2)


# =============================================================================
# 11. BOX PLOTS
# =============================================================================

box = ax.boxplot(
    plot_arrays,
    positions=positions,
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=2.0),
    whiskerprops=dict(color="#333333", linewidth=1.4),
    capprops=dict(color="#333333", linewidth=1.4),
    boxprops=dict(edgecolor="#222222", linewidth=1.4)
)

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)


# =============================================================================
# 12. JITTERED POINTS + MEAN MARKERS
# =============================================================================

rng = np.random.default_rng(42)

for i, (arr, color) in enumerate(zip(plot_arrays, colors), start=1):

    x_jitter = rng.normal(loc=i, scale=0.042, size=len(arr))

    ax.scatter(
        x_jitter,
        arr,
        s=24,
        color="#404040",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        zorder=3
    )

    ax.scatter(
        i,
        arr.mean(),
        marker="D",
        s=58,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.2,
        zorder=4
    )


# =============================================================================
# 13. AXES / LABELS
# =============================================================================

ax.set_xticks(positions)
ax.set_xticklabels(x_labels)

ax.set_ylabel("Accuracy", labelpad=10)
ax.set_xlabel("Model", labelpad=10)

ax.set_ylim(0.30, 1.02)
ax.set_yticks(np.arange(0.3, 1.01, 0.1))


# =============================================================================
# 14. GRID / SPINES
# =============================================================================

ax.grid(
    axis="y",
    color="#D0D5DA",
    linewidth=0.7,
    alpha=0.75
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)


# =============================================================================
# 15. OPTIONAL SMALL ANNOTATION
# =============================================================================

ax.text(
    0.015, 0.985,
    "LOYO per-year accuracy distributions",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=18,
    color="#4A4A4A"
)


# =============================================================================
# 16. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.9)


# =============================================================================
# 17. SAVE
# =============================================================================

tiff_file = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.tiff"
pdf_file  = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.pdf"
svg_file  = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.svg"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

fig.savefig(
    svg_file,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 18. OUTPUT SUMMARY
# =============================================================================

print("\nAccuracy violin + box plot saved:")
print(f"TIFF : {tiff_file}")
print(f"PDF  : {pdf_file}")
print(f"SVG  : {svg_file}")


# In[7]:


"""LOYO accuracy distributions across held-out years
as a polished violin + box plot."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PRED_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 24,

    "axes.labelsize": 34,
    "axes.titlesize": 30,

    "xtick.labelsize": 28,
    "ytick.labelsize": 28,

    "legend.fontsize": 20,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. SETTINGS
# =============================================================================

TRUE_COL = "True_State"
YEAR_COL = "Test_Year"

MODEL_INFO = [
    ("CatBoost",              "CatBoost_Prediction",              "CatBoost",  "#4E79A7"),
    ("XGBoost",               "XGBoost_Prediction",               "XGB",       "#F28E2B"),
    ("HistGradientBoosting",  "HistGradientBoosting_Prediction",  "HGB",       "#59A14F"),
    ("Equal Soft Voting",     "Equal_Soft_Voting_Prediction",     "Soft Vote", "#9C6FB6"),
    ("TCN",                   "TCN_Prediction",                   "TCN",       "#D37267"),
    ("CNN-LSTM",              "CNN_LSTM_Prediction",              "CNN-LSTM",  "#5B9EA0"),
]


# =============================================================================
# 4. LOAD DATA
# =============================================================================

data = pd.read_csv(PRED_FILE)

required_base = [TRUE_COL, YEAR_COL]
missing_base = [col for col in required_base if col not in data.columns]

if missing_base:
    raise ValueError(
        f"Missing required base columns: {missing_base}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )

prediction_cols = [item[1] for item in MODEL_INFO]
missing_preds = [col for col in prediction_cols if col not in data.columns]

if missing_preds:
    raise ValueError(
        f"Missing required prediction columns: {missing_preds}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. CLEAN DATA
# =============================================================================

data = data.copy()
data = data.dropna(subset=[TRUE_COL, YEAR_COL] + prediction_cols)

data[TRUE_COL] = data[TRUE_COL].astype(int)
data[YEAR_COL] = data[YEAR_COL].astype(int)

for _, pred_col, _, _ in MODEL_INFO:
    data[pred_col] = data[pred_col].astype(int)


# =============================================================================
# 6. COMPUTE LOYO ACCURACY FOR EACH HELD-OUT YEAR
# =============================================================================

records = []

for model_name, pred_col, short_label, color in MODEL_INFO:

    for year, group in data.groupby(YEAR_COL):

        y_true = group[TRUE_COL].to_numpy()
        y_pred = group[pred_col].to_numpy()

        accuracy = np.mean(y_true == y_pred)

        records.append({
            "Model": model_name,
            "Short_Label": short_label,
            "Color": color,
            "Year": int(year),
            "Accuracy": float(accuracy)
        })

metrics_df = pd.DataFrame(records)


# =============================================================================
# 7. SUMMARY TABLE
# =============================================================================

summary = (
    metrics_df
    .groupby("Model")["Accuracy"]
    .agg(["mean", "std", "median", "min", "max"])
    .reindex([item[0] for item in MODEL_INFO])
)

print("\nLOYO accuracy distributions")
print("=" * 70)
print(summary.round(3))


# =============================================================================
# 8. PREPARE PLOTTING DATA
# =============================================================================

plot_arrays = []
x_labels = []
colors = []

for model_name, pred_col, short_label, color in MODEL_INFO:

    arr = (
        metrics_df.loc[metrics_df["Model"] == model_name, "Accuracy"]
        .to_numpy()
    )

    plot_arrays.append(arr)
    x_labels.append(short_label)
    colors.append(color)


# =============================================================================
# 9. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(13.0, 10.0))

positions = np.arange(1, len(plot_arrays) + 1)


# =============================================================================
# 10. VIOLIN PLOTS
# =============================================================================

violin = ax.violinplot(
    plot_arrays,
    positions=positions,
    widths=0.82,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, color in zip(violin["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.68)
    body.set_linewidth(1.3)


# =============================================================================
# 11. BOX PLOTS
# =============================================================================

box = ax.boxplot(
    plot_arrays,
    positions=positions,
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=2.1),
    whiskerprops=dict(color="#333333", linewidth=1.5),
    capprops=dict(color="#333333", linewidth=1.5),
    boxprops=dict(edgecolor="#222222", linewidth=1.5)
)

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)


# =============================================================================
# 12. JITTERED POINTS + MEAN MARKERS
# =============================================================================

rng = np.random.default_rng(42)

for i, (arr, color) in enumerate(zip(plot_arrays, colors), start=1):

    x_jitter = rng.normal(loc=i, scale=0.045, size=len(arr))

    # black yearly dots
    ax.scatter(
        x_jitter,
        arr,
        s=34,
        color="#3B3B3B",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.90,
        zorder=3
    )

    # white mean diamond
    ax.scatter(
        i,
        arr.mean(),
        marker="D",
        s=82,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.3,
        zorder=4
    )


# =============================================================================
# 13. AXES / LABELS
# =============================================================================

ax.set_xticks(positions)
ax.set_xticklabels(x_labels)

ax.set_ylabel("Accuracy", labelpad=10)
ax.set_xlabel("")   # removed x-axis label

ax.set_ylim(0.30, 1.02)
ax.set_yticks(np.arange(0.3, 1.01, 0.1))

ax.tick_params(axis="x", pad=3)
ax.tick_params(axis="y", pad=3)


# =============================================================================
# 14. GRID / SPINES
# =============================================================================

ax.grid(
    axis="y",
    color="#D0D5DA",
    linewidth=0.7,
    alpha=0.75
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)


# =============================================================================
# 15. SMALL ANNOTATION
# =============================================================================

ax.text(
    0.985, 0.08,
    "LOYO per-year accuracy distributions",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=24,
    color="#4A4A4A"
)


# =============================================================================
# 16. FINAL LAYOUT
# =============================================================================

fig.tight_layout(pad=0.9)


# =============================================================================
# 17. SAVE
# =============================================================================

tiff_file = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.tiff"
pdf_file  = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.pdf"
svg_file  = OUTPUT_DIR / "Figure_05_Accuracy_Violin_Box.svg"

fig.savefig(
    tiff_file,
    dpi=1000,
    format="tiff",
    bbox_inches="tight",
    pad_inches=0.04,
    pil_kwargs={"compression": "tiff_lzw"}
)

fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04
)

fig.savefig(
    svg_file,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.04
)

plt.show()
plt.close(fig)


# =============================================================================
# 18. OUTPUT SUMMARY
# =============================================================================

print("\nAccuracy violin + box plot saved:")
print(f"TIFF : {tiff_file}")
print(f"PDF  : {pdf_file}")
print(f"SVG  : {svg_file}")


# In[9]:


"""Create final LOYO distribution plots for:
1. Accuracy
2. Macro-F1

Style:
- violin + box plot
- larger fonts everywhere
- larger black yearly dots
- larger white mean diamonds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

PRED_FILE = (
    ROOT
    / "Results"
    / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_05"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIGURE STYLE  (increased font sizes)
# =============================================================================

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 28,

    "axes.labelsize": 40,
    "axes.titlesize": 34,

    "xtick.labelsize": 31,
    "ytick.labelsize": 31,

    "legend.fontsize": 24,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 3. SETTINGS
# =============================================================================

TRUE_COL = "True_State"
YEAR_COL = "Test_Year"

MODEL_INFO = [
    ("CatBoost",              "CatBoost_Prediction",              "CatBoost",  "#4E79A7"),
    ("XGBoost",               "XGBoost_Prediction",               "XGB",       "#F28E2B"),
    ("HistGradientBoosting",  "HistGradientBoosting_Prediction",  "HGB",       "#59A14F"),
    ("Equal Soft Voting",     "Equal_Soft_Voting_Prediction",     "Soft Vote", "#9C6FB6"),
    ("TCN",                   "TCN_Prediction",                   "TCN",       "#D37267"),
    ("CNN-LSTM",              "CNN_LSTM_Prediction",              "CNN-LSTM",  "#5B9EA0"),
]


# =============================================================================
# 4. LOAD DATA
# =============================================================================

data = pd.read_csv(PRED_FILE)

required_base = [TRUE_COL, YEAR_COL]
missing_base = [col for col in required_base if col not in data.columns]

if missing_base:
    raise ValueError(
        f"Missing required base columns: {missing_base}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )

prediction_cols = [item[1] for item in MODEL_INFO]
missing_preds = [col for col in prediction_cols if col not in data.columns]

if missing_preds:
    raise ValueError(
        f"Missing required prediction columns: {missing_preds}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 5. CLEAN DATA
# =============================================================================

data = data.copy()
data = data.dropna(subset=[TRUE_COL, YEAR_COL] + prediction_cols)

data[TRUE_COL] = data[TRUE_COL].astype(int)
data[YEAR_COL] = data[YEAR_COL].astype(int)

for _, pred_col, _, _ in MODEL_INFO:
    data[pred_col] = data[pred_col].astype(int)


# =============================================================================
# 6. COMPUTE LOYO PER-YEAR METRICS
# =============================================================================

accuracy_records = []
f1_records = []

for model_name, pred_col, short_label, color in MODEL_INFO:

    for year, group in data.groupby(YEAR_COL):

        y_true = group[TRUE_COL].to_numpy()
        y_pred = group[pred_col].to_numpy()

        accuracy_value = np.mean(y_true == y_pred)
        macro_f1_value = f1_score(y_true, y_pred, average="macro")

        accuracy_records.append({
            "Model": model_name,
            "Short_Label": short_label,
            "Color": color,
            "Year": int(year),
            "Metric": float(accuracy_value)
        })

        f1_records.append({
            "Model": model_name,
            "Short_Label": short_label,
            "Color": color,
            "Year": int(year),
            "Metric": float(macro_f1_value)
        })

accuracy_df = pd.DataFrame(accuracy_records)
f1_df = pd.DataFrame(f1_records)


# =============================================================================
# 7. HELPER FUNCTION TO DRAW VIOLIN + BOX PLOT
# =============================================================================

def make_distribution_plot(
    df,
    y_label,
    annotation_text,
    output_stub
):
    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    summary = (
        df.groupby("Model")["Metric"]
        .agg(["mean", "std", "median", "min", "max"])
        .reindex([item[0] for item in MODEL_INFO])
    )

    print(f"\n{annotation_text}")
    print("=" * 72)
    print(summary.round(3))

    # -------------------------------------------------------------------------
    # Prepare plotting arrays
    # -------------------------------------------------------------------------
    plot_arrays = []
    x_labels = []
    colors = []

    for model_name, pred_col, short_label, color in MODEL_INFO:
        arr = df.loc[df["Model"] == model_name, "Metric"].to_numpy()

        plot_arrays.append(arr)
        x_labels.append(short_label)
        colors.append(color)

    positions = np.arange(1, len(plot_arrays) + 1)

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13.2, 10.2))

    # -------------------------------------------------------------------------
    # Violin
    # -------------------------------------------------------------------------
    violin = ax.violinplot(
        plot_arrays,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(1.4)

    # -------------------------------------------------------------------------
    # Box plot
    # -------------------------------------------------------------------------
    box = ax.boxplot(
        plot_arrays,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.3),
        whiskerprops=dict(color="#333333", linewidth=1.6),
        capprops=dict(color="#333333", linewidth=1.6),
        boxprops=dict(edgecolor="#222222", linewidth=1.6)
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)

    # -------------------------------------------------------------------------
    # Jittered yearly points + mean diamond
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(42)

    for i, (arr, color) in enumerate(zip(plot_arrays, colors), start=1):

        x_jitter = rng.normal(loc=i, scale=0.045, size=len(arr))

        # black yearly points (increased)
        ax.scatter(
            x_jitter,
            arr,
            s=46,
            color="#3A3A3A",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            zorder=3
        )

        # white mean diamond (increased)
        ax.scatter(
            i,
            arr.mean(),
            marker="D",
            s=110,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.4,
            zorder=4
        )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)

    ax.set_ylabel(y_label, labelpad=12)
    ax.set_xlabel("")

    ax.set_ylim(0.30, 1.02)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))

    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)

    # -------------------------------------------------------------------------
    # Grid / spines
    # -------------------------------------------------------------------------
    ax.grid(
        axis="y",
        color="#D0D5DA",
        linewidth=0.75,
        alpha=0.78
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)

    # -------------------------------------------------------------------------
    # Lower-right annotation (bigger)
    # -------------------------------------------------------------------------
    ax.text(
        0.985, 0.085,
        annotation_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=26,
        color="#4A4A4A"
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig.tight_layout(pad=0.9)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    tiff_file = OUTPUT_DIR / f"{output_stub}.tiff"
    pdf_file  = OUTPUT_DIR / f"{output_stub}.pdf"
    svg_file  = OUTPUT_DIR / f"{output_stub}.svg"

    fig.savefig(
        tiff_file,
        dpi=1000,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    fig.savefig(
        pdf_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04
    )

    fig.savefig(
        svg_file,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.04
    )

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {output_stub}")
    print(f"TIFF : {tiff_file}")
    print(f"PDF  : {pdf_file}")
    print(f"SVG  : {svg_file}")


# =============================================================================
# 8. ACCURACY PLOT
# =============================================================================

make_distribution_plot(
    df=accuracy_df,
    y_label="Accuracy",
    annotation_text="LOYO per-year accuracy distributions",
    output_stub="Figure_05_Accuracy_Violin_Box_Final"
)


# =============================================================================
# 9. MACRO-F1 PLOT
# =============================================================================

make_distribution_plot(
    df=f1_df,
    y_label="Macro-F1",
    annotation_text="LOYO per-year macro-F1 distributions",
    output_stub="Figure_05_MacroF1_Violin_Box_Final"
)


# In[10]:


"""Publication-quality Spearman correlation heatmap
for the 14 current environmental predictors.

Purpose
-------
Show the correlation structure among the environmental predictors used
for ecological-state prediction.

Features
--------
- Spearman rank correlation
- lower-triangular matrix only
- correlation coefficient inside every visible cell
- large publication-quality fonts
- fixed colour range from -1 to +1
- no significance stars
- chemical notation for NO3, PO4 and pCO2
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(
    r"D:\Prof Ali 2026\Spec_August_analysis\Final_Analysis"
)

DATA_FILE = (
    ROOT
    / "Data"
    / "04_Final_Current_Environmental_Modeling_Data.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Figures"
    / "Main_Figures"
    / "Figure_08"
)

RESULT_DIR = (
    ROOT
    / "Results"
    / "02_Environmental_Analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FINAL 14 CURRENT ENVIRONMENTAL PREDICTORS
# =============================================================================

FEATURES = [
    "SST",
    "NO3",
    "PO4",
    "SPCo2",
    "MLD",
    "SSS",
    "SSH",
    "PAR",
    "PDO",
    "NINO_3.4",
    "WPI",
    "MHW_MeanInt",
    "MHW_MaxInt",
    "MHW_CumInt",
]


# =============================================================================
# 3. PUBLICATION LABELS
# =============================================================================

FEATURE_LABELS = {
    "SST": "SST",

    "NO3": r"NO$_3$",
    "PO4": r"PO$_4$",
    "SPCo2": r"$p$CO$_2$",

    "MLD": "MLD",
    "SSS": "SSS",
    "SSH": "SSH",
    "PAR": "PAR",

    "PDO": "PDO",
    "NINO_3.4": "Niño 3.4",
    "WPI": "WPI",

    "MHW_MeanInt": r"MHW$_{\mathrm{mean}}$",
    "MHW_MaxInt": r"MHW$_{\mathrm{max}}$",
    "MHW_CumInt": r"MHW$_{\mathrm{cum}}$",
}


# =============================================================================
# 4. FIGURE STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "Calibri",

    "font.size": 27,

    "axes.labelsize": 30,
    "axes.titlesize": 31,

    "xtick.labelsize": 25,
    "ytick.labelsize": 25,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =============================================================================
# 5. LOAD DATA
# =============================================================================

data = pd.read_csv(
    DATA_FILE
)


missing = [
    feature
    for feature in FEATURES
    if feature not in data.columns
]


if missing:

    raise ValueError(
        f"Missing environmental variables: {missing}\n\n"
        f"Available columns:\n{list(data.columns)}"
    )


# =============================================================================
# 6. PREPARE NUMERIC DATA
# =============================================================================

env = data[FEATURES].copy()


for feature in FEATURES:

    env[feature] = pd.to_numeric(
        env[feature],
        errors="coerce"
    )


# We allow pairwise missing values for correlations.
print(
    f"\nNumber of observations : {len(env)}"
)

print(
    f"Number of predictors   : {len(FEATURES)}"
)


# =============================================================================
# 7. SPEARMAN CORRELATION
# =============================================================================

corr = env.corr(
    method="spearman"
)


# Save numeric matrix
corr_file = (
    RESULT_DIR
    / "Spearman_Correlation_14_Environmental_Predictors.csv"
)

corr.to_csv(
    corr_file
)


# =============================================================================
# 8. PRINT STRONGEST PAIRWISE CORRELATIONS
# =============================================================================

pairs = []


for i in range(len(FEATURES)):

    for j in range(i):

        pairs.append({

            "Feature_1": FEATURES[i],

            "Feature_2": FEATURES[j],

            "Spearman_rho":
                corr.iloc[i, j],

            "Absolute_rho":
                abs(corr.iloc[i, j])
        })


pair_df = pd.DataFrame(
    pairs
).sort_values(
    "Absolute_rho",
    ascending=False
)


print(
    "\nStrongest Spearman correlations:"
)

print(
    "=" * 68
)

print(
    pair_df[
        [
            "Feature_1",
            "Feature_2",
            "Spearman_rho"
        ]
    ]
    .head(15)
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 9. LOWER-TRIANGLE MASK
#
# Hide the upper half but KEEP the diagonal.
# =============================================================================

mask = np.triu(
    np.ones_like(
        corr,
        dtype=bool
    ),
    k=1
)


# =============================================================================
# 10. PUBLICATION LABELS
# =============================================================================

display_labels = [

    FEATURE_LABELS.get(
        feature,
        feature
    )

    for feature in FEATURES
]


# =============================================================================
# 11. CREATE FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13.5, 11.5)
)


# =============================================================================
# 12. CORRELATION HEATMAP
# =============================================================================

heatmap = sns.heatmap(

    corr,

    mask=mask,

    cmap="RdBu",

    vmin=-1,
    vmax=1,
    center=0,

    square=True,

    linewidths=1.3,
    linecolor="white",

    annot=True,
    fmt=".2f",

    annot_kws={
        "fontsize": 19,
        "fontweight": "normal"
    },

    xticklabels=display_labels,
    yticklabels=display_labels,

    cbar_kws={
        "shrink": 0.78,
        "pad": 0.025,
        "aspect": 28
    },

    ax=ax
)


# =============================================================================
# 13. IMPROVE ANNOTATION CONTRAST
#
# White text in strongly coloured cells;
# dark text in weakly coloured cells.
# =============================================================================

visible_values = []


for i in range(len(FEATURES)):

    for j in range(len(FEATURES)):

        if not mask[i, j]:

            visible_values.append(
                corr.iloc[i, j]
            )


for text, value in zip(
    ax.texts,
    visible_values
):

    if abs(value) >= 0.55:

        text.set_color(
            "white"
        )

        text.set_fontweight(
            "bold"
        )

    else:

        text.set_color(
            "#202020"
        )


# =============================================================================
# 14. AXIS LABEL PRESENTATION
# =============================================================================

ax.set_xlabel("")
ax.set_ylabel("")


# X labels rotated for readability
plt.setp(
    ax.get_xticklabels(),

    rotation=45,

    ha="right",

    rotation_mode="anchor"
)


# Y labels horizontal
plt.setp(
    ax.get_yticklabels(),

    rotation=0
)


# Remove unnecessary tick marks
ax.tick_params(
    axis="both",
    length=0,
    pad=7
)


# =============================================================================
# 15. COLORBAR
# =============================================================================

cbar = heatmap.collections[0].colorbar


cbar.set_ticks(
    [-1, -0.5, 0, 0.5, 1]
)


cbar.set_ticklabels(
    ["−1.0", "−0.5", "0", "0.5", "1.0"]
)


cbar.ax.tick_params(
    labelsize=24,
    width=1.0,
    length=5
)


cbar.set_label(

    r"Spearman $\rho$",

    fontsize=29,

    labelpad=14
)


# =============================================================================
# 16. CLEAN BORDER
# =============================================================================

for spine in ax.spines.values():

    spine.set_visible(
        False
    )


# =============================================================================
# 17. NO MAIN TITLE
#
# Intentionally omitted because this will likely become panel F
# in a composite manuscript figure.
# =============================================================================


# =============================================================================
# 18. FINAL LAYOUT
# =============================================================================

fig.tight_layout(
    pad=0.7
)


# =============================================================================
# 19. SAVE
# =============================================================================

tiff_file = (
    OUTPUT_DIR
    / "Environmental_Spearman_Correlation_Heatmap.tiff"
)

pdf_file = (
    OUTPUT_DIR
    / "Environmental_Spearman_Correlation_Heatmap.pdf"
)

svg_file = (
    OUTPUT_DIR
    / "Environmental_Spearman_Correlation_Heatmap.svg"
)


fig.savefig(

    tiff_file,

    dpi=1000,

    format="tiff",

    bbox_inches="tight",

    pad_inches=0.04,

    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


fig.savefig(

    pdf_file,

    format="pdf",

    bbox_inches="tight",

    pad_inches=0.04
)


fig.savefig(

    svg_file,

    format="svg",

    bbox_inches="tight",

    pad_inches=0.04
)


plt.show()

plt.close(
    fig
)


# =============================================================================
# 20. OUTPUT SUMMARY
# =============================================================================

print(
    "\nEnvironmental Spearman correlation heatmap saved:"
)

print(
    f"TIFF : {tiff_file}"
)

print(
    f"PDF  : {pdf_file}"
)

print(
    f"SVG  : {svg_file}"
)

print(
    f"\nCorrelation matrix:\n{corr_file}"
)


# In[ ]:




