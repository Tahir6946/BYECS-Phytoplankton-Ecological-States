#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


"""Final ecological-state construction for the phytoplankton study.

Raw data -> relative composition -> CLR -> STL detrending -> PCA -> K-means.
The previous state file is used only for a final agreement check.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, silhouette_score, calinski_harabasz_score,
    davies_bouldin_score
)
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. SETTINGS AND FOLDERS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
RESULT_DIR = ROOT / "Results" / "01_State_Discovery"
MODEL_DIR = ROOT / "Models"
FIGURE_DIR = ROOT / "Figures"
SCRIPT_DIR = ROOT / "Scripts"

RAW_FILE = DATA_DIR / "01_Raw_Species_Environmental_Monthly_Data.xlsx"
REFERENCE_FILE = DATA_DIR / "02_Reference_Final_Ecological_States.xlsx"

DATE_COL = "Months"
PHYTO = ["DIATO", "DINO", "HAPTO", "GREEN", "PROKAR", "PROCHLO"]
RANDOM_STATE = 42
FINAL_K = 3
RETAINED_PCS = 2
BOOTSTRAP_RUNS = 200
SEED_RUNS = 100
STL_PERIOD = 12

for folder in [RESULT_DIR, MODEL_DIR, FIGURE_DIR / "Main",
               FIGURE_DIR / "Supplementary", SCRIPT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HELPERS
# =============================================================================
def ecological_relabel(raw_labels, relative_data):
    """Assign stable ecological labels independent of K-means numbering."""
    means = relative_data.assign(Raw_Cluster=raw_labels).groupby("Raw_Cluster").mean()
    state_1_cluster = means["DIATO"].idxmax()
    state_3_cluster = means["PROKAR"].idxmax()

    if state_1_cluster == state_3_cluster:
        raise RuntimeError("DIATO-rich and PROKAR-rich states were not separable.")

    state_2_cluster = next(
        cluster for cluster in means.index
        if cluster not in {state_1_cluster, state_3_cluster}
    )
    mapping = {
        int(state_1_cluster): 1,
        int(state_2_cluster): 2,
        int(state_3_cluster): 3
    }
    states = np.array([mapping[int(label)] for label in raw_labels])
    return states, mapping


def align_labels(reference, candidate):
    """Align arbitrary candidate cluster numbers to reference labels."""
    ref_values = np.sort(np.unique(reference))
    cand_values = np.sort(np.unique(candidate))
    contingency = np.zeros((len(ref_values), len(cand_values)), dtype=int)
    for i, ref_label in enumerate(ref_values):
        for j, cand_label in enumerate(cand_values):
            contingency[i, j] = np.sum(
                (reference == ref_label) & (candidate == cand_label)
            )
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {cand_values[column]: ref_values[row]
               for row, column in zip(rows, columns)}
    return np.array([mapping[label] for label in candidate])


STATE_NAMES = {
    1: "Diatom-associated early-year state",
    2: "Mixed-prokaryote mid-year state",
    3: "Prokaryote-dominated late-year state"
}


# =============================================================================
# 3. LOAD AND VALIDATE THE RAW DATA
# =============================================================================
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw input was not found:\n{RAW_FILE}")

raw = pd.read_excel(RAW_FILE)
missing_columns = [column for column in [DATE_COL, *PHYTO]
                   if column not in raw.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
raw = raw.sort_values(DATE_COL).reset_index(drop=True)

if raw[DATE_COL].isna().any():
    raise ValueError("Invalid dates were found.")
if raw[DATE_COL].duplicated().any():
    raise ValueError("Duplicate months were found.")
if raw[PHYTO].isna().any().any():
    raise ValueError("Missing phytoplankton values were found.")
if (raw[PHYTO] <= 0).any().any():
    raise ValueError("CLR requires strictly positive phytoplankton values.")


# =============================================================================
# 4. RELATIVE COMMUNITY COMPOSITION AND CLR
# =============================================================================
abundance = raw[PHYTO].astype(float)
phyto_total = abundance.sum(axis=1)
relative = abundance.div(phyto_total, axis=0)

log_relative = np.log(relative)
clr = log_relative.sub(log_relative.mean(axis=1), axis=0)
clr.columns = [f"{column}_CLR" for column in PHYTO]

composition_output = pd.concat([
    raw[[DATE_COL]],
    phyto_total.rename("PHYTO_TOTAL"),
    abundance,
    relative.rename(columns={column: f"{column}_REL" for column in PHYTO})
], axis=1)
composition_output.to_csv(
    RESULT_DIR / "01_Community_Composition.csv", index=False
)

pd.concat([raw[[DATE_COL]], clr], axis=1).to_csv(
    RESULT_DIR / "02_CLR_Values.csv", index=False
)


# =============================================================================
# 5. REMOVE LONG-TERM TREND FROM EACH CLR SERIES
# =============================================================================
detrended = pd.DataFrame(index=raw.index)
stl_trends = pd.DataFrame(index=raw.index)

for phyto_group, clr_column in zip(PHYTO, clr.columns):
    stl_result = STL(
        clr[clr_column].to_numpy(), period=STL_PERIOD, robust=True
    ).fit()
    stl_trends[f"{phyto_group}_TREND"] = stl_result.trend
    detrended[f"{phyto_group}_DCLR"] = (
        clr[clr_column].to_numpy() - stl_result.trend
    )

# Recenter every month to preserve the zero-sum CLR geometry.
detrended = detrended.sub(detrended.mean(axis=1), axis=0)

detrended_output = pd.concat([raw[[DATE_COL]], detrended, stl_trends], axis=1)
detrended_output.to_csv(
    RESULT_DIR / "03_Detrended_CLR_Values.csv", index=False
)


# =============================================================================
# 6. PCA ON DETRENDED CLR VALUES
# =============================================================================
pca = PCA(n_components=len(PHYTO) - 1)
pca_scores = pca.fit_transform(detrended)
pc_names = [f"PC{index + 1}" for index in range(pca_scores.shape[1])]

pca_score_table = pd.DataFrame(pca_scores, columns=pc_names)
pca_score_table.insert(0, DATE_COL, raw[DATE_COL])
pca_score_table.to_csv(RESULT_DIR / "04_PCA_Scores.csv", index=False)

loading_table = pd.DataFrame(
    pca.components_.T,
    index=PHYTO,
    columns=pc_names
)
loading_table.index.name = "Phytoplankton_Group"
loading_table.to_csv(RESULT_DIR / "05_PCA_Loadings.csv")

variance_table = pd.DataFrame({
    "Component": pc_names,
    "Explained_Variance_Percent": pca.explained_variance_ratio_ * 100,
    "Cumulative_Variance_Percent": np.cumsum(
        pca.explained_variance_ratio_ * 100
    )
})
variance_table.to_csv(
    RESULT_DIR / "06_PCA_Explained_Variance.csv", index=False
)

cluster_input = pca_scores[:, :RETAINED_PCS]


# =============================================================================
# 7. K-MEANS MODEL-SELECTION CHECK: K = 2 TO 6
# =============================================================================
validation_records = []
for number_of_clusters in range(2, 7):
    candidate = KMeans(
        n_clusters=number_of_clusters,
        n_init=50,
        random_state=RANDOM_STATE
    )
    labels = candidate.fit_predict(cluster_input)
    counts = np.bincount(labels)
    validation_records.append({
        "K": number_of_clusters,
        "Silhouette": silhouette_score(cluster_input, labels),
        "Calinski_Harabasz": calinski_harabasz_score(cluster_input, labels),
        "Davies_Bouldin": davies_bouldin_score(cluster_input, labels),
        "Smallest_Cluster": counts.min(),
        "Largest_Cluster": counts.max()
    })

validation_table = pd.DataFrame(validation_records)
validation_table.to_csv(
    RESULT_DIR / "07_Cluster_Validation.csv", index=False
)


# =============================================================================
# 8. FINAL K-MEANS AND STABILITY
# =============================================================================
final_kmeans = KMeans(
    n_clusters=FINAL_K,
    n_init=100,
    random_state=RANDOM_STATE
)
raw_cluster = final_kmeans.fit_predict(cluster_input)
final_states, cluster_to_state = ecological_relabel(raw_cluster, relative)

rng = np.random.default_rng(RANDOM_STATE)
bootstrap_ari = []
for run in range(BOOTSTRAP_RUNS):
    sample_index = rng.choice(len(cluster_input), len(cluster_input), replace=True)
    bootstrap_model = KMeans(
        n_clusters=FINAL_K, n_init=30, random_state=RANDOM_STATE + run + 1
    )
    bootstrap_model.fit(cluster_input[sample_index])
    bootstrap_labels = bootstrap_model.predict(cluster_input)
    bootstrap_ari.append(adjusted_rand_score(raw_cluster, bootstrap_labels))

seed_ari = []
for seed in range(SEED_RUNS):
    seed_labels = KMeans(
        n_clusters=FINAL_K, n_init=20, random_state=seed
    ).fit_predict(cluster_input)
    seed_ari.append(adjusted_rand_score(raw_cluster, seed_labels))

stability_summary = pd.DataFrame([
    {
        "Stability_Test": "Bootstrap",
        "Runs": BOOTSTRAP_RUNS,
        "Mean_ARI": np.mean(bootstrap_ari),
        "SD_ARI": np.std(bootstrap_ari, ddof=1),
        "Median_ARI": np.median(bootstrap_ari),
        "Minimum_ARI": np.min(bootstrap_ari),
        "ARI_Above_0.80_Percent": np.mean(np.array(bootstrap_ari) >= 0.80) * 100
    },
    {
        "Stability_Test": "Random seed",
        "Runs": SEED_RUNS,
        "Mean_ARI": np.mean(seed_ari),
        "SD_ARI": np.std(seed_ari, ddof=1),
        "Median_ARI": np.median(seed_ari),
        "Minimum_ARI": np.min(seed_ari),
        "ARI_Above_0.80_Percent": np.mean(np.array(seed_ari) >= 0.80) * 100
    }
])
stability_summary.to_csv(
    RESULT_DIR / "08_Cluster_Stability.csv", index=False
)

pd.DataFrame({
    "Bootstrap_Run": np.arange(1, BOOTSTRAP_RUNS + 1),
    "ARI": bootstrap_ari
}).to_csv(RESULT_DIR / "08b_Bootstrap_ARI_Runs.csv", index=False)


# =============================================================================
# 9. FINAL ECOLOGICAL-STATE TABLES
# =============================================================================
state_output = pd.DataFrame({
    DATE_COL: raw[DATE_COL],
    "PC1": pca_scores[:, 0],
    "PC2": pca_scores[:, 1],
    "Raw_Cluster": raw_cluster,
    "Ecological_State": final_states,
    "State_Name": [STATE_NAMES[state] for state in final_states]
})
state_output.to_csv(
    RESULT_DIR / "09_Final_Ecological_States.csv", index=False
)

relative_percent = relative * 100
relative_percent["Ecological_State"] = final_states
state_composition = relative_percent.groupby("Ecological_State")[PHYTO].mean()
state_composition.insert(0, "State_Name", [STATE_NAMES[state]
                                           for state in state_composition.index])
state_composition.to_csv(RESULT_DIR / "10_State_Composition.csv")

monthly_occurrence = pd.crosstab(
    raw[DATE_COL].dt.month,
    final_states,
    normalize="index"
) * 100
monthly_occurrence = monthly_occurrence.reindex(index=range(1, 13), columns=[1, 2, 3], fill_value=0)
monthly_occurrence.index.name = "Month_Number"
monthly_occurrence.columns = [f"State_{state}_Percent" for state in monthly_occurrence.columns]
monthly_occurrence.to_csv(RESULT_DIR / "11_Monthly_State_Occurrence.csv")


# =============================================================================
# 10. REFERENCE AGREEMENT CHECK — NEVER USED FOR MODEL FITTING
# =============================================================================
agreement_record = {
    "Reference_File_Found": REFERENCE_FILE.exists(),
    "Matched_Months": 0,
    "Adjusted_Rand_Index": np.nan,
    "Direct_Agreement_Percent": np.nan
}

if REFERENCE_FILE.exists():
    reference = pd.read_excel(REFERENCE_FILE)
    if DATE_COL in reference.columns and "Ecological_State" in reference.columns:
        reference[DATE_COL] = pd.to_datetime(reference[DATE_COL])
        comparison = state_output[[DATE_COL, "Ecological_State"]].merge(
            reference[[DATE_COL, "Ecological_State"]],
            on=DATE_COL,
            suffixes=("_New", "_Reference")
        )
        agreement_record["Matched_Months"] = len(comparison)
        agreement_record["Adjusted_Rand_Index"] = adjusted_rand_score(
            comparison["Ecological_State_Reference"],
            comparison["Ecological_State_New"]
        )
        agreement_record["Direct_Agreement_Percent"] = np.mean(
            comparison["Ecological_State_Reference"].to_numpy()
            == comparison["Ecological_State_New"].to_numpy()
        ) * 100

pd.DataFrame([agreement_record]).to_csv(
    RESULT_DIR / "12_Reference_Agreement.csv", index=False
)


# =============================================================================
# 11. SAVE MODELS AND SETTINGS
# =============================================================================
joblib.dump({
    "pca_model": pca,
    "phytoplankton_groups": PHYTO,
    "retained_pcs": RETAINED_PCS,
    "stl_period": STL_PERIOD,
    "clr_definition": "log(x_i / geometric_mean(x))"
}, MODEL_DIR / "CLR_PCA_Model.pkl")

joblib.dump({
    "kmeans_model": final_kmeans,
    "cluster_to_ecological_state": cluster_to_state,
    "state_names": STATE_NAMES
}, MODEL_DIR / "Final_KMeans_Model.pkl")

settings = {
    "raw_file": str(RAW_FILE),
    "phytoplankton_groups": PHYTO,
    "stl_period": STL_PERIOD,
    "retained_pcs": RETAINED_PCS,
    "final_k": FINAL_K,
    "kmeans_n_init": 100,
    "random_state": RANDOM_STATE,
    "bootstrap_runs": BOOTSTRAP_RUNS,
    "seed_runs": SEED_RUNS
}
(RESULT_DIR / "00_Analysis_Settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8"
)


# =============================================================================
# 12. COMPACT SUMMARY
# =============================================================================
state_sizes = pd.Series(final_states).value_counts().sort_index()

print("\n" + "=" * 94)
print("FINAL ECOLOGICAL-STATE CONSTRUCTION COMPLETED")
print("=" * 94)
print(f"Observations                  : {len(raw)}")
print(f"PC1-PC2 cumulative variance  : {variance_table.loc[1, 'Cumulative_Variance_Percent']:.3f}%")
print(f"Final K                      : {FINAL_K}")
for state in [1, 2, 3]:
    print(f"State {state}: {state_sizes[state]:3d} months ({state_sizes[state] / len(raw) * 100:5.2f}%)")
print(f"Bootstrap mean ARI           : {np.mean(bootstrap_ari):.3f}")
print(f"Random-seed mean ARI         : {np.mean(seed_ari):.3f}")
if agreement_record["Matched_Months"]:
    print(f"Reference ARI                : {agreement_record['Adjusted_Rand_Index']:.3f}")
    print(f"Reference direct agreement   : {agreement_record['Direct_Agreement_Percent']:.2f}%")
print("\nResults saved in:")
print(RESULT_DIR)


# In[1]:


"""Build the final current-feature environmental modelling dataset.

Inputs
------
1. Clean raw monthly dataset
2. Ecological states reconstructed by Script 01

Outputs
-------
- One leakage-free modelling table containing 14 current predictors
- Environmental summaries, state comparisons, effect sizes and correlations
- Plotting-ready long-format environmental data

No lagged or anomaly features are created in the final analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, skew
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


# =============================================================================
# 1. SETTINGS AND FOLDERS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
STATE_DIR = ROOT / "Results" / "01_State_Discovery"
RESULT_DIR = ROOT / "Results" / "02_Environmental_Analysis"

RAW_FILE = DATA_DIR / "01_Raw_Species_Environmental_Monthly_Data.xlsx"
STATE_FILE = STATE_DIR / "09_Final_Ecological_States.csv"
REFERENCE_FILE = DATA_DIR / "03_Reference_Environmental_Modeling_Data.xlsx"

FINAL_MODEL_FILE = DATA_DIR / "04_Final_Current_Environmental_Modeling_Data.csv"

DATE_COL = "Months"
TARGET_COL = "Ecological_State"

ENVIRONMENTAL_FEATURES = [
    "SST", "NO3", "PO4", "SPCo2", "MLD", "SSS", "SSH", "PAR",
    "PDO", "NINO_3.4", "WPI",
    "MHW_MeanInt", "MHW_MaxInt", "MHW_CumInt"
]

STATE_NAMES = {
    1: "Diatom-associated early-year state",
    2: "Mixed-prokaryote mid-year state",
    3: "Prokaryote-dominated late-year state"
}

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HELPERS
# =============================================================================
def epsilon_squared_kruskal(h_statistic, n_observations, n_groups):
    """Kruskal-Wallis epsilon-squared effect size, restricted to [0, 1]."""
    value = (h_statistic - n_groups + 1) / (n_observations - n_groups)
    return float(np.clip(value, 0, 1))


def effect_size_label(value):
    if value < 0.01:
        return "Negligible"
    if value < 0.06:
        return "Small"
    if value < 0.14:
        return "Moderate"
    return "Large"


# =============================================================================
# 3. LOAD AND VALIDATE INPUT FILES
# =============================================================================
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw data file was not found:\n{RAW_FILE}")
if not STATE_FILE.exists():
    raise FileNotFoundError(
        "Final state file was not found. Run Script 01 first:\n"
        f"{STATE_FILE}"
    )

raw = pd.read_excel(RAW_FILE)
states = pd.read_csv(STATE_FILE)

raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
states[DATE_COL] = pd.to_datetime(states[DATE_COL], errors="coerce")

required_raw = [DATE_COL, *ENVIRONMENTAL_FEATURES]
required_states = [DATE_COL, TARGET_COL]

missing_raw = [column for column in required_raw if column not in raw.columns]
missing_states = [column for column in required_states if column not in states.columns]
if missing_raw:
    raise ValueError(f"Missing raw-data columns: {missing_raw}")
if missing_states:
    raise ValueError(f"Missing state-file columns: {missing_states}")

if raw[DATE_COL].isna().any() or states[DATE_COL].isna().any():
    raise ValueError("Invalid dates were found.")
if raw[DATE_COL].duplicated().any() or states[DATE_COL].duplicated().any():
    raise ValueError("Duplicate months were found.")


# =============================================================================
# 4. MERGE CURRENT ENVIRONMENTAL FEATURES WITH FINAL STATES
# =============================================================================
model_data = raw[required_raw].merge(
    states[[DATE_COL, TARGET_COL]],
    on=DATE_COL,
    how="inner",
    validate="one_to_one"
).sort_values(DATE_COL).reset_index(drop=True)

if len(model_data) != len(raw) or len(model_data) != len(states):
    raise ValueError(
        f"Date alignment failed: raw={len(raw)}, states={len(states)}, "
        f"merged={len(model_data)}"
    )

model_data[TARGET_COL] = model_data[TARGET_COL].astype(int)
unexpected_states = sorted(set(model_data[TARGET_COL]) - {1, 2, 3})
if unexpected_states:
    raise ValueError(f"Unexpected ecological states: {unexpected_states}")

numeric_values = model_data[ENVIRONMENTAL_FEATURES].to_numpy(float)
if np.isnan(numeric_values).any():
    raise ValueError("Missing environmental predictor values were found.")
if not np.isfinite(numeric_values).all():
    raise ValueError("Infinite environmental predictor values were found.")

model_data.insert(1, "Year", model_data[DATE_COL].dt.year)
model_data.insert(2, "Month_Number", model_data[DATE_COL].dt.month)
model_data.insert(
    4,
    "State_Name",
    model_data[TARGET_COL].map(STATE_NAMES)
)

# Arrange metadata first and predictors afterward.
final_columns = [
    DATE_COL, "Year", "Month_Number", TARGET_COL, "State_Name",
    *ENVIRONMENTAL_FEATURES
]
model_data = model_data[final_columns]
model_data.to_csv(FINAL_MODEL_FILE, index=False)


# =============================================================================
# 5. DATA AND STATE SUMMARIES
# =============================================================================
feature_summary = pd.DataFrame({
    "Variable": ENVIRONMENTAL_FEATURES,
    "Count": [model_data[column].count() for column in ENVIRONMENTAL_FEATURES],
    "Mean": [model_data[column].mean() for column in ENVIRONMENTAL_FEATURES],
    "Standard_Deviation": [model_data[column].std(ddof=1) for column in ENVIRONMENTAL_FEATURES],
    "Minimum": [model_data[column].min() for column in ENVIRONMENTAL_FEATURES],
    "Median": [model_data[column].median() for column in ENVIRONMENTAL_FEATURES],
    "Maximum": [model_data[column].max() for column in ENVIRONMENTAL_FEATURES],
    "Skewness": [skew(model_data[column], bias=False) for column in ENVIRONMENTAL_FEATURES]
})
feature_summary.to_csv(
    RESULT_DIR / "01_Environmental_Feature_Summary.csv", index=False
)

state_counts = (
    model_data[TARGET_COL].value_counts().sort_index()
    .rename("Number_of_Months").to_frame()
)
state_counts["Percentage"] = state_counts["Number_of_Months"] / len(model_data) * 100
state_counts["State_Name"] = [STATE_NAMES[state] for state in state_counts.index]
state_counts.index.name = TARGET_COL
state_counts.to_csv(RESULT_DIR / "02_State_Distribution.csv")

state_medians = model_data.groupby(TARGET_COL)[ENVIRONMENTAL_FEATURES].median().T
state_medians.columns = [f"State_{state}" for state in state_medians.columns]
state_medians.index.name = "Variable"
state_medians.to_csv(RESULT_DIR / "03_State_Environmental_Medians.csv")

state_means = model_data.groupby(TARGET_COL)[ENVIRONMENTAL_FEATURES].mean().T
state_means.columns = [f"State_{state}" for state in state_means.columns]
state_means.index.name = "Variable"
state_means.to_csv(RESULT_DIR / "04_State_Environmental_Means.csv")


# =============================================================================
# 6. STANDARDIZED STATE PROFILES
# =============================================================================
scaler = StandardScaler()
standardized_values = scaler.fit_transform(model_data[ENVIRONMENTAL_FEATURES])
standardized = pd.DataFrame(
    standardized_values,
    columns=ENVIRONMENTAL_FEATURES,
    index=model_data.index
)
standardized[TARGET_COL] = model_data[TARGET_COL].to_numpy()

standardized_state_means = standardized.groupby(TARGET_COL)[ENVIRONMENTAL_FEATURES].mean().T
standardized_state_means.columns = [f"State_{state}" for state in standardized_state_means.columns]
standardized_state_means.index.name = "Variable"
standardized_state_means.to_csv(
    RESULT_DIR / "05_Standardized_Environmental_State_Means.csv"
)


# =============================================================================
# 7. KRUSKAL-WALLIS TESTS, FDR AND EFFECT SIZES
# =============================================================================
test_records = []
for variable in ENVIRONMENTAL_FEATURES:
    groups = [
        model_data.loc[model_data[TARGET_COL] == state, variable].to_numpy()
        for state in [1, 2, 3]
    ]
    h_statistic, p_value = kruskal(*groups)
    epsilon_squared = epsilon_squared_kruskal(
        h_statistic, len(model_data), len(groups)
    )
    test_records.append({
        "Variable": variable,
        "H_Statistic": h_statistic,
        "Raw_P_Value": p_value,
        "Epsilon_Squared": epsilon_squared,
        "Effect_Size": effect_size_label(epsilon_squared)
    })

environmental_tests = pd.DataFrame(test_records)
environmental_tests["FDR_P_Value"] = multipletests(
    environmental_tests["Raw_P_Value"], method="fdr_bh"
)[1]
environmental_tests["Significant_FDR"] = environmental_tests["FDR_P_Value"] < 0.05
environmental_tests = environmental_tests.sort_values(
    "Epsilon_Squared", ascending=False
).reset_index(drop=True)
environmental_tests.to_csv(
    RESULT_DIR / "06_Environmental_State_Tests.csv", index=False
)


# =============================================================================
# 8. CORRELATION AND PLOTTING-READY LONG DATA
# =============================================================================
correlation_matrix = model_data[ENVIRONMENTAL_FEATURES].corr(method="spearman")
correlation_matrix.index.name = "Variable"
correlation_matrix.to_csv(
    RESULT_DIR / "07_Environmental_Spearman_Correlation.csv"
)

correlation_pairs = []
for i, first_variable in enumerate(ENVIRONMENTAL_FEATURES):
    for second_variable in ENVIRONMENTAL_FEATURES[i + 1:]:
        value = correlation_matrix.loc[first_variable, second_variable]
        correlation_pairs.append({
            "Variable_1": first_variable,
            "Variable_2": second_variable,
            "Spearman_R": value,
            "Absolute_Spearman_R": abs(value),
            "Above_0.80": abs(value) >= 0.80
        })

pd.DataFrame(correlation_pairs).sort_values(
    "Absolute_Spearman_R", ascending=False
).to_csv(RESULT_DIR / "07b_Environmental_Correlation_Pairs.csv", index=False)

environmental_long = model_data.melt(
    id_vars=[DATE_COL, "Year", "Month_Number", TARGET_COL, "State_Name"],
    value_vars=ENVIRONMENTAL_FEATURES,
    var_name="Environmental_Variable",
    value_name="Value"
)
environmental_long.to_csv(
    RESULT_DIR / "08_Environmental_Long_Format.csv", index=False
)


# =============================================================================
# 9. REFERENCE CHECK — NOT USED TO BUILD THE NEW DATASET
# =============================================================================
reference_record = {
    "Reference_File_Found": REFERENCE_FILE.exists(),
    "Matched_Months": 0,
    "State_ARI": np.nan,
    "State_Direct_Agreement_Percent": np.nan,
    "Maximum_Absolute_Predictor_Difference": np.nan
}

if REFERENCE_FILE.exists():
    reference = pd.read_excel(REFERENCE_FILE)
    if DATE_COL in reference.columns:
        reference[DATE_COL] = pd.to_datetime(reference[DATE_COL], errors="coerce")
        common_columns = [
            column for column in ENVIRONMENTAL_FEATURES
            if column in reference.columns
        ]
        reference_columns = [DATE_COL, *common_columns]
        if TARGET_COL in reference.columns:
            reference_columns.append(TARGET_COL)

        comparison = model_data.merge(
            reference[reference_columns],
            on=DATE_COL,
            how="inner",
            suffixes=("_New", "_Reference")
        )
        reference_record["Matched_Months"] = len(comparison)

        if TARGET_COL in reference.columns and len(comparison):
            reference_record["State_ARI"] = adjusted_rand_score(
                comparison[f"{TARGET_COL}_New"],
                comparison[f"{TARGET_COL}_Reference"]
            )
            reference_record["State_Direct_Agreement_Percent"] = np.mean(
                comparison[f"{TARGET_COL}_New"].to_numpy()
                == comparison[f"{TARGET_COL}_Reference"].to_numpy()
            ) * 100

        if common_columns and len(comparison):
            differences = [
                np.max(np.abs(
                    comparison[f"{column}_New"].to_numpy(float)
                    - comparison[f"{column}_Reference"].to_numpy(float)
                ))
                for column in common_columns
            ]
            reference_record["Maximum_Absolute_Predictor_Difference"] = max(differences)

pd.DataFrame([reference_record]).to_csv(
    RESULT_DIR / "09_Reference_Modeling_Data_Check.csv", index=False
)


# =============================================================================
# 10. SAVE SETTINGS AND PRINT COMPACT SUMMARY
# =============================================================================
settings = {
    "raw_file": str(RAW_FILE),
    "state_file": str(STATE_FILE),
    "final_model_file": str(FINAL_MODEL_FILE),
    "predictor_family": "Current environmental conditions only",
    "environmental_features": ENVIRONMENTAL_FEATURES,
    "number_of_predictors": len(ENVIRONMENTAL_FEATURES),
    "number_of_observations": len(model_data),
    "target": TARGET_COL,
    "fdr_method": "Benjamini-Hochberg",
    "primary_effect_size": "Kruskal-Wallis epsilon-squared"
}
(RESULT_DIR / "00_Environmental_Analysis_Settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8"
)

print("\n" + "=" * 96)
print("FINAL CURRENT-FEATURE ENVIRONMENTAL DATASET COMPLETED")
print("=" * 96)
print(f"Observations                 : {len(model_data)}")
print(f"Environmental predictors     : {len(ENVIRONMENTAL_FEATURES)}")
print(f"First month                  : {model_data[DATE_COL].min().date()}")
print(f"Last month                   : {model_data[DATE_COL].max().date()}")
print(f"Missing predictor values     : {model_data[ENVIRONMENTAL_FEATURES].isna().sum().sum()}")
print(f"Infinite predictor values    : {np.isinf(model_data[ENVIRONMENTAL_FEATURES]).sum().sum()}")
for state in [1, 2, 3]:
    count = int((model_data[TARGET_COL] == state).sum())
    print(f"State {state}: {count:3d} months ({count / len(model_data) * 100:5.2f}%)")
print(f"FDR-significant variables    : {environmental_tests['Significant_FDR'].sum()} / {len(ENVIRONMENTAL_FEATURES)}")
print("\nTop five environmental effect sizes:")
print(environmental_tests[
    ["Variable", "Epsilon_Squared", "Effect_Size", "FDR_P_Value"]
].head(5).round(4).to_string(index=False))

if reference_record["Matched_Months"]:
    print(f"\nReference matched months     : {reference_record['Matched_Months']}")
    print(f"Reference state ARI          : {reference_record['State_ARI']:.3f}")
    print(f"Reference state agreement    : {reference_record['State_Direct_Agreement_Percent']:.2f}%")

print("\nFinal modelling dataset:")
print(FINAL_MODEL_FILE)
print("\nEnvironmental results:")
print(RESULT_DIR)


# In[3]:


"""Final optimized six-model LOYO analysis.

Final models
------------
1. CatBoost (Optuna-TPE)
2. XGBoost (Optuna-TPE)
3. HistGradientBoosting (Optuna-TPE)
4. Equal soft voting of the three tree models
5. TCN with six-month sequences
6. CNN-LSTM with six-month sequences

The seasonal baseline is retained as a reference. Every reported prediction is
held out by year. All models are evaluated on the same sequence-eligible months.
Checkpoint CSV files allow safe resumption after completed outer years.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import json
import random
import warnings
from pathlib import Path

# Suppress the optional Jupyter progress-widget warning. It does not affect
# Optuna or model training and no ipywidgets installation is required.
warnings.filterwarnings("ignore", message="IProgress not found.*")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as exc:
    raise ImportError(
        "Install Optuna once in the gpu-env environment:\n"
        "python -m pip install optuna"
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 1. SETTINGS AND FOLDERS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Data" / "04_Final_Current_Environmental_Modeling_Data.csv"
RESULT_DIR = ROOT / "Results" / "03_Final_Models"
MODEL_DIR = ROOT / "Models" / "Final_Six_Models"
CHECKPOINT_DIR = RESULT_DIR / "Checkpoints"

for folder in [RESULT_DIR, MODEL_DIR, CHECKPOINT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATE_COL = "Months"
TARGET_COL = "Ecological_State"
FEATURES = [
    "SST", "NO3", "PO4", "SPCo2", "MLD", "SSS", "SSH", "PAR",
    "PDO", "NINO_3.4", "WPI",
    "MHW_MeanInt", "MHW_MaxInt", "MHW_CumInt"
]

TREE_MODELS = ["CatBoost", "XGBoost", "HistGradientBoosting"]
FINAL_MODELS = [
    "CatBoost", "XGBoost", "HistGradientBoosting",
    "Equal Soft Voting", "TCN", "CNN-LSTM"
]
ALL_MODELS = ["Seasonal Baseline", *FINAL_MODELS]

RANDOM_STATE = 42
LOOKBACK = 6
N_TRIALS = 20
INNER_FOLDS = 4
MAX_EPOCHS = 200
BATCH_SIZE = 16
DL_VALIDATION_YEAR_FRACTION = 0.20
USE_GPU_FOR_TREES = True
RESUME_COMPLETED_YEARS = True

PREDICTION_CHECKPOINT = CHECKPOINT_DIR / "LOYO_Predictions_Checkpoint.csv"
PARAMETER_CHECKPOINT = CHECKPOINT_DIR / "Optuna_Parameters_Checkpoint.csv"
FOLD_CHECKPOINT = CHECKPOINT_DIR / "Outer_Fold_Scores_Checkpoint.csv"
HISTORY_CHECKPOINT = CHECKPOINT_DIR / "DL_Training_History_Checkpoint.csv"


# =============================================================================
# 2. REPRODUCIBILITY AND GPU SETUP
# =============================================================================
def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Compatible with older TensorFlow/Keras releases used by gpu-env.
    tf.random.set_seed(seed)


set_all_seeds(RANDOM_STATE)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================
def safe_read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def seasonal_probabilities(train_months, y_train, test_months):
    """Training-only monthly class probabilities with Laplace smoothing."""
    table = pd.DataFrame({"Month": train_months, "State": y_train})
    classes = np.array([0, 1, 2])
    overall = table["State"].value_counts().reindex(classes, fill_value=0).to_numpy(float)
    overall = (overall + 1) / (overall.sum() + len(classes))

    output = []
    for month in test_months:
        counts = (
            table.loc[table["Month"] == month, "State"]
            .value_counts().reindex(classes, fill_value=0).to_numpy(float)
        )
        output.append(
            overall if counts.sum() == 0
            else (counts + 1) / (counts.sum() + len(classes))
        )
    return np.asarray(output)


def macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    total = cm.sum()
    values = []
    for index in range(3):
        tn = total - cm[index, :].sum() - cm[:, index].sum() + cm[index, index]
        fp = cm[:, index].sum() - cm[index, index]
        values.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return float(np.nanmean(values))


def multiclass_brier_score(y_true, probability):
    one_hot = label_binarize(y_true, classes=[1, 2, 3])
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def pooled_metrics(model_name, y_true, y_pred, probability):
    one_hot = label_binarize(y_true, classes=[1, 2, 3])
    recalls = recall_score(
        y_true, y_pred, labels=[1, 2, 3], average=None, zero_division=0
    )
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
        "Macro_Specificity": macro_specificity(y_true, y_pred),
        "Macro_AUROC": roc_auc_score(
            one_hot, probability, average="macro", multi_class="ovr"
        ),
        "Macro_AUPRC": average_precision_score(
            one_hot, probability, average="macro"
        ),
        "Log_Loss": log_loss(y_true, probability, labels=[1, 2, 3]),
        "Multiclass_Brier_Score": multiclass_brier_score(y_true, probability),
        "State_1_Recall": recalls[0],
        "State_2_Recall": recalls[1],
        "State_3_Recall": recalls[2]
    }


# =============================================================================
# 4. OPTUNA TREE-MODEL HELPERS
# =============================================================================
def xgb_device_parameters():
    major = int(xgboost.__version__.split(".")[0])
    if USE_GPU_FOR_TREES:
        if major >= 2:
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
    return {"tree_method": "hist", **({"device": "cpu"} if major >= 2 else {})}


def suggest_tree_parameters(trial, model_name):
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 250, 900, step=50),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128])
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)
        }
    return {
        "max_iter": trial.suggest_int("max_iter", 150, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-4, 5.0, log=True
        )
    }


def build_tree_model(model_name, parameters, seed):
    if model_name == "CatBoost":
        gpu_parameters = (
            {"task_type": "GPU", "devices": "0"}
            if USE_GPU_FOR_TREES else {"task_type": "CPU"}
        )
        return CatBoostClassifier(
            **parameters,
            **gpu_parameters,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            bootstrap_type="Bayesian",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            **parameters,
            **xgb_device_parameters(),
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1
        )
    return HistGradientBoostingClassifier(
        **parameters,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed
    )


def fit_tree_model(model, model_name, X_fit, y_fit):
    if model_name == "XGBoost":
        weights = compute_sample_weight("balanced", y_fit)
        model.fit(X_fit, y_fit, sample_weight=weights)
    else:
        model.fit(X_fit, y_fit)
    return model


def optimize_tree_model(model_name, X_train, y_train, training_years, seed):
    inner_cv = GroupKFold(
        n_splits=min(INNER_FOLDS, len(np.unique(training_years)))
    )

    def objective(trial):
        parameters = suggest_tree_parameters(trial, model_name)
        scores = []
        for split_number, (fit_index, valid_index) in enumerate(
            inner_cv.split(X_train, y_train, groups=training_years)
        ):
            candidate = build_tree_model(
                model_name, parameters, seed + split_number
            )
            candidate = fit_tree_model(
                candidate, model_name,
                X_train.iloc[fit_index], y_train[fit_index]
            )
            prediction = np.asarray(
                candidate.predict(X_train.iloc[valid_index])
            ).reshape(-1).astype(int)
            scores.append(
                balanced_accuracy_score(y_train[valid_index], prediction)
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=5)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# =============================================================================
# 5. TEMPORAL SEQUENCE AND DL HELPERS
# =============================================================================
def make_sequences(X_values, y_values, years, months, dates):
    sequences, targets, target_indices = [], [], []
    window_years = []
    for target_index in range(LOOKBACK - 1, len(X_values)):
        start = target_index - LOOKBACK + 1
        sequences.append(X_values[start:target_index + 1])
        targets.append(y_values[target_index])
        target_indices.append(target_index)
        window_years.append(years[start:target_index + 1])
    return {
        "X": np.asarray(sequences, dtype=np.float32),
        "y": np.asarray(targets, dtype=int),
        "target_index": np.asarray(target_indices, dtype=int),
        "target_year": years[np.asarray(target_indices, dtype=int)],
        "target_month": months[np.asarray(target_indices, dtype=int)],
        "target_date": dates[np.asarray(target_indices, dtype=int)],
        "window_years": np.asarray(window_years, dtype=int)
    }


def select_dl_fit_validation_indices(sequence_data, candidate_indices, seed):
    """Choose validation years and purge them from training windows."""
    candidate_years = sequence_data["target_year"][candidate_indices]
    candidate_y = sequence_data["y"][candidate_indices]

    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=DL_VALIDATION_YEAR_FRACTION,
        random_state=seed
    )
    chosen_validation_years = None
    for fit_local, valid_local in splitter.split(
        candidate_indices, candidate_y, groups=candidate_years
    ):
        if (
            len(np.unique(candidate_y[fit_local])) == 3
            and len(np.unique(candidate_y[valid_local])) == 3
        ):
            chosen_validation_years = np.unique(candidate_years[valid_local])
            break
    if chosen_validation_years is None:
        raise RuntimeError("Could not construct a three-class DL validation split.")

    target_year = sequence_data["target_year"]
    windows = sequence_data["window_years"]
    candidate_mask = np.zeros(len(target_year), dtype=bool)
    candidate_mask[candidate_indices] = True

    validation_mask = candidate_mask & np.isin(target_year, chosen_validation_years)
    window_contains_validation_year = np.any(
        np.isin(windows, chosen_validation_years), axis=1
    )
    fit_mask = (
        candidate_mask
        & ~np.isin(target_year, chosen_validation_years)
        & ~window_contains_validation_year
    )
    return np.where(fit_mask)[0], np.where(validation_mask)[0], chosen_validation_years


def scale_sequence_sets(X_fit, X_valid, X_test):
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, X_fit.shape[-1]))

    def transform(values):
        shape = values.shape
        return scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    return transform(X_fit), transform(X_valid), transform(X_test), scaler


def build_tcn(input_shape, seed):
    set_all_seeds(seed)
    inputs = tf.keras.Input(shape=input_shape)
    first = tf.keras.layers.Conv1D(
        64, 3, padding="causal", dilation_rate=1, activation="relu"
    )(inputs)
    first = tf.keras.layers.BatchNormalization()(first)
    first = tf.keras.layers.Dropout(0.25)(first)
    second = tf.keras.layers.Conv1D(
        64, 3, padding="causal", dilation_rate=2
    )(first)
    second = tf.keras.layers.BatchNormalization()(second)
    second = tf.keras.layers.Activation("relu")(second)
    second = tf.keras.layers.Dropout(0.25)(second)
    x = tf.keras.layers.Add()([first, second])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="TCN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_lstm(input_shape, seed):
    set_all_seeds(seed)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(
        64, 3, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(48)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        learning_rate = getattr(
            self.model.optimizer,
            "learning_rate",
            getattr(self.model.optimizer, "lr", np.nan)
        )
        logs["learning_rate"] = float(
            tf.keras.backend.get_value(learning_rate)
        )


def train_dl_model(model_name, X_fit, y_fit, X_valid, y_valid, seed):
    tf.keras.backend.clear_session()
    gc.collect()
    model = (
        build_tcn(X_fit.shape[1:], seed)
        if model_name == "TCN"
        else build_cnn_lstm(X_fit.shape[1:], seed)
    )

    class_values = np.array([0, 1, 2])
    weights = compute_class_weight(
        class_weight="balanced", classes=class_values, y=y_fit
    )
    class_weight = {index: weight for index, weight in enumerate(weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-5, verbose=0
        ),
        LearningRateLogger()
    ]

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_valid, y_valid),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )
    return model, history.history


def history_to_records(history, model_name, run_type, test_year):
    number_of_epochs = len(history["loss"])
    records = []
    for epoch in range(number_of_epochs):
        records.append({
            "Run_Type": run_type,
            "Outer_Test_Year": test_year,
            "Model": model_name,
            "Epoch": epoch + 1,
            "Training_Loss": history["loss"][epoch],
            "Validation_Loss": history["val_loss"][epoch],
            "Training_Accuracy": history["accuracy"][epoch],
            "Validation_Accuracy": history["val_accuracy"][epoch],
            "Learning_Rate": history.get("learning_rate", [np.nan] * number_of_epochs)[epoch]
        })
    return records


# =============================================================================
# 6. LOAD FINAL 240-MONTH DATASET
# =============================================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Final modelling data were not found. Run Script 02 first:\n"
        f"{DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)
data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")

required_columns = [DATE_COL, TARGET_COL, *FEATURES]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    raise ValueError(f"Missing modelling columns: {missing_columns}")

data = data.sort_values(DATE_COL).reset_index(drop=True)
X = data[FEATURES].astype(float)
y_original = data[TARGET_COL].astype(int).to_numpy()
y_zero = y_original - 1
years = data[DATE_COL].dt.year.to_numpy()
months = data[DATE_COL].dt.month.to_numpy()
dates = data[DATE_COL].to_numpy()

if set(np.unique(y_original)) != {1, 2, 3}:
    raise ValueError("The target must contain ecological states 1, 2 and 3.")

sequence_data = make_sequences(
    X.to_numpy(float), y_zero, years, months, dates
)
common_target_indices = sequence_data["target_index"]
common_years = np.sort(np.unique(sequence_data["target_year"]))


# =============================================================================
# 7. LOAD CHECKPOINTS, IF PRESENT
# =============================================================================
prediction_records = safe_read_csv(PREDICTION_CHECKPOINT).to_dict("records")
parameter_records = safe_read_csv(PARAMETER_CHECKPOINT).to_dict("records")
fold_records = safe_read_csv(FOLD_CHECKPOINT).to_dict("records")
history_records = safe_read_csv(HISTORY_CHECKPOINT).to_dict("records")

completed_years = set()
if prediction_records and RESUME_COMPLETED_YEARS:
    prediction_checkpoint_df = pd.DataFrame(prediction_records)
    completed_years = set(
        prediction_checkpoint_df.groupby("Test_Year").filter(
            lambda group: all(
                f"{model.replace(' ', '_').replace('-', '_')}_Prediction" in group.columns
                for model in ALL_MODELS
            )
        )["Test_Year"].unique()
    )


# =============================================================================
# 8. OUTER LEAVE-ONE-YEAR-OUT LOOP
# =============================================================================
for outer_number, test_year in enumerate(common_years, start=1):
    if test_year in completed_years:
        print(f"Outer year {outer_number:02d}/{len(common_years)}: {test_year} — checkpoint found")
        continue

    print(f"\nOuter year {outer_number:02d}/{len(common_years)}: {test_year}")

    # Tree models: all non-test-year monthly rows are training data.
    tree_train_index = np.where(years != test_year)[0]
    sequence_test_index = np.where(sequence_data["target_year"] == test_year)[0]
    raw_test_index = sequence_data["target_index"][sequence_test_index]

    X_tree_train = X.iloc[tree_train_index]
    y_tree_train = y_zero[tree_train_index]
    tree_training_years = years[tree_train_index]
    X_tree_test = X.iloc[raw_test_index]
    y_test_zero = y_zero[raw_test_index]
    y_test_original = y_original[raw_test_index]

    model_probabilities = {}
    model_predictions = {}

    # Training-derived seasonal baseline.
    baseline_probability = seasonal_probabilities(
        months[tree_train_index], y_tree_train, months[raw_test_index]
    )
    model_probabilities["Seasonal Baseline"] = baseline_probability
    model_predictions["Seasonal Baseline"] = np.argmax(
        baseline_probability, axis=1
    ) + 1

    # Bayesian-optimized tree models.
    for model_number, model_name in enumerate(TREE_MODELS):
        optimization_seed = RANDOM_STATE + test_year * 10 + model_number
        best_parameters, inner_score = optimize_tree_model(
            model_name,
            X_tree_train,
            y_tree_train,
            tree_training_years,
            optimization_seed
        )
        final_tree = build_tree_model(
            model_name, best_parameters, RANDOM_STATE + test_year
        )
        final_tree = fit_tree_model(
            final_tree, model_name, X_tree_train, y_tree_train
        )
        probability = final_tree.predict_proba(X_tree_test)
        prediction = np.argmax(probability, axis=1) + 1

        model_probabilities[model_name] = probability
        model_predictions[model_name] = prediction
        parameter_records.append({
            "Run_Type": "Outer_LOYO",
            "Test_Year": test_year,
            "Model": model_name,
            "Best_Inner_Balanced_Accuracy": inner_score,
            "Best_Parameters": json.dumps(best_parameters)
        })
        print(f"  {model_name:<25} inner BA = {inner_score:.3f}")

    voting_probability = np.mean(
        [model_probabilities[name] for name in TREE_MODELS], axis=0
    )
    model_probabilities["Equal Soft Voting"] = voting_probability
    model_predictions["Equal Soft Voting"] = np.argmax(
        voting_probability, axis=1
    ) + 1

    # Strict DL training candidates: target and every input-window month must
    # be outside the held-out year.
    window_contains_test_year = np.any(
        sequence_data["window_years"] == test_year, axis=1
    )
    dl_candidate_index = np.where(
        (sequence_data["target_year"] != test_year)
        & ~window_contains_test_year
    )[0]
    dl_fit_index, dl_valid_index, validation_years = (
        select_dl_fit_validation_indices(
            sequence_data, dl_candidate_index,
            RANDOM_STATE + test_year
        )
    )

    X_dl_fit, X_dl_valid, X_dl_test, dl_scaler = scale_sequence_sets(
        sequence_data["X"][dl_fit_index],
        sequence_data["X"][dl_valid_index],
        sequence_data["X"][sequence_test_index]
    )
    y_dl_fit = sequence_data["y"][dl_fit_index]
    y_dl_valid = sequence_data["y"][dl_valid_index]

    for dl_number, model_name in enumerate(["TCN", "CNN-LSTM"]):
        dl_model, history = train_dl_model(
            model_name,
            X_dl_fit, y_dl_fit,
            X_dl_valid, y_dl_valid,
            RANDOM_STATE + test_year * 10 + dl_number
        )
        probability = dl_model.predict(X_dl_test, verbose=0)
        prediction = np.argmax(probability, axis=1) + 1
        model_probabilities[model_name] = probability
        model_predictions[model_name] = prediction
        history_records.extend(history_to_records(
            history, model_name, "Outer_LOYO", test_year
        ))
        print(
            f"  {model_name:<25} epochs = {len(history['loss']):3d}; "
            f"validation years = {','.join(map(str, validation_years))}"
        )
        del dl_model
        tf.keras.backend.clear_session()
        gc.collect()

    # Save one complete prediction row per common eligible month.
    year_prediction_records = []
    for local_index, raw_index in enumerate(raw_test_index):
        record = {
            "Months": data.loc[raw_index, DATE_COL].strftime("%Y-%m-%d"),
            "Test_Year": int(test_year),
            "True_State": int(y_original[raw_index])
        }
        for model_name in ALL_MODELS:
            prefix = model_name.replace(" ", "_").replace("-", "_")
            record[f"{prefix}_Prediction"] = int(
                model_predictions[model_name][local_index]
            )
            for state_index in range(3):
                record[f"{prefix}_P_State_{state_index + 1}"] = float(
                    model_probabilities[model_name][local_index, state_index]
                )
        year_prediction_records.append(record)

    prediction_records.extend(year_prediction_records)

    # Year-specific diagnostic scores; pooled scores remain the final result.
    for model_name in ALL_MODELS:
        fold_records.append({
            "Test_Year": test_year,
            "Test_Months": len(raw_test_index),
            "Model": model_name,
            "Accuracy": accuracy_score(
                y_test_original, model_predictions[model_name]
            ),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test_original, model_predictions[model_name]
            ),
            "Macro_F1": f1_score(
                y_test_original, model_predictions[model_name],
                average="macro", zero_division=0
            )
        })

    # A checkpoint is written only after the entire held-out year is complete.
    pd.DataFrame(prediction_records).to_csv(PREDICTION_CHECKPOINT, index=False)
    pd.DataFrame(parameter_records).to_csv(PARAMETER_CHECKPOINT, index=False)
    pd.DataFrame(fold_records).to_csv(FOLD_CHECKPOINT, index=False)
    pd.DataFrame(history_records).to_csv(HISTORY_CHECKPOINT, index=False)


# =============================================================================
# 9. FINAL POOLED HELD-OUT TABLES
# =============================================================================
predictions = pd.DataFrame(prediction_records)
predictions["Months"] = pd.to_datetime(predictions["Months"])
predictions = predictions.sort_values("Months").drop_duplicates(
    "Months", keep="last"
).reset_index(drop=True)

if len(predictions) != len(common_target_indices):
    raise RuntimeError(
        f"Expected {len(common_target_indices)} common predictions, "
        f"but found {len(predictions)}."
    )

performance_records = []
per_state_records = []
confusion_records = []
y_true = predictions["True_State"].to_numpy(int)

for model_name in ALL_MODELS:
    prefix = model_name.replace(" ", "_").replace("-", "_")
    y_pred = predictions[f"{prefix}_Prediction"].to_numpy(int)
    probability = predictions[
        [f"{prefix}_P_State_{state}" for state in [1, 2, 3]]
    ].to_numpy(float)

    performance_records.append(
        pooled_metrics(model_name, y_true, y_pred, probability)
    )

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    total = cm.sum()
    one_hot = label_binarize(y_true, classes=[1, 2, 3])
    for state_index, state in enumerate([1, 2, 3]):
        tp = cm[state_index, state_index]
        fn = cm[state_index, :].sum() - tp
        fp = cm[:, state_index].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        state_f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity else 0
        )
        per_state_records.append({
            "Model": model_name,
            "State": state,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1": state_f1,
            "AUROC_OVR": roc_auc_score(
                one_hot[:, state_index], probability[:, state_index]
            ),
            "AUPRC_OVR": average_precision_score(
                one_hot[:, state_index], probability[:, state_index]
            )
        })
        for predicted_state in [1, 2, 3]:
            confusion_records.append({
                "Model": model_name,
                "True_State": state,
                "Predicted_State": predicted_state,
                "Count": int(cm[state - 1, predicted_state - 1])
            })

performance = pd.DataFrame(performance_records).sort_values(
    "Balanced_Accuracy", ascending=False
).reset_index(drop=True)
per_state_metrics = pd.DataFrame(per_state_records)
confusion_table = pd.DataFrame(confusion_records)
fold_scores = pd.DataFrame(fold_records)
training_history = pd.DataFrame(history_records)
best_parameters = pd.DataFrame(parameter_records)

predictions.to_csv(RESULT_DIR / "01_LOYO_Held_Out_Predictions.csv", index=False)
performance.to_csv(RESULT_DIR / "02_Pooled_Model_Performance.csv", index=False)
per_state_metrics.to_csv(RESULT_DIR / "03_Per_State_Metrics.csv", index=False)
confusion_table.to_csv(RESULT_DIR / "04_Confusion_Matrix_Counts.csv", index=False)
fold_scores.to_csv(RESULT_DIR / "05_Outer_Year_Scores.csv", index=False)
training_history.to_csv(RESULT_DIR / "06_DL_Training_History.csv", index=False)
best_parameters.to_csv(RESULT_DIR / "07_Optuna_Best_Parameters.csv", index=False)


# =============================================================================
# 10. FIT AND SAVE FULL-DATA TREE MODELS
# =============================================================================
final_tree_models = {}
final_parameter_records = []

for model_number, model_name in enumerate(TREE_MODELS):
    print(f"\nFinal full-data optimization: {model_name}")
    best_parameters_full, inner_score_full = optimize_tree_model(
        model_name, X, y_zero, years,
        RANDOM_STATE + 9000 + model_number
    )
    final_model = build_tree_model(
        model_name, best_parameters_full,
        RANDOM_STATE + 9000 + model_number
    )
    final_model = fit_tree_model(final_model, model_name, X, y_zero)
    final_tree_models[model_name] = final_model
    final_parameter_records.append({
        "Run_Type": "Final_Full_Data",
        "Test_Year": "NA",
        "Model": model_name,
        "Best_Inner_Balanced_Accuracy": inner_score_full,
        "Best_Parameters": json.dumps(best_parameters_full)
    })

    if model_name == "CatBoost":
        final_model.save_model(str(MODEL_DIR / "Final_CatBoost.cbm"))
    elif model_name == "XGBoost":
        final_model.save_model(str(MODEL_DIR / "Final_XGBoost.json"))
    else:
        joblib.dump(final_model, MODEL_DIR / "Final_HistGradientBoosting.pkl")


# =============================================================================
# 11. FIT AND SAVE FULL-DATA DL MODELS
# =============================================================================
all_sequence_indices = np.arange(len(sequence_data["X"]))
final_fit_index, final_valid_index, final_validation_years = (
    select_dl_fit_validation_indices(
        sequence_data, all_sequence_indices, RANDOM_STATE + 9500
    )
)
X_final_fit, X_final_valid, _, final_scaler = scale_sequence_sets(
    sequence_data["X"][final_fit_index],
    sequence_data["X"][final_valid_index],
    sequence_data["X"][final_valid_index]
)
y_final_fit = sequence_data["y"][final_fit_index]
y_final_valid = sequence_data["y"][final_valid_index]

joblib.dump(final_scaler, MODEL_DIR / "Final_DL_StandardScaler.pkl")

for model_number, model_name in enumerate(["TCN", "CNN-LSTM"]):
    final_dl, final_history = train_dl_model(
        model_name,
        X_final_fit, y_final_fit,
        X_final_valid, y_final_valid,
        RANDOM_STATE + 9600 + model_number
    )
    history_records.extend(history_to_records(
        final_history, model_name, "Final_Full_Data", "NA"
    ))
    final_dl.save(
        MODEL_DIR / (
            "Final_TCN.keras" if model_name == "TCN"
            else "Final_CNN_LSTM.keras"
        )
    )
    del final_dl
    tf.keras.backend.clear_session()
    gc.collect()

pd.DataFrame(history_records).to_csv(
    RESULT_DIR / "06_DL_Training_History.csv", index=False
)
pd.concat([
    best_parameters,
    pd.DataFrame(final_parameter_records)
], ignore_index=True).to_csv(
    RESULT_DIR / "07_Optuna_Best_Parameters.csv", index=False
)

soft_voting_metadata = {
    "name": "Equal Soft Voting",
    "members": TREE_MODELS,
    "weights": [1 / 3, 1 / 3, 1 / 3],
    "combination": "Arithmetic mean of class probabilities"
}
(MODEL_DIR / "Equal_Soft_Voting_Metadata.json").write_text(
    json.dumps(soft_voting_metadata, indent=2), encoding="utf-8"
)


# =============================================================================
# 12. SETTINGS AND FINAL SUMMARY
# =============================================================================
settings = {
    "data_file": str(DATA_FILE),
    "features": FEATURES,
    "final_models": FINAL_MODELS,
    "reference_model": "Seasonal Baseline",
    "outer_validation": "Leave-one-year-out",
    "tree_optimization": "Optuna TPE with grouped inner CV",
    "optuna_trials": N_TRIALS,
    "inner_folds": INNER_FOLDS,
    "lookback_months": LOOKBACK,
    "common_held_out_observations": len(predictions),
    "maximum_dl_epochs": MAX_EPOCHS,
    "dl_batch_size": BATCH_SIZE,
    "random_state": RANDOM_STATE,
    "tree_gpu_enabled": USE_GPU_FOR_TREES,
    "final_dl_validation_years": list(map(int, final_validation_years))
}
(RESULT_DIR / "00_Final_Model_Settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8"
)

print("\n" + "=" * 132)
print("FINAL OPTIMIZED SIX-MODEL POOLED HELD-OUT PERFORMANCE")
print("=" * 132)
print(performance.round(3).to_string(index=False))
print(f"\nCommon evaluated months: {len(predictions)}")
print(f"First evaluated month  : {predictions['Months'].min().date()}")
print(f"Last evaluated month   : {predictions['Months'].max().date()}")
print("\nResults saved in:")
print(RESULT_DIR)
print("\nModels saved in:")
print(MODEL_DIR)


# In[4]:


"""Generate final evaluation and curve tables from saved LOYO predictions.

This script performs no model fitting. It reads the common held-out prediction
table created by Script 03 and exports plotting-ready numerical results for
ROC, PR, calibration, threshold trade-offs, DCA, confidence, temporal
agreement, and year-grouped bootstrap uncertainty.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    log_loss, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# 1. SETTINGS AND FOLDERS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT / "Results" / "03_Final_Models"
    / "01_LOYO_Held_Out_Predictions.csv"
)
OUTPUT_DIR = ROOT / "Results" / "04_Evaluation_Curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "CatBoost",
    "Equal Soft Voting",
    "CNN-LSTM",
    "TCN",
    "XGBoost",
    "HistGradientBoosting",
    "Seasonal Baseline"
]

STATES = [1, 2, 3]
CALIBRATION_BINS = 10
THRESHOLDS = np.linspace(0.01, 0.99, 99)
BOOTSTRAP_RUNS = 2000
RANDOM_STATE = 42


# =============================================================================
# 2. HELPERS
# =============================================================================
def model_prefix(model_name):
    return model_name.replace(" ", "_").replace("-", "_")


def probability_columns(model_name):
    prefix = model_prefix(model_name)
    return [f"{prefix}_P_State_{state}" for state in STATES]


def prediction_column(model_name):
    return f"{model_prefix(model_name)}_Prediction"


def multiclass_brier(y_true, probability):
    one_hot = label_binarize(y_true, classes=STATES)
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=STATES)
    total = cm.sum()
    values = []
    for index in range(len(STATES)):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = total - tp - fn - fp
        values.append(safe_divide(tn, tn + fp))
    return float(np.nanmean(values))


def calculate_metrics(y_true, y_pred, probability):
    one_hot = label_binarize(y_true, classes=STATES)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Macro_Specificity": macro_specificity(y_true, y_pred),
        "Macro_AUROC": roc_auc_score(
            one_hot, probability, average="macro", multi_class="ovr"
        ),
        "Macro_AUPRC": average_precision_score(
            one_hot, probability, average="macro"
        ),
        "Log_Loss": log_loss(y_true, probability, labels=STATES),
        "Multiclass_Brier_Score": multiclass_brier(y_true, probability)
    }


def quantile_calibration_rows(model_name, state, y_binary, probability):
    """Create quantile calibration bins and return rows plus ECE."""
    number_of_bins = min(CALIBRATION_BINS, len(np.unique(probability)))
    bin_id = pd.qcut(
        probability,
        q=number_of_bins,
        labels=False,
        duplicates="drop"
    )
    temporary = pd.DataFrame({
        "Observed": y_binary,
        "Probability": probability,
        "Bin": bin_id
    }).dropna(subset=["Bin"])

    rows = []
    total = len(temporary)
    for order, (_, group) in enumerate(temporary.groupby("Bin"), start=1):
        mean_probability = group["Probability"].mean()
        observed_fraction = group["Observed"].mean()
        gap = abs(observed_fraction - mean_probability)
        rows.append({
            "Model": model_name,
            "State": state,
            "Bin": order,
            "Count": len(group),
            "Probability_Minimum": group["Probability"].min(),
            "Probability_Maximum": group["Probability"].max(),
            "Mean_Predicted_Probability": mean_probability,
            "Observed_Fraction": observed_fraction,
            "Absolute_Calibration_Gap": gap
        })
    ece = sum(row["Count"] / total * row["Absolute_Calibration_Gap"]
              for row in rows)
    return rows, float(ece)


def class_threshold_metrics(y_binary, probability, threshold):
    prediction = probability >= threshold
    tp = int(np.sum((y_binary == 1) & prediction))
    fn = int(np.sum((y_binary == 1) & ~prediction))
    fp = int(np.sum((y_binary == 0) & prediction))
    tn = int(np.sum((y_binary == 0) & ~prediction))

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1_value = (
        safe_divide(2 * precision * sensitivity, precision + sensitivity)
        if np.isfinite(precision) and np.isfinite(sensitivity)
        else np.nan
    )
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1_value,
        "Youden_J": sensitivity + specificity - 1,
        "Predicted_Positive_Rate": prediction.mean()
    }


def bootstrap_sample_indices(year_values, rng):
    unique_years = np.unique(year_values)
    sampled_years = rng.choice(
        unique_years, size=len(unique_years), replace=True
    )
    return np.concatenate([
        np.where(year_values == year)[0] for year in sampled_years
    ])


# =============================================================================
# 3. LOAD AND VALIDATE HELD-OUT PREDICTIONS
# =============================================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Held-out prediction file was not found. Run Script 03 first:\n"
        f"{INPUT_FILE}"
    )

predictions = pd.read_csv(INPUT_FILE)
predictions["Months"] = pd.to_datetime(
    predictions["Months"], errors="coerce"
)

if predictions["Months"].isna().any():
    raise ValueError("Invalid prediction dates were found.")
if predictions["Months"].duplicated().any():
    raise ValueError("Duplicate prediction months were found.")
if len(predictions) != 235:
    raise ValueError(
        f"Expected 235 common held-out months; found {len(predictions)}."
    )

required_columns = ["Months", "Test_Year", "True_State"]
for model_name in MODELS:
    required_columns.append(prediction_column(model_name))
    required_columns.extend(probability_columns(model_name))

missing_columns = [
    column for column in required_columns
    if column not in predictions.columns
]
if missing_columns:
    raise ValueError(f"Missing prediction columns: {missing_columns}")

predictions = predictions.sort_values("Months").reset_index(drop=True)
y_true = predictions["True_State"].to_numpy(int)
years = predictions["Test_Year"].to_numpy(int)

if set(np.unique(y_true)) != {1, 2, 3}:
    raise ValueError("True states must contain 1, 2 and 3.")

for model_name in MODELS:
    probability = predictions[probability_columns(model_name)].to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError(f"Non-finite probabilities found for {model_name}.")
    if np.any(probability < -1e-8) or np.any(probability > 1 + 1e-8):
        raise ValueError(f"Probabilities outside [0, 1] found for {model_name}.")
    if not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"Probabilities do not sum to one for {model_name}.")


# =============================================================================
# 4. ROC AND PRECISION-RECALL CURVE POINTS
# =============================================================================
roc_records = []
pr_records = []
one_hot_true = label_binarize(y_true, classes=STATES)

for model_name in MODELS:
    probability = predictions[probability_columns(model_name)].to_numpy(float)
    for state_index, state in enumerate(STATES):
        y_binary = one_hot_true[:, state_index]
        state_probability = probability[:, state_index]

        fpr, tpr, roc_thresholds = roc_curve(y_binary, state_probability)
        state_auroc = roc_auc_score(y_binary, state_probability)
        for point, (false_positive_rate, true_positive_rate, threshold) in enumerate(
            zip(fpr, tpr, roc_thresholds), start=1
        ):
            roc_records.append({
                "Model": model_name,
                "State": state,
                "Point": point,
                "False_Positive_Rate": false_positive_rate,
                "True_Positive_Rate": true_positive_rate,
                "Threshold": threshold if np.isfinite(threshold) else np.nan,
                "State_AUROC": state_auroc
            })

        precision_values, recall_values, pr_thresholds = precision_recall_curve(
            y_binary, state_probability
        )
        state_auprc = average_precision_score(y_binary, state_probability)
        prevalence = y_binary.mean()
        for point in range(len(precision_values)):
            pr_records.append({
                "Model": model_name,
                "State": state,
                "Point": point + 1,
                "Recall": recall_values[point],
                "Precision": precision_values[point],
                "Threshold": (
                    pr_thresholds[point]
                    if point < len(pr_thresholds) else np.nan
                ),
                "State_AUPRC": state_auprc,
                "State_Prevalence": prevalence
            })

pd.DataFrame(roc_records).to_csv(
    OUTPUT_DIR / "01_ROC_Curve_Points.csv", index=False
)
pd.DataFrame(pr_records).to_csv(
    OUTPUT_DIR / "02_PR_Curve_Points.csv", index=False
)


# =============================================================================
# 5. CALIBRATION AND PROBABILITY-QUALITY METRICS
# =============================================================================
calibration_records = []
quality_records = []

for model_name in MODELS:
    y_pred = predictions[prediction_column(model_name)].to_numpy(int)
    probability = predictions[probability_columns(model_name)].to_numpy(float)
    metrics = calculate_metrics(y_true, y_pred, probability)
    state_ece_values = []

    for state_index, state in enumerate(STATES):
        y_binary = one_hot_true[:, state_index]
        state_probability = probability[:, state_index]
        rows, state_ece = quantile_calibration_rows(
            model_name, state, y_binary, state_probability
        )
        calibration_records.extend(rows)
        state_ece_values.append(state_ece)
        quality_records.append({
            "Model": model_name,
            "Level": "State",
            "State": state,
            "AUROC": roc_auc_score(y_binary, state_probability),
            "AUPRC": average_precision_score(y_binary, state_probability),
            "Binary_Brier_Score": np.mean(
                (state_probability - y_binary) ** 2
            ),
            "Expected_Calibration_Error": state_ece,
            "Log_Loss": np.nan,
            "Multiclass_Brier_Score": np.nan,
            "Macro_AUROC": np.nan,
            "Macro_AUPRC": np.nan
        })

    quality_records.append({
        "Model": model_name,
        "Level": "Overall",
        "State": "All",
        "AUROC": np.nan,
        "AUPRC": np.nan,
        "Binary_Brier_Score": np.nan,
        "Expected_Calibration_Error": np.mean(state_ece_values),
        "Log_Loss": metrics["Log_Loss"],
        "Multiclass_Brier_Score": metrics["Multiclass_Brier_Score"],
        "Macro_AUROC": metrics["Macro_AUROC"],
        "Macro_AUPRC": metrics["Macro_AUPRC"]
    })

pd.DataFrame(calibration_records).to_csv(
    OUTPUT_DIR / "03_Calibration_Curve_Points.csv", index=False
)
pd.DataFrame(quality_records).to_csv(
    OUTPUT_DIR / "04_Probability_Quality_Metrics.csv", index=False
)


# =============================================================================
# 6. THRESHOLD TRADE-OFF AND DCA/NET BENEFIT
# =============================================================================
tradeoff_records = []
dca_records = []

for model_name in MODELS:
    probability = predictions[probability_columns(model_name)].to_numpy(float)
    for state_index, state in enumerate(STATES):
        y_binary = one_hot_true[:, state_index]
        state_probability = probability[:, state_index]
        prevalence = y_binary.mean()
        number_of_observations = len(y_binary)

        for threshold in THRESHOLDS:
            threshold_metrics = class_threshold_metrics(
                y_binary, state_probability, threshold
            )
            tradeoff_records.append({
                "Model": model_name,
                "State": state,
                "Threshold": threshold,
                **threshold_metrics
            })

            odds = threshold / (1 - threshold)
            model_net_benefit = (
                threshold_metrics["TP"] / number_of_observations
                - threshold_metrics["FP"] / number_of_observations * odds
            )
            treat_all_net_benefit = prevalence - (1 - prevalence) * odds
            dca_records.append({
                "Model": model_name,
                "State": state,
                "Threshold": threshold,
                "Model_Net_Benefit": model_net_benefit,
                "Treat_All_Net_Benefit": treat_all_net_benefit,
                "Treat_None_Net_Benefit": 0.0,
                "State_Prevalence": prevalence,
                "Interpretation": "One-versus-rest ecological decision utility"
            })

pd.DataFrame(tradeoff_records).to_csv(
    OUTPUT_DIR / "05_Threshold_Tradeoff.csv", index=False
)
pd.DataFrame(dca_records).to_csv(
    OUTPUT_DIR / "06_DCA_Net_Benefit.csv", index=False
)


# =============================================================================
# 7. PREDICTION CONFIDENCE AND TEMPORAL AGREEMENT
# =============================================================================
confidence_records = []

for model_name in MODELS:
    y_pred = predictions[prediction_column(model_name)].to_numpy(int)
    probability = predictions[probability_columns(model_name)].to_numpy(float)
    sorted_probability = np.sort(probability, axis=1)
    maximum_probability = sorted_probability[:, -1]
    probability_margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1)
    normalized_entropy = entropy / np.log(len(STATES))
    true_probability = probability[np.arange(len(y_true)), y_true - 1]

    for index in range(len(predictions)):
        confidence_records.append({
            "Months": predictions.loc[index, "Months"],
            "Year": predictions.loc[index, "Months"].year,
            "Month_Number": predictions.loc[index, "Months"].month,
            "Model": model_name,
            "True_State": y_true[index],
            "Predicted_State": y_pred[index],
            "Correct": y_true[index] == y_pred[index],
            "Maximum_Probability": maximum_probability[index],
            "True_State_Probability": true_probability[index],
            "Top_Two_Probability_Margin": probability_margin[index],
            "Entropy": entropy[index],
            "Normalized_Entropy": normalized_entropy[index]
        })

confidence_data = pd.DataFrame(confidence_records)
confidence_data.to_csv(
    OUTPUT_DIR / "07_Prediction_Confidence.csv", index=False
)

confidence_summary_records = []
for model_name, group in confidence_data.groupby("Model", sort=False):
    correct_group = group[group["Correct"]]
    incorrect_group = group[~group["Correct"]]
    confidence_summary_records.append({
        "Model": model_name,
        "Mean_Maximum_Probability": group["Maximum_Probability"].mean(),
        "Median_Maximum_Probability": group["Maximum_Probability"].median(),
        "Mean_True_State_Probability": group["True_State_Probability"].mean(),
        "Mean_Probability_Margin": group["Top_Two_Probability_Margin"].mean(),
        "Mean_Normalized_Entropy": group["Normalized_Entropy"].mean(),
        "Correct_Mean_Confidence": correct_group["Maximum_Probability"].mean(),
        "Incorrect_Mean_Confidence": incorrect_group["Maximum_Probability"].mean(),
        "Below_0.50_Percent": (group["Maximum_Probability"] < 0.50).mean() * 100,
        "Below_0.60_Percent": (group["Maximum_Probability"] < 0.60).mean() * 100,
        "Below_0.70_Percent": (group["Maximum_Probability"] < 0.70).mean() * 100
    })

pd.DataFrame(confidence_summary_records).to_csv(
    OUTPUT_DIR / "08_Confidence_Summary.csv", index=False
)

# This tidy table can directly produce actual-versus-predicted timelines.
confidence_data[[
    "Months", "Year", "Month_Number", "Model",
    "True_State", "Predicted_State", "Correct",
    "Maximum_Probability"
]].to_csv(
    OUTPUT_DIR / "09_Temporal_Agreement.csv", index=False
)


# =============================================================================
# 8. YEARLY PERFORMANCE
# =============================================================================
yearly_records = []
for year, year_group in predictions.groupby("Test_Year"):
    year_true = year_group["True_State"].to_numpy(int)
    for model_name in MODELS:
        year_pred = year_group[prediction_column(model_name)].to_numpy(int)
        recalls = recall_score(
            year_true, year_pred, labels=STATES,
            average=None, zero_division=0
        )
        yearly_records.append({
            "Year": int(year),
            "Model": model_name,
            "Number_of_Months": len(year_group),
            "Accuracy": accuracy_score(year_true, year_pred),
            "Balanced_Accuracy": recall_score(
                year_true, year_pred, labels=STATES,
                average="macro", zero_division=0
            ),
            "Macro_F1": f1_score(
                year_true, year_pred, labels=STATES,
                average="macro", zero_division=0
            ),
            "State_1_Recall": recalls[0],
            "State_2_Recall": recalls[1],
            "State_3_Recall": recalls[2]
        })

pd.DataFrame(yearly_records).to_csv(
    OUTPUT_DIR / "10_Yearly_Performance.csv", index=False
)


# =============================================================================
# 9. YEAR-GROUPED BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
rng = np.random.default_rng(RANDOM_STATE)
bootstrap_records = []
paired_records = []

cat_prediction = predictions[prediction_column("CatBoost")].to_numpy(int)
cat_probability = predictions[probability_columns("CatBoost")].to_numpy(float)
baseline_prediction = predictions[
    prediction_column("Seasonal Baseline")
].to_numpy(int)
baseline_probability = predictions[
    probability_columns("Seasonal Baseline")
].to_numpy(float)

for bootstrap_run in range(1, BOOTSTRAP_RUNS + 1):
    sample_index = bootstrap_sample_indices(years, rng)
    sample_true = y_true[sample_index]

    run_model_metrics = {}
    for model_name in MODELS:
        sample_prediction = predictions[
            prediction_column(model_name)
        ].to_numpy(int)[sample_index]
        sample_probability = predictions[
            probability_columns(model_name)
        ].to_numpy(float)[sample_index]
        metrics = calculate_metrics(
            sample_true, sample_prediction, sample_probability
        )
        run_model_metrics[model_name] = metrics
        for metric_name, value in metrics.items():
            bootstrap_records.append({
                "Bootstrap_Run": bootstrap_run,
                "Model": model_name,
                "Metric": metric_name,
                "Value": value
            })

    cat_metrics = run_model_metrics["CatBoost"]
    baseline_metrics = run_model_metrics["Seasonal Baseline"]
    for metric_name in cat_metrics:
        lower_is_better = metric_name in {
            "Log_Loss", "Multiclass_Brier_Score"
        }
        improvement = (
            baseline_metrics[metric_name] - cat_metrics[metric_name]
            if lower_is_better
            else cat_metrics[metric_name] - baseline_metrics[metric_name]
        )
        paired_records.append({
            "Bootstrap_Run": bootstrap_run,
            "Metric": metric_name,
            "Improvement": improvement,
            "Positive_Means": (
                "Lower CatBoost value is better"
                if lower_is_better
                else "Higher CatBoost value is better"
            )
        })

bootstrap_data = pd.DataFrame(bootstrap_records)
bootstrap_summary = (
    bootstrap_data.groupby(["Model", "Metric"])["Value"]
    .agg(
        Bootstrap_Mean="mean",
        Bootstrap_SD="std",
        CI_2_5=lambda values: np.quantile(values, 0.025),
        CI_97_5=lambda values: np.quantile(values, 0.975)
    )
    .reset_index()
)

# Add the original pooled estimate beside each interval.
original_estimates = []
for model_name in MODELS:
    original_metrics = calculate_metrics(
        y_true,
        predictions[prediction_column(model_name)].to_numpy(int),
        predictions[probability_columns(model_name)].to_numpy(float)
    )
    for metric_name, value in original_metrics.items():
        original_estimates.append({
            "Model": model_name,
            "Metric": metric_name,
            "Original_Estimate": value
        })

bootstrap_summary = bootstrap_summary.merge(
    pd.DataFrame(original_estimates),
    on=["Model", "Metric"],
    how="left"
)
bootstrap_summary.to_csv(
    OUTPUT_DIR / "11_Bootstrap_Confidence_Intervals.csv", index=False
)

paired_data = pd.DataFrame(paired_records)
paired_summary_records = []
for metric_name, group in paired_data.groupby("Metric"):
    differences = group["Improvement"].to_numpy(float)
    probability_positive = np.mean(differences > 0)
    two_sided_p = min(
        1.0,
        2 * min(np.mean(differences <= 0), np.mean(differences >= 0))
    )
    paired_summary_records.append({
        "Metric": metric_name,
        "Mean_Improvement": differences.mean(),
        "Median_Improvement": np.median(differences),
        "CI_2_5": np.quantile(differences, 0.025),
        "CI_97_5": np.quantile(differences, 0.975),
        "Probability_Improvement_Positive": probability_positive,
        "Bootstrap_Two_Sided_P": two_sided_p,
        "Positive_Means": group["Positive_Means"].iloc[0]
    })

pd.DataFrame(paired_summary_records).to_csv(
    OUTPUT_DIR / "12_CatBoost_vs_Baseline_Bootstrap.csv", index=False
)
paired_data.to_csv(
    OUTPUT_DIR / "12b_CatBoost_vs_Baseline_Bootstrap_Runs.csv", index=False
)


# =============================================================================
# 10. SETTINGS AND COMPACT SUMMARY
# =============================================================================
settings = {
    "input_file": str(INPUT_FILE),
    "models": MODELS,
    "states": STATES,
    "observations": len(predictions),
    "calibration_bins": CALIBRATION_BINS,
    "calibration_strategy": "Quantile bins",
    "threshold_minimum": float(THRESHOLDS.min()),
    "threshold_maximum": float(THRESHOLDS.max()),
    "threshold_count": len(THRESHOLDS),
    "bootstrap_unit": "Year",
    "bootstrap_runs": BOOTSTRAP_RUNS,
    "random_state": RANDOM_STATE,
    "dca_interpretation": "One-versus-rest ecological decision utility"
}
(OUTPUT_DIR / "00_Evaluation_Settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8"
)

quality_overall = pd.DataFrame(quality_records)
quality_overall = quality_overall[quality_overall["Level"] == "Overall"]
paired_summary = pd.DataFrame(paired_summary_records)

print("\n" + "=" * 100)
print("FINAL HELD-OUT EVALUATION OUTPUTS COMPLETED")
print("=" * 100)
print(f"Held-out observations       : {len(predictions)}")
print(f"Models                      : {len(MODELS)}")
print(f"States                      : {len(STATES)}")
print(f"Year-grouped bootstrap runs : {BOOTSTRAP_RUNS}")
print("\nOverall probability quality:")
print(quality_overall[[
    "Model", "Expected_Calibration_Error", "Log_Loss",
    "Multiclass_Brier_Score", "Macro_AUROC", "Macro_AUPRC"
]].round(3).to_string(index=False))
print("\nCatBoost versus seasonal baseline:")
print(paired_summary[[
    "Metric", "Mean_Improvement", "CI_2_5", "CI_97_5",
    "Probability_Improvement_Positive"
]].round(4).to_string(index=False))
print("\nResults saved in:")
print(OUTPUT_DIR)


# In[5]:


"""Create final CatBoost SHAP tables without retraining any model.

This script loads the full-data CatBoost model saved by Script 03, calculates
native multiclass SHAP values for the 14 current environmental predictors, and
exports numerical files for the manuscript figures. Predictive performance
must be reported from the held-out LOYO results, not from this full-data model.
"""

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr


# =============================================================================
# 1. SETTINGS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Data" / "04_Final_Current_Environmental_Modeling_Data.csv"
MODEL_FILE = ROOT / "Models" / "Final_Six_Models" / "Final_CatBoost.cbm"
OUTPUT_DIR = ROOT / "Results" / "05_CatBoost_SHAP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "Months"
TARGET_COL = "Ecological_State"
FEATURES = [
    "SST", "NO3", "PO4", "SPCo2", "MLD", "SSS", "SSH", "PAR",
    "PDO", "NINO_3.4", "WPI",
    "MHW_MeanInt", "MHW_MaxInt", "MHW_CumInt"
]
STATE_NAMES = {
    1: "Diatom-associated early-year state",
    2: "Mixed-prokaryote mid-year state",
    3: "Prokaryote-dominated late-year state"
}


# =============================================================================
# 2. LOAD AND VALIDATE DATA
# =============================================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Modelling dataset not found:\n{DATA_FILE}")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Saved CatBoost model not found:\n{MODEL_FILE}")

data = pd.read_csv(DATA_FILE)
required_columns = [DATE_COL, TARGET_COL, *FEATURES]
missing_columns = [column for column in required_columns if column not in data]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
if data[DATE_COL].isna().any():
    raise ValueError("Invalid dates were found in the modelling dataset.")
if data[DATE_COL].duplicated().any():
    raise ValueError("Duplicate months were found in the modelling dataset.")

data = data.sort_values(DATE_COL).reset_index(drop=True)
X = data[FEATURES].astype(float)
y_state = data[TARGET_COL].astype(int).to_numpy()

if len(data) != 240:
    raise ValueError(f"Expected 240 observations, but found {len(data)}.")
if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Missing or infinite environmental values were found.")
if set(np.unique(y_state)) != {1, 2, 3}:
    raise ValueError("Ecological states must be coded as 1, 2 and 3.")


# =============================================================================
# 3. LOAD FINAL MODEL AND CALCULATE NATIVE MULTICLASS SHAP VALUES
# =============================================================================
model = CatBoostClassifier()
model.load_model(str(MODEL_FILE))

model_features = list(model.feature_names_)
if model_features and model_features != FEATURES:
    raise ValueError(
        "Saved CatBoost feature order does not match the final feature list.\n"
        f"Model: {model_features}\nExpected: {FEATURES}"
    )

pool = Pool(X, feature_names=FEATURES)
probability = np.asarray(model.predict_proba(pool), dtype=float)
raw_prediction = np.asarray(
    model.predict(pool, prediction_type="RawFormulaVal"), dtype=float
)
raw_shap = np.asarray(
    model.get_feature_importance(pool, type="ShapValues"), dtype=float
)

model_classes = np.asarray(model.classes_).astype(int)
n_samples = len(X)
n_features = len(FEATURES)
n_classes = len(model_classes)

if n_classes != 3:
    raise ValueError(f"Expected three model classes, but found {model_classes}.")

# Script 03 trained CatBoost with zero-based classes (0, 1, 2). Convert them
# back to ecological states (1, 2, 3) for every output table.
if set(model_classes) == {0, 1, 2}:
    explained_states = model_classes + 1
elif set(model_classes) == {1, 2, 3}:
    explained_states = model_classes.copy()
else:
    raise ValueError(f"Unexpected CatBoost class labels: {model_classes}")

# Standard CatBoost ordering is samples x classes x (features + base value).
# The second branch supports versions returning the last two axes reversed.
if raw_shap.shape == (n_samples, n_classes, n_features + 1):
    shap_values = raw_shap[:, :, :-1]
    expected_values = raw_shap[:, :, -1]
elif raw_shap.shape == (n_samples, n_features + 1, n_classes):
    shap_values = np.transpose(raw_shap[:, :-1, :], (0, 2, 1))
    expected_values = raw_shap[:, -1, :]
else:
    raise ValueError(f"Unexpected CatBoost SHAP shape: {raw_shap.shape}")

if raw_prediction.ndim == 1:
    raw_prediction = raw_prediction[:, None]
additive_prediction = expected_values + shap_values.sum(axis=2)
maximum_additivity_error = float(
    np.max(np.abs(additive_prediction - raw_prediction))
)
if maximum_additivity_error > 1e-5:
    raise ValueError(
        "SHAP additivity check failed. Maximum absolute error: "
        f"{maximum_additivity_error:.8f}"
    )


# =============================================================================
# 4. OVERALL AND STATE-SPECIFIC IMPORTANCE
# =============================================================================
overall_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_Absolute_SHAP": np.mean(np.abs(shap_values), axis=(0, 1))
}).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
overall_importance.insert(0, "Rank", np.arange(1, n_features + 1))
overall_importance["Relative_Importance_Percent"] = (
    100 * overall_importance["Mean_Absolute_SHAP"]
    / overall_importance["Mean_Absolute_SHAP"].sum()
)

state_importance_rows = []
direction_rows = []

for class_index, state in enumerate(explained_states):
    for feature_index, feature in enumerate(FEATURES):
        state_shap = shap_values[:, class_index, feature_index]
        mean_absolute = float(np.mean(np.abs(state_shap)))
        correlation, p_value = spearmanr(X[feature].to_numpy(), state_shap)

        state_importance_rows.append({
            "Ecological_State": int(state),
            "State_Name": STATE_NAMES[int(state)],
            "Feature": feature,
            "Mean_Absolute_SHAP": mean_absolute
        })
        direction_rows.append({
            "Ecological_State": int(state),
            "State_Name": STATE_NAMES[int(state)],
            "Feature": feature,
            "Spearman_Rho_Feature_vs_SHAP": float(correlation),
            "Spearman_P_Value": float(p_value),
            "Higher_Value_Association": (
                "Supports state" if correlation > 0
                else "Opposes state" if correlation < 0
                else "No monotonic direction"
            )
        })

state_importance = pd.DataFrame(state_importance_rows)
state_importance["State_Rank"] = (
    state_importance.groupby("Ecological_State")["Mean_Absolute_SHAP"]
    .rank(method="first", ascending=False).astype(int)
)
state_importance = state_importance.sort_values(
    ["Ecological_State", "State_Rank"]
).reset_index(drop=True)
direction_summary = pd.DataFrame(direction_rows)


# =============================================================================
# 5. OBSERVED-STATE SUPPORT AND RAW LONG-FORM SHAP TABLE
# =============================================================================
observed_support_rows = []
raw_rows = []

for class_index, state in enumerate(explained_states):
    observed_mask = y_state == state

    for feature_index, feature in enumerate(FEATURES):
        observed_values = shap_values[observed_mask, class_index, feature_index]
        observed_support_rows.append({
            "Ecological_State": int(state),
            "State_Name": STATE_NAMES[int(state)],
            "Feature": feature,
            "Mean_Signed_SHAP_in_Observed_State": float(
                np.mean(observed_values)
            ),
            "Median_Signed_SHAP_in_Observed_State": float(
                np.median(observed_values)
            ),
            "Mean_Absolute_SHAP_in_Observed_State": float(
                np.mean(np.abs(observed_values))
            )
        })

    for row_index in range(n_samples):
        for feature_index, feature in enumerate(FEATURES):
            raw_rows.append({
                "Months": data.loc[row_index, DATE_COL].strftime("%Y-%m-%d"),
                "Observed_State": int(y_state[row_index]),
                "Explained_State": int(state),
                "State_Name": STATE_NAMES[int(state)],
                "Feature": feature,
                "Feature_Value": float(X.iloc[row_index, feature_index]),
                "SHAP_Value": float(
                    shap_values[row_index, class_index, feature_index]
                )
            })

observed_state_support = pd.DataFrame(observed_support_rows)
raw_shap_long = pd.DataFrame(raw_rows)


# =============================================================================
# 6. PREDICTION AND BASE-VALUE AUDIT TABLE
# =============================================================================
predicted_class_index = np.argmax(probability, axis=1)
predicted_state = explained_states[predicted_class_index]

prediction_audit = pd.DataFrame({
    "Months": data[DATE_COL].dt.strftime("%Y-%m-%d"),
    "Observed_State": y_state,
    "Full_Data_Model_Predicted_State": predicted_state,
    "Correct": (predicted_state == y_state).astype(int)
})

for class_index, state in enumerate(explained_states):
    prediction_audit[f"State_{state}_Probability"] = probability[:, class_index]
    prediction_audit[f"State_{state}_Raw_Score"] = raw_prediction[:, class_index]
    prediction_audit[f"State_{state}_SHAP_Base_Value"] = (
        expected_values[:, class_index]
    )


# =============================================================================
# 7. SAVE FINAL NUMERICAL OUTPUTS
# =============================================================================
overall_importance.to_csv(
    OUTPUT_DIR / "01_Overall_SHAP_Importance.csv", index=False
)
state_importance.to_csv(
    OUTPUT_DIR / "02_State_Specific_SHAP_Importance.csv", index=False
)
direction_summary.to_csv(
    OUTPUT_DIR / "03_SHAP_Direction_Summary.csv", index=False
)
observed_state_support.to_csv(
    OUTPUT_DIR / "04_Observed_State_SHAP_Support.csv", index=False
)
raw_shap_long.to_csv(
    OUTPUT_DIR / "05_Raw_SHAP_Long_Format.csv", index=False
)
prediction_audit.to_csv(
    OUTPUT_DIR / "06_Full_Data_Model_Prediction_Audit.csv", index=False
)

settings = {
    "purpose": "Final CatBoost interpretation; no model training performed",
    "data_file": str(DATA_FILE),
    "model_file": str(MODEL_FILE),
    "observations": n_samples,
    "features": FEATURES,
    "model_classes": model_classes.tolist(),
    "reported_ecological_states": explained_states.tolist(),
    "maximum_shap_additivity_error": maximum_additivity_error,
    "important_reporting_note": (
        "Use LOYO results for predictive performance. The full-data model "
        "is used only for final SHAP interpretation."
    )
}
(OUTPUT_DIR / "00_SHAP_Settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8"
)


# =============================================================================
# 8. COMPACT CONSOLE SUMMARY
# =============================================================================
print("\n" + "=" * 104)
print("FINAL CATBOOST SHAP OUTPUTS COMPLETED")
print("=" * 104)
print(f"Observations                 : {n_samples}")
print(f"Environmental predictors    : {n_features}")
print(f"Ecological states           : {n_classes}")
print(f"Maximum additivity error    : {maximum_additivity_error:.10f}")

print("\nTop environmental predictors:")
print(
    overall_importance.head(10)[
        ["Rank", "Feature", "Mean_Absolute_SHAP",
         "Relative_Importance_Percent"]
    ].round(4).to_string(index=False)
)

print("\nTop three predictors for each state:")
print(
    state_importance[state_importance["State_Rank"] <= 3][
        ["Ecological_State", "Feature", "Mean_Absolute_SHAP", "State_Rank"]
    ].round(4).to_string(index=False)
)

print("\nResults saved in:")
print(OUTPUT_DIR)
print("\nImportant: report predictive scores from LOYO, not from the SHAP audit table.")


# In[ ]:




