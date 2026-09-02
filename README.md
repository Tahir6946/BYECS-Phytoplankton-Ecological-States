# Phytoplankton Ecological-State Analysis

This repository contains the analysis code, processed data, trained models, and numerical results supporting the manuscript:

**“Discovering and Predicting Phytoplankton Ecological States and Their Environmental Associations Using Compositional Analysis and Explainable Machine Learning”**

The study examines monthly phytoplankton community organization across the Bohai, Yellow, and East China Seas (BYECS) from **2003 to 2022**. Six phytoplankton functional groups are used to identify recurrent ecological states, characterize their environmental associations, and evaluate their predictability using machine-learning and deep-learning models.

## Repository structure

```text
Phytoplankton_Ecological_States/
│
├── Scripts/
│   ├── 01_Complete_Analysis_Pipeline.py
│   └── 02_Figure_Generation.py
│
├── Data/
│   ├── 01_Raw_Species_Environmental_Monthly_Data.xlsx
│   ├── 04_Final_Current_Environmental_Modeling_Data.csv
│   ├── Data_Manifest.csv
│   └── README_Data.txt
│
├── Models/
│   ├── CLR_PCA_Model.pkl
│   ├── Final_KMeans_Model.pkl
│   └── Final_Six_Models/
│       ├── Final_CatBoost.cbm
│       ├── Final_XGBoost.json
│       ├── Final_HistGradientBoosting.pkl
│       ├── Final_TCN.keras
│       ├── Final_CNN_LSTM.keras
│       ├── Final_DL_StandardScaler.pkl
│       └── Equal_Soft_Voting_Metadata.json
│
├── Results/
│   ├── 01_State_Discovery/
│   ├── 02_Environmental_Analysis/
│   ├── 03_Final_Models/
│   ├── 04_Evaluation_Curves/
│   └── 05_CatBoost_SHAP/
│
├── README.md
└── requirements.txt
```

The final publication figures are not stored in the repository because they can be regenerated from the saved numerical results using `02_Figure_Generation.py`.

## Analysis workflow

The complete analysis is implemented in `01_Complete_Analysis_Pipeline.py` and includes:

1. Conversion of six phytoplankton groups to monthly relative community composition.
2. Centered log-ratio (CLR) transformation.
3. Robust STL detrending with a 12-month seasonal period.
4. Row-wise recentering to preserve CLR zero-sum geometry.
5. Principal component analysis (PCA).
6. K-means ecological-state identification and robustness assessment.
7. Environmental characterization using 14 contemporaneous predictors.
8. Kruskal–Wallis tests, epsilon-squared effect sizes, and Benjamini–Hochberg FDR correction.
9. Leave-one-year-out (LOYO) model evaluation.
10. Optuna-based hyperparameter optimization using grouped inner cross-validation.
11. CatBoost, XGBoost, HistGradientBoosting, Equal Soft Voting, TCN, and CNN-LSTM modelling.
12. ROC, precision–recall, calibration, threshold, yearly-performance, and bootstrap evaluation.
13. CatBoost SHAP analysis for overall and state-specific model interpretation.

The deep-learning models use a **6-month input sequence**. All final model comparisons are therefore evaluated on the same **235 sequence-eligible held-out months**.

## Data

The primary input is:

```text
Data/01_Raw_Species_Environmental_Monthly_Data.xlsx
```

The final modelling dataset is:

```text
Data/04_Final_Current_Environmental_Modeling_Data.csv
```

It contains the final ecological-state labels together with 14 contemporaneous environmental predictors:

- SST
- NO3
- PO4
- surface pCO2
- MLD
- SSS
- SSH
- PAR
- PDO
- Niño 3.4
- WPI
- marine-heatwave mean intensity
- marine-heatwave maximum intensity
- marine-heatwave cumulative intensity

Further information on the data files is provided in `Data/README_Data.txt` and `Data/Data_Manifest.csv`.

## Ecological-state construction

The six phytoplankton functional groups are:

- DIATO
- DINO
- HAPTO
- GREEN
- PROKAR
- PROCHLO

The final ecological states are derived from detrended CLR-transformed community composition using PCA followed by K-means clustering with **K = 3**.

Cluster robustness is evaluated using internal clustering criteria, bootstrap adjusted Rand index (ARI), random-initialization stability, and sensitivity to the number of retained principal components.

## Predictive modelling

The ecological states are predicted from current-month environmental conditions using:

- CatBoost
- XGBoost
- HistGradientBoosting
- Equal Soft Voting
- Temporal Convolutional Network (TCN)
- CNN-LSTM

A month-of-year seasonal baseline is retained as a reference model.

Model performance is assessed using temporally independent **leave-one-year-out validation**. Hyperparameter optimization for the tree-based models is performed only within the corresponding LOYO training data.

## Model interpretation

SHAP analysis is performed using the final full-data CatBoost model to characterize overall and state-specific environmental associations.

**Important:** predictive performance is reported exclusively from held-out LOYO predictions. The full-data CatBoost model is used only for final SHAP interpretation.

## Running the analysis

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the complete analysis:

```bash
python Code/01_Complete_Analysis_Pipeline.py
```

Generate the publication and supplementary figures:

```bash
python Code/02_Figure_Generation.py
```

The analysis script writes numerical outputs to `Results/` and trained models to `Models/`.

### Path setting

The public version of the scripts should use the repository directory as the project root. If a local absolute path is still present in either script, update the `ROOT` variable before running.

## Reproducibility

The main reproducibility settings include:

- Random seed: `42`
- STL period: `12` months
- Final number of ecological states: `3`
- Retained PCA dimensions for state construction: `PC1–PC2`
- Bootstrap clustering runs: `200`
- Optuna trials per tree-model optimization: `20`
- Inner grouped cross-validation folds: `4`
- Outer validation: leave-one-year-out
- Deep-learning sequence length: `6` months
- Maximum deep-learning epochs: `200`
- Deep-learning batch size: `16`

Saved intermediate tables, held-out predictions, trained models, evaluation outputs, and SHAP values are included to facilitate verification without requiring every computationally intensive step to be rerun.

## Key result files

Important numerical outputs include:

```text
Results/01_State_Discovery/09_Final_Ecological_States.csv
Results/02_Environmental_Analysis/06_Environmental_State_Tests.csv
Results/03_Final_Models/01_LOYO_Held_Out_Predictions.csv
Results/03_Final_Models/02_Pooled_Model_Performance.csv
Results/03_Final_Models/03_Per_State_Metrics.csv
Results/03_Final_Models/07_Optuna_Best_Parameters.csv
Results/05_CatBoost_SHAP/01_Overall_SHAP_Importance.csv
Results/05_CatBoost_SHAP/02_State_Specific_SHAP_Importance.csv
```

## Software environment

The exact package versions used for the final analysis should be recorded in `requirements.txt`. Core dependencies include Python, NumPy, pandas, SciPy, scikit-learn, statsmodels, matplotlib, CatBoost, XGBoost, TensorFlow/Keras, Optuna, and joblib.

## Citation

If you use this repository, please cite the associated manuscript and the archived Zenodo release.

**Manuscript citation:** To be added after publication.

**Zenodo DOI:** Zenodo DOI: 10.5281/zenodo.22241189

## Code availability statement

The analysis and figure-generation code supporting this study is publicly available through this repository. A versioned archival copy will also be deposited on Zenodo and assigned a DOI.
