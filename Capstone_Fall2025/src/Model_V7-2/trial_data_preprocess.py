import argparse
from pathlib import Path
import numpy as np
import pandas as pd

data_version = "Data_v4"

def count_non_na(s: pd.Series) -> int:
    return int(s.notna().sum())

def safe_slope(trial: pd.Series, pe: pd.Series) -> float:
    x = trial.to_numpy()
    y = pe.to_numpy()
    m = ~pd.isna(x) & ~pd.isna(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        return np.nan
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    return float(cov_xy / var_x)

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    m = (~values.isna()) & (~weights.isna())
    v, w = values[m], weights[m]
    if w.sum() == 0 or v.empty:
        return np.nan
    return float((v * w).sum() / w.sum())

def preprocess_trial_data():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    in_path = project_root / "data" / data_version / f"LLM Trial Level Data.csv"
    out_path = project_root / "data" / data_version / f"Processed_Trial_Level_Data.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"sub", "profile", "subcat", "trial", "PE"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # ----- trial counts (optional print)
    trial_counts = (
        df.groupby(["sub", "profile", "subcat"], dropna=False)
        .agg(n_trials=("trial", count_non_na))
        .reset_index()
    )
    avg_trial_counts = (
        trial_counts.groupby("subcat", dropna=False)
        .agg(avg_n_trials=("n_trials", "mean"))
        .reset_index()
    )

    print("\nAverage trials per subcat:")
    print(avg_trial_counts.sort_values("subcat").to_string(index=False))

    # ----- slopes within (sub, profile, subcat)
    df_non_na = df[~df["PE"].isna()].copy()

    def slope_and_n(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "slope": safe_slope(g["trial"], g["PE"]),
            "n_trials": len(g)
        })

    subcat_slopes = (
        df_non_na.groupby(["sub", "profile", "subcat"], dropna=False)
        .apply(slope_and_n)
        .reset_index()
    )

    # ----- weighted mean by profile within each sub
    slopes_non_nan = subcat_slopes.dropna(subset=["slope"]).copy()

    by_profile = (
        slopes_non_nan.groupby(["sub", "profile"], dropna=False)
        .apply(lambda g: pd.Series({
            "mean_slope": weighted_mean(g["slope"], g["n_trials"]),
            "n_subcats": len(g)
        }))
        .reset_index()
    )

    # Merge with a skeleton of all (sub, profile) present in subcat_slopes
    skeleton = subcat_slopes[["sub", "profile"]].drop_duplicates()
    by_profile = skeleton.merge(by_profile, on=["sub", "profile"], how="left")

    # ----- overall weighted mean per sub
    overall = (
        slopes_non_nan.groupby("sub", dropna=False)
        .apply(lambda g: pd.Series({
            "mean_slope_overall": weighted_mean(g["slope"], g["n_trials"])
        }))
        .reset_index()
    )

    # ----- final: long format + overall repeated per sub
    final_df = (
        by_profile.merge(overall, on="sub", how="left")
        .drop(columns=["n_subcats"])
        .sort_values(["sub", "profile"])
        .reset_index(drop=True)
    )

    # Order columns to match your example
    final_df = final_df[["sub", "profile", "mean_slope", "mean_slope_overall"]]

    # Write with NA literal for missing values
    final_df.to_csv(out_path, index=False, na_rep="NA")

    print(f"\nProcessed Trial Level Data and saved to: {out_path}")
    print("\nPreview:")
    print(final_df.head(12).to_string(index=False))
    return final_df


if __name__ == "__main__":
    processed_trial_data = preprocess_trial_data()
