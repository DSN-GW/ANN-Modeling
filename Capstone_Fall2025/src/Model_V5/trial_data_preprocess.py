import argparse
from pathlib import Path
import numpy as np
import pandas as pd

data_version = "Data_v5"

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

def compute_level_summary(df: pd.DataFrame, level_col: str) -> pd.DataFrame:
    """
    Compute long-format summary for a given level column (e.g., 'subcat' or 'cat'):
      - slope within (sub, profile, level_col)
      - weighted mean slope by profile within sub
      - weighted mean slope overall within sub
      - final columns: ['sub','profile','mean_slope','mean_slope_overall']
    """
    # trial counts (for info)
    trial_counts = (
        df.groupby(["sub", "profile", level_col], dropna=False)
          .agg(n_trials=("trial", count_non_na))
          .reset_index()
    )
    avg_trial_counts = (
        trial_counts.groupby(level_col, dropna=False)
                    .agg(avg_n_trials=("n_trials", "mean"))
                    .reset_index()
    )
    print(f"\nAverage trials per {level_col}:")
    print(avg_trial_counts.sort_values(level_col).to_string(index=False))

    # slopes within (sub, profile, level_col) using non-NA PE
    df_non_na = df[~df["PE"].isna()].copy()

    def slope_and_n(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "slope": safe_slope(g["trial"], g["PE"]),
            "n_trials": len(g)
        })

    unit_slopes = (
        df_non_na.groupby(["sub", "profile", level_col], dropna=False)
                 .apply(slope_and_n)
                 .reset_index()
    )

    # weighted mean by profile within sub (long)
    slopes_non_nan = unit_slopes.dropna(subset=["slope"]).copy()

    by_profile = (
        slopes_non_nan.groupby(["sub", "profile"], dropna=False)
                      .apply(lambda g: pd.Series({
                          "mean_slope": weighted_mean(g["slope"], g["n_trials"]),
                          "n_subunits": len(g)
                      }))
                      .reset_index()
    )

    # ensure we keep (sub, profile) pairs that had only NaN slopes
    skeleton = unit_slopes[["sub", "profile"]].drop_duplicates()
    by_profile = skeleton.merge(by_profile, on=["sub", "profile"], how="left")

    # overall weighted mean per sub
    overall = (
        slopes_non_nan.groupby("sub", dropna=False)
                      .apply(lambda g: pd.Series({
                          "mean_slope_overall": weighted_mean(g["slope"], g["n_trials"])
                      }))
                      .reset_index()
    )

    final_df = (
        by_profile.merge(overall, on="sub", how="left")
                  .drop(columns=["n_subunits"], errors="ignore")
                  .sort_values(["sub", "profile"])
                  .reset_index(drop=True)
    )
    final_df = final_df[["sub", "profile", "mean_slope", "mean_slope_overall"]]
    return final_df

def preprocess_trial_data():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    in_path  = project_root / "data" / data_version / "LLM Trial Level Data.csv"
    out_dir  = project_root / "data" / data_version


    df = pd.read_csv(in_path)

    base_required = {"sub", "profile", "trial", "PE"}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Subcategory level ---
    if "subcat" not in df.columns:
        raise ValueError("Expected 'subcat' column for subcategory-level processing, but it was not found.")
    subcat_summary = compute_level_summary(df, level_col="subcat")
    subcat_out = out_dir / "Processed_Trial_Level_Data_SUBCAT.csv"
    subcat_summary = subcat_summary.rename(columns={"mean_slope": "mean_slope_subcat", "mean_slope_overall": "mean_slope_overall_subcat"})
    subcat_summary.to_csv(subcat_out, index=False, na_rep="NA")
    print(f"\nSaved subcategory-level summary to: {subcat_out}")
    print("\nSubcategory preview:")
    print(subcat_summary.head(12).to_string(index=False))

    if "cat" not in df.columns:
        raise ValueError("Expected 'subcat' column for subcategory-level processing, but it was not found.")
    category_summary = compute_level_summary(df, level_col="cat")
    cat_out = out_dir / "Processed_Trial_Level_Data_CAT.csv"
    category_summary = category_summary.rename(columns={"mean_slope": "mean_slope_cat", "mean_slope_overall": "mean_slope_overall_cat"})
    category_summary.to_csv(cat_out, index=False, na_rep="NA")
    print(f"\nSaved category-level summary to: {cat_out}")
    print("\nCategory preview:")
    print(category_summary.head(12).to_string(index=False))

    return subcat_summary, category_summary

if __name__ == "__main__":
    preprocess_trial_data()
