"""
Updated April 25, 2025
"""

# === THINGS TO CHECK ===
# - correct paths are set inside _00_config.py
# - vendor and master file must each have the columns "Year", "Make", and "Model"

import os, pandas as pd
from _00_config import (
    INPUT_DIR,
    OUTPUT_DIR,
    VENDOR_FILE,
    MASTER_FILE,
    VENDOR_SHEET_NAME,
    COLS_TO_USE,
)

# === FUNCTIONS ===

def process_vehicle_df(df):
    """Clean and prepare the vehicle dataframe."""
    df = df.dropna(subset=["Year"])    # Drop rows where Year is missing
    df = df.drop_duplicates(subset=COLS_TO_USE).copy()
    df["Year"] = df["Year"].astype(int)
    df["Car_ID"] = (
        df["Year"].astype(str) + "_" +
        df["Make"].astype(str) + "_" +
        df["Model"].astype(str)
    )
    df["match_car_id"] = (
        df["Year"].astype(str) + "_" +
        df["Make"].astype(str) + "_" +
        df["Model"].astype(str).str.replace("-", "", regex=False).str.upper()
    )
    return df

def find_unmatched_and_exact_matches(vendor_csv, master_csv, out_vendor_unmatched, out_master_unmatched, out_exact_match):
    """Finds unmatched vendor/master rows and saves exact matches."""
    try:
        vendor_df = pd.read_csv(vendor_csv)
        master_df = pd.read_csv(master_csv)

        vendor_df['match_car_id'] = vendor_df['match_car_id'].astype(str)
        master_df['match_car_id'] = master_df['match_car_id'].astype(str)

        vendor_ids = set(vendor_df['match_car_id'])
        master_ids = set(master_df['match_car_id'])

        vendor_unmatched_ids = vendor_ids - master_ids
        master_unmatched_ids = master_ids - vendor_ids

        vendor_unmatched = vendor_df[
            (vendor_df['match_car_id'].isin(vendor_unmatched_ids)) &
            (vendor_df[['Year', 'Make']].apply(tuple, axis=1).isin(master_df[['Year', 'Make']].apply(tuple, axis=1)))
        ].drop(columns=["match_car_id"])

        vendor_unmatched_years = vendor_unmatched['Year'].unique().tolist()

        master_unmatched = master_df[
            (master_df['match_car_id'].isin(master_unmatched_ids)) &
            (master_df[['Year', 'Make']].apply(tuple, axis=1).isin(vendor_df[['Year', 'Make']].apply(tuple, axis=1))) &
            (master_df['Year'].isin(vendor_unmatched_years))
        ].drop(columns=["match_car_id"])

        # Save unmatched
        vendor_unmatched.to_csv(out_vendor_unmatched, index=False)
        master_unmatched.to_csv(out_master_unmatched, index=False)

        # Exact matches
        exact_matches = pd.merge(
            vendor_df, master_df,
            on=["match_car_id", "Year", "Make"],
            suffixes=('_vendor', '_master')
        )

        exact_matches_final = pd.DataFrame({
            "Vendor_Car_ID": exact_matches["Car_ID_vendor"],
            "Best_Match_Car_ID": (
                exact_matches["Year"].astype(int).astype(str) + "_" +
                exact_matches["Make"].astype(str) + "_" +
                exact_matches["Model_master"].astype(str)
            ),
            "Match_Confidence": 100
        })

        exact_matches_final.to_csv(out_exact_match, index=False)

        print(f"✅ Match lists saved to:\n- {out_vendor_unmatched}\n- {out_master_unmatched}\n- {out_exact_match}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

# === MAIN FLOW ===

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vendor_path = os.path.join(INPUT_DIR, VENDOR_FILE)
    master_path = os.path.join(INPUT_DIR, MASTER_FILE)

    vendor_df = pd.read_excel(vendor_path, sheet_name=VENDOR_SHEET_NAME, usecols=COLS_TO_USE)
    master_df = pd.read_excel(master_path, usecols=COLS_TO_USE)

    vendor_df = process_vehicle_df(vendor_df)
    master_df = process_vehicle_df(master_df)

    vendor_csv = os.path.join(OUTPUT_DIR, "vendor_file.csv")
    master_csv = os.path.join(OUTPUT_DIR, "master_file.csv")

    vendor_df.to_csv(vendor_csv, index=False)
    master_df.to_csv(master_csv, index=False)
    print("✅ Pre-processing complete. Files saved.")

    # Run match comparison
    find_unmatched_and_exact_matches(
        vendor_csv,
        master_csv,
        os.path.join(OUTPUT_DIR, "vendor_match_list.csv"),
        os.path.join(OUTPUT_DIR, "master_match_list.csv"),
        os.path.join(OUTPUT_DIR, "exact_match_list.csv")
    )

if __name__ == "__main__":
    main()
