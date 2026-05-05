import pandas as pd
import numpy as np

# Define failure thresholds (krad) based on the CONSTELLATION_SHIELDING dictionary
TID_THRESHOLDS = {
    "STARLINK (LEO)": 5.0,
    "ONEWEB": 10.0,
    "IRIDIUM": 10.0,
    "PLANET (LEO)": 5.0,
    "GPS (MEO)": 100.0,
    "GALILEO": 30.0,
    "GLOBALSTAR": 10.0,
    "ORBCOMM": 5.0,
    "WORLDVIEW": 200.0,
    "ISS": 50.0,
    "GLONASS": 75.0,
    "BEIDOU": 100.0,
    "INTELSAT (GEO)": 50.0,
    "INMARSAT": 50.0,
    "AEHF": 300.0,
    "MILITARY": 200.0
}

def get_threshold(constellation_name):
    """Matches the constellation string to its TID threshold."""
    name_upper = str(constellation_name).upper()
    for key, threshold in TID_THRESHOLDS.items():
        if key in name_upper:
            return threshold
    return 10.0  # Default COTS fallback

def analyze_simulation(csv_path="results4/hand_simulation_final.csv"):
    print("==========================================================")
    print("      HAND PUNCH - SATELLITE SURVIVABILITY REPORT         ")
    print("==========================================================\n")
    
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Could not find {csv_path}. Ensure the simulation has run.")
        return

    # Apply Thresholds & Determine Delayed Death
    df['TID_Threshold_krad'] = df['Constellation'].apply(get_threshold)
    
    # Logic: Dead if it exceeds its specific TID limit
    df['Delayed_Fatal'] = df['Corrected_Argus_krad'] > df['TID_Threshold_krad']
    
    # Logic: Promptly dead if Status contains 'Fatal'
    df['Prompt_Fatal'] = df['Prompt_Status'].str.contains('Fatal', na=False, case=False)
    
    # Logic: Overall dead if Prompt OR Delayed
    df['Overall_Fatal'] = df['Prompt_Fatal'] | df['Delayed_Fatal']
    
    total_sats = len(df)
    
    # ---------------------------------------------------------
    # INSIGHT 1: OVERALL NUMBERS (MACRO)
    # ---------------------------------------------------------
    prompt_dead_count = df['Prompt_Fatal'].sum()
    delayed_dead_count = df['Delayed_Fatal'].sum()
    overall_dead_count = df['Overall_Fatal'].sum()
    survived_count = total_sats - overall_dead_count
    
    print("--- 1. MACRO SURVIVABILITY ---")
    print(f"Total Satellites Tracked:   {total_sats:,}")
    print(f"Destroyed Instantly:        {prompt_dead_count:,} ({prompt_dead_count/total_sats:.1%})")
    print(f"Destroyed by Day 30:        {delayed_dead_count:,} ({delayed_dead_count/total_sats:.1%})")
    print(f"Total Assets Lost:          {overall_dead_count:,} ({overall_dead_count/total_sats:.1%})")
    print(f"Total Assets Survived:      {survived_count:,} ({survived_count/total_sats:.1%})\n")

    # ---------------------------------------------------------
    # INSIGHT 2: EARTH SHADOWING (LINE OF SIGHT)
    # ---------------------------------------------------------
    # Satellites with ~0 X-Ray fluence were shielded by the Earth's bulk
    shadowed_sats = df[df['XRay_Jm2'] < 1e-5]
    print("--- 2. GEOMETRIC SHIELDING ---")
    print(f"Satellites Shielded by Earth at Detonation: {len(shadowed_sats):,} ({len(shadowed_sats)/total_sats:.1%})\n")

    # ---------------------------------------------------------
    # INSIGHT 3: BREAKDOWN BY CONSTELLATION
    # ---------------------------------------------------------
    print("--- 3. CONSTELLATION IMPACT BREAKDOWN ---")
    print(f"{'Constellation':<20} | {'Total':<6} | {'Instant Loss':<15} | {'30-Day Loss':<15} | {'Survival Rate'}")
    print("-" * 80)
    
    for const in sorted(df['Constellation'].unique()):
        subset = df[df['Constellation'] == const]
        c_total = len(subset)
        c_prompt = subset['Prompt_Fatal'].sum()
        c_delayed = subset['Delayed_Fatal'].sum()
        c_overall = subset['Overall_Fatal'].sum()
        c_survival_rate = (c_total - c_overall) / c_total
        
        print(f"{const[:19]:<20} | {c_total:<6} | {c_prompt:<6} ({c_prompt/c_total:>5.1%}) | {c_delayed:<6} ({c_delayed/c_total:>5.1%}) | {c_survival_rate:>5.1%}")

    print("\n")
    
    # ---------------------------------------------------------
    # INSIGHT 4: THE MOST DANGEROUS ORBITAL BANDS
    # ---------------------------------------------------------
    print("--- 4. RADIATION ENVIRONMENT BY ALTITUDE ---")
    # Group satellites into rough altitude bins (LEO, MEO, GEO)
    bins = [0, 2000, 30000, 50000]
    labels = ['LEO (<2000km)', 'MEO (2k-30k km)', 'GEO (>30k km)']
    df['Orbit_Regime'] = pd.cut(df['Mean_Alt_km'], bins=bins, labels=labels)
    
    regime_stats = df.groupby('Orbit_Regime')['Corrected_Argus_krad'].agg(['mean', 'max'])
    for regime, row in regime_stats.iterrows():
        if not pd.isna(row['mean']):
            print(f"{regime:<18}: Avg Dose = {row['mean']:>8.1f} krad | Max Dose = {row['max']:>8.1f} krad")

if __name__ == "__main__":
    analyze_simulation()