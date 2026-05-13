"""
HAND Simulation Pipeline
================================================================
Integrated framework for High-Altitude Nuclear Detonation satellite 
survivability modeling.

Features:
- Prompt X-Ray, Neutron, and Gamma attenuation and effect modeling.
- Tilted dipole L-shell magnetic mapping.
- Bi-exponential temporal decay for Argus effects (Conrad et al. 2008).
- SHIELDOSE-2 power-law shielding attenuation (Seltzer 1994).
- Publication-quality visualization suite.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import timedelta
from skyfield.api import wgs84, load

warnings.filterwarnings("ignore")

# =============================================================================
# PART A: CATALOGS AND CONSTANTS
# =============================================================================

RE_KM = 6371.0

CONSTELLATION_CATALOG = {
    "STARLINK (LEO)":          ("STARLINK",),        
    "ONEWEB":            ("ONEWEB",),          
    "IRIDIUM":           ("IRIDIUM",),    
    "PLANET (LEO)":            ("FLOCK", "PELICAN", "SKYSAT"), # Combined into a single Planet key                   
    "GALILEO":           ("GALILEO",),        
    # "GLOBALSTAR":        ("GLOBALSTAR",),      # ~1414 km, 52° incl. 
    # "ORBCOMM":           ("ORBCOMM",),         # ~715 km, various incl.
    # "WORLDVIEW":         ("WORLDVIEW",),       # Maxar, ~617 km, 97.9° SSO
    "ISS":               ("ISS (ZARYA)",),     # ~408 km, 51.6° incl. HUMAN CREW
    "GPS (MEO)":               ("NAVSTAR",),         # Legacy GPS naming (could also add "GPS IIR" here)
    "GLONASS":           ("GLONASS",),         # Russian GNSS, ~19 130 km, 64.8° incl.
    "BEIDOU_MEO":        ("BEIDOU",),          # Chinese GNSS MEO component
    "INTELSAT (GEO)":          ("INTELSAT",),        # GEO, 35 786 km, ~0° incl.
    # "INMARSAT":          ("INMARSAT",),        # GEO maritime/aero comms
    "MILITARY":          ("USA ",),            # Will match USA-### designations
    "AEHF":              ("AEHF",),            # Advanced EHF (nuclear C2), GEO
}

CONSTELLATION_SHIELDING = {
    # name_key      : (shield_mm_Al, rad_hard, tid_krad_fail, notes)
    "STARLINK (LEO)":        (2.0,  False, 5.0,   "COTS, min shielding"),
    "ONEWEB":          (3.0,  False, 10.0,  "COTS LEO broadband"),
    "IRIDIUM":         (3.0,  False, 10.0,  "Commercial LEO"),
    "PLANET (LEO)":    (2.5,  False, 5.0,   "COTS cubesat"),
    "GPS (MEO)":             (10.0, True,  100.0, "Rad-hardened per MIL-spec"),
    "GALILEO":         (8.0,  False, 30.0,  "Commercial components, some hardening"),
    "GLOBALSTAR":      (3.0,  False, 10.0,  "Commercial LEO"),
    "ORBCOMM":         (2.0,  False, 5.0,   "COTS IoT LEO"),
    "ISS":             (15.0, False, 50.0,  "Crew vehicle — human dose limits apply"),
    "BEIDOU_MEO":      (10.0, True,  100.0, "Rad-hardened estimate?"),
    "WORLDVIEW":       (10.0, True,  200.0, "Rad-hardened BAE Systems RAD750 computer"),
    "GLONASS":         (8.0,  True,  75.0,  "Partial hardening"),
    "INTELSAT (GEO)":        (8.0,  True,  50.0,  "GEO, designed for belt exposure"),
    "INMARSAT":        (8.0,  True,  50.0,  "GEO commercial"),
    "MILITARY":         (15.0, True,  200.0, "General Military designation"),
    "AEHF":            (20.0, True,  300.0, "Nuclear C2, highest hardening"),
}

# =============================================================================
# PART B: PHYSICS HELPERS (Argus Decay Model & Prompt Effects)
# =============================================================================
    
def ae8_max_flux_simple(l_shell):
    l = np.atleast_1d(np.asarray(l_shell, dtype=float))
    r = 1e9*np.exp(-((l-1.5)**2)/(2*0.3**2)) + 5e7*np.exp(-((l-4.5)**2)/(2*0.8**2))
    r[l<1.05]=1e2; r[l>8.5]=1e2
    return float(r[0]) if r.size==1 else r

def hand_enhancement_factor(l_shell, burst_l, yield_kt, time_days):
    starfish_kt = 1400.0
    peak_enh    = 100.0 * (yield_kt / starfish_kt) ** 0.7
    sigma_l     = 0.25 + 0.35 * np.log10(max(yield_kt / 10.0, 1.0) + 1.0)

    spatial  = np.exp(-((l_shell - burst_l)**2) / (2 * sigma_l**2))
    alpha    = min(0.1 + 0.15 * burst_l, 0.5)
    temporal = alpha * np.exp(-time_days / 40.0) + (1 - alpha) * np.exp(-time_days / 500.0)

    return 1.0 + peak_enh * spatial * temporal

    
def shieldose2_factor(mm):
    d=2.70*mm/10; D0,lam,dbup,b=8.5e-4,0.90,0.05,2.0
    return D0*np.exp(-d/lam) if d<dbup else D0*(1+d/lam)**(-b)

def corrected_argus_dose(l_shell, burst_l, yield_kt, shielding_mm_al, days):
    phi_base = ae8_max_flux_simple(l_shell)
    peak_enh = 100.0 * (yield_kt / 1400.0) ** 0.7
    sigma_l  = 0.25 + 0.35 * np.log10(max(yield_kt / 10.0, 1.0) + 1.0)
    spatial  = np.exp(-((l_shell - burst_l)**2) / (2 * sigma_l**2))
    alpha    = min(0.1 + 0.15 * burst_l, 0.5)
    cf = shieldose2_factor(shielding_mm_al)

    base_dr = phi_base * cf
    dose_base = base_dr * days
    
    A = peak_enh * phi_base * spatial * cf
    dose_enh = A * (alpha * 40.0 * (1 - np.exp(-days / 40.0)) +
                    (1 - alpha) * 500.0 * (1 - np.exp(-days / 500.0)))

    dose_rad  = dose_base + np.maximum(dose_enh, 0.0)
    return dose_rad, dose_rad / 1000.0 

def corrected_argus_dose_krad(l_shell, burst_l, yield_kt, shield_mm, days):
    phi  = ae8_max_flux_simple(l_shell)
    enh  = 100.0*(yield_kt/1400.0)**0.7
    sig  = 0.25+0.35*np.log10(max(yield_kt/10,1)+1)
    spat = np.exp(-((np.asarray(l_shell)-burst_l)**2)/(2*sig**2))
    alp  = min(0.1+0.15*burst_l, 0.5)
    cf   = shieldose2_factor(shield_mm)
    D    = np.asarray(phi)*cf*days + enh*np.asarray(phi)*spat*cf*(
           alp*40*(1-np.exp(-days/40))+(1-alp)*500*(1-np.exp(-days/500)))
    return float(np.mean(np.maximum(D,0)))/1000

def classify_prompt_effects(xray_jm2, neutron_nm2, gamma_rads):
    effects = []
    if xray_jm2 >= 400: effects.append("X-RAY:THERMAL_FATAL")
    if xray_jm2 >= 4.0: effects.append("X-RAY:SGEMP_FATAL")
    if xray_jm2 >= 0.4: effects.append("X-RAY:IONIZ_UPSET")
    if neutron_nm2 >= 1e16: effects.append("NEUTRON:LATTICE_FATAL")
    elif neutron_nm2 >= 1e10: effects.append("NEUTRON:UPSET")
    if gamma_rads >= 1e4: effects.append("GAMMA:TID_FATAL")
    elif gamma_rads >= 1e3: effects.append("GAMMA:TID_UPSET")
    return effects if effects else ["NOMINAL"]

def prompt_overall_status(effects_list):
    joined = " ".join(effects_list)
    if "FATAL" in joined: return "Fatal/Destroyed"
    elif "UPSET" in joined or "SGEMP" in joined: return "Severely Degraded"
    return "Nominal"

# =============================================================================
# PART C: SIMULATION CLASS
# =============================================================================

class HANDSimulationAdvanced:
    import re
from skyfield.api import load, wgs84

class HANDSimulationAdvanced:
    def __init__(self, tle_file, target_constellation="STARLINK", preloaded_sats=None):
        self.ts = load.timescale()
        self.planets = load('de421.bsp')
        self.earth = self.planets['earth']
        
        # ---------------------------------------------------------
        # OPTIMIZATION 1: Eliminate Redundant Disk I/O
        # ---------------------------------------------------------
        if preloaded_sats is None:
            # Only hit the disk if no pre-loaded list was provided
            self.satellites = load.tle_file(tle_file)
        else:
            # Use the existing memory reference (Instantaneous)
            self.satellites = preloaded_sats
            
        # ---------------------------------------------------------
        # OPTIMIZATION 2: Regex Multi-String Filtering
        # ---------------------------------------------------------
        # Get the tuple of strings (e.g. ("FLOCK", "PELICAN", "SKYSAT"))
        filter_tuple = CONSTELLATION_CATALOG.get(target_constellation, (target_constellation,))
        
        # Compile a regex pattern (e.g. 'FLOCK|PELICAN|SKYSAT')
        # This executes in highly optimized C code, much faster than python loops
        pattern = re.compile('|'.join(filter_tuple))
        
        # Filter the list
        self.target_sats = [sat for sat in self.satellites if pattern.search(sat.name)]
        
        print(f"Loaded {len(self.target_sats)} satellites for {target_constellation}")

    def simulate_burst(self, burst_lat, burst_lon, burst_alt_km, yield_kt, burst_time_utc):
        self.burst_time = self.ts.utc(*burst_time_utc)
        self.yield_kt = yield_kt
        self.burst_alt_km = burst_alt_km
        
        # Modern Skyfield approach using the WGS84 Geoid
        self.burst_geo = wgs84.latlon(latitude_degrees=burst_lat, 
                                      longitude_degrees=burst_lon, 
                                      elevation_m=burst_alt_km * 1000)
        
        # Calling .at(t) on a wgs84 geographic position directly returns 
        # Geocentric coordinates (Earth center = 399). No solar system barycenter math!
        self.burst_icrf = self.burst_geo.at(self.burst_time)
        
        # Calculate burst L-shell mapping
        self.burst_L_shell = self.calculate_l_shell(burst_lat, burst_lon, burst_alt_km)

    def calculate_l_shell(self, lat_deg, lon_deg, alt_km):
        lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
        lat_p, lon_p = np.radians(80.8), np.radians(-72.6)
        sin_mag_lat = np.sin(lat_rad)*np.sin(lat_p) + np.cos(lat_rad)*np.cos(lat_p)*np.cos(lon_rad - lon_p)
        mag_lat = np.arcsin(sin_mag_lat)
        r = (RE_KM + alt_km) / RE_KM
        return r / (np.cos(mag_lat)**2)

    def calculate_ray_minimum_altitude(self, pos1_km, pos2_km):
        d = pos2_km - pos1_km
        t = -np.dot(pos1_km, d) / np.dot(d, d)
        if t < 0: min_point = pos1_km
        elif t > 1: min_point = pos2_km
        else: min_point = pos1_km + t * d
        return np.linalg.norm(min_point) - RE_KM

    def get_attenuation_factors(self, min_alt_km):
        if min_alt_km > 90: x_ray_surv = 1.0
        elif min_alt_km > 55: x_ray_surv = (min_alt_km - 55) / (90 - 55) 
        else: x_ray_surv = 0.0
            
        if min_alt_km > 25: gamma_neutron_surv = 1.0
        else: gamma_neutron_surv = 0.0
        return x_ray_surv, gamma_neutron_surv

    def calculate_prompt_effects(self):
        results = []
        r_burst_km = self.burst_icrf.position.km
        
        for sat in self.target_sats:
            sat_pos = sat.at(self.burst_time)
            r_sat_km = sat_pos.position.km
            distance_km = np.linalg.norm(r_sat_km - r_burst_km) # <-- 1. Calculate KM
            distance_m = distance_km * 1000.0
            
            min_alt_km = self.calculate_ray_minimum_altitude(r_burst_km, r_sat_km)
            x_ray_surv, g_n_surv = self.get_attenuation_factors(min_alt_km)
            
            fluence_xray = ((2.3e11 * self.yield_kt) / (distance_m ** 2)) * x_ray_surv
            fluence_neutron = ((1.6e11 * self.yield_kt) / (distance_m ** 2)) * g_n_surv
            dose_gamma = ((2.5e1 * self.yield_kt) / (distance_m ** 2)) * g_n_surv
            
            effects = classify_prompt_effects(fluence_xray, fluence_neutron, dose_gamma)
            
            results.append({
                'NORAD_ID': sat.model.satnum,
                'Name': sat.name,
                'Distance_km': distance_km,  # <-- 2. SAVE IT TO THE DATAFRAME
                'XRay_Jm2': fluence_xray,
                'Neutron_nm2': fluence_neutron,
                'Gamma_radsSi': dose_gamma,
                'Effects_List': effects,
                'Prompt_Status': prompt_overall_status(effects)
            })
        return pd.DataFrame(results)

    def calculate_delayed_argus_effect(self, days=30, shielding_al_mm=2.5):
        # Sample orbit over 24 hours to find L-Shell distribution (5 minute intervals)
        base_time = self.burst_time.utc_datetime()
        datetime_list = [base_time + timedelta(minutes=m) for m in range(0, 1440, 5)]
        t_array = self.ts.from_datetimes(datetime_list)
        
        results = []
        for sat in self.target_sats:
            sat_positions = sat.at(t_array)
            subpoint = sat_positions.subpoint()
            
            sat_l_shells = self.calculate_l_shell(subpoint.latitude.degrees, 
                                                  subpoint.longitude.degrees, 
                                                  subpoint.elevation.km)
            
            # Apply analytical dose model to the orbital distribution and average it
            _, doses_krad = corrected_argus_dose(sat_l_shells, self.burst_L_shell, 
                                                 self.yield_kt, shielding_al_mm, days)
            avg_dose_krad = np.mean(doses_krad)
            
            results.append({
                'NORAD_ID': sat.model.satnum,
                'Name': sat.name,
                'Corrected_Argus_krad': avg_dose_krad,
                'Inclination_deg': np.degrees(sat.model.inclo), # <-- ADD THIS
                'Mean_Alt_km': np.mean(subpoint.elevation.km)   # <-- ADD THIS
            })
            
        return pd.DataFrame(results)

# =============================================================================
# PART D: VISUALIZATIONS (Imported from your script)
# =============================================================================
# [I have included your plot_multi_effect_scatter exactly as you provided it, 
# you can seamlessly paste the rest of your Part D plotting functions here]


STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

def plot_xray_fluence_vs_distance_empirical(df, simulated_yield_kt=1400, output_dir="results"):
    import os; os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))

    # Empirical Scatter
    if df is not None and not df.empty and 'Distance_km' in df.columns:
        for const in df['Constellation'].unique():
            subset = df[(df['Constellation'] == const) & (df['XRay_Jm2'] > 1e-5)]
            ax.scatter(subset['Distance_km'], subset['XRay_Jm2'], alpha=0.6, s=20, edgecolor='none', label=f"Simulated {const}")

    # Thresholds
    for val, col, lbl in [(400, "#c0392b", "Thermal (400)"), (4.0, "#e67e22", "SGEMP (4)"), (0.4, "#f1c40f", "Ionization (0.4)")]:
        ax.axhline(val, color=col, ls=":", lw=1.5, label=lbl)

    # Theoretical
    r_km = np.linspace(10, 50000, 500) 
    fluence_theory = 2.3e11 * simulated_yield_kt / (r_km * 1e3)**2
    ax.semilogy(r_km, fluence_theory, color="black", ls="--", lw=2, zorder=-1, label=f"Theoretical Curve ({simulated_yield_kt} kt)")

    ax.axvspan(160, 2000, alpha=0.07, color="#3498db", label="LEO band")
    ax.set_xlabel("Distance from Burst Point [km]", fontsize=12)
    ax.set_ylabel("Prompt X-Ray Fluence [J m⁻²]", fontsize=12)
    ax.set_title(f"X-Ray Fluence vs. Distance ({simulated_yield_kt} kt Simulation)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 50000)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "VIZ1_xray_fluence_empirical.png"), dpi=150)
    plt.close(fig)


def plot_prompt_damage_radius_vs_yield_empirical(df, simulated_yield_kt=1400, output_dir="results"):
    import os; os.makedirs(output_dir, exist_ok=True)
    yields_kt = np.logspace(0, 4, 300)
    
    r_xray_ion  = np.sqrt(2.3e11 * yields_kt / 0.4) / 1e3
    r_xray_sgemp= np.sqrt(2.3e11 * yields_kt / 4.0) / 1e3
    r_xray_therm= np.sqrt(2.3e11 * yields_kt / 400.) / 1e3
    r_horizon = np.sqrt((RE_KM + 400)**2 - RE_KM**2)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(yields_kt, r_xray_ion,   color="#f1c40f",  lw=2.5, label="X-ray ionization (0.4 J/m²)")
    ax.loglog(yields_kt, r_xray_sgemp, color="#e67e22",  lw=2.5, label="X-ray SGEMP (4 J/m²)")
    ax.loglog(yields_kt, r_xray_therm, color="#c0392b",  lw=2.5, label="X-ray thermal (400 J/m²)")
    
    ax.axhline(r_horizon, color="black", ls=":", lw=1.5, label=f"Earth horizon (400 km burst)")
    
    # Overlay Simulation Run Context
    ax.axvline(simulated_yield_kt, color="black", ls="-", lw=1.5, alpha=0.5, label=f"Simulated Yield ({simulated_yield_kt} kt)")
    
    if df is not None and not df.empty and 'Distance_km' in df.columns:
        # Plot the CLOSEST satellite of each constellation to the burst
        for const in df['Constellation'].unique():
            min_dist = df[df['Constellation'] == const]['Distance_km'].min()
            ax.scatter([simulated_yield_kt], [min_dist], s=80, marker='*', zorder=5, label=f"Closest {const} in Sim")

    ax.set_xlabel("Total Weapon Yield [kt]", fontsize=12)
    ax.set_ylabel("Threshold Radius for Damage [km]", fontsize=12)
    ax.set_title("Prompt Damage Radius vs. Weapon Yield", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(1, 1e4); ax.set_ylim(10, 1e4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "VIZ2_prompt_damage_radius_empirical.png"), dpi=150)
    plt.close(fig)

def plot_argus_dose_vs_time_empirical(df, yield_kt=1400, burst_alt_km=400, sim_days=30, output_dir="results"):
    import os; os.makedirs(output_dir, exist_ok=True)
    burst_L = (RE_KM + burst_alt_km) / RE_KM
    t_days  = np.linspace(0, 100, 200)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. THE FIX: Define a clean, representative list of constellations to plot
    # Make sure these strings exactly match your keys in CONSTELLATION_CATALOG
    representatives = ["STARLINK (LEO)","GPS (MEO)", "INTELSAT (GEO)"]
    
    if df is not None and not df.empty:
        for const in representatives:
            # Check if the constellation actually exists in the simulation data
            if const not in df['Constellation'].values: 
                continue
                
            const_data = df[df['Constellation'] == const]
            if const_data.empty: continue
                
            mean_L = const_data['Mean_Alt_km'].mean() 
            proxy_L = (RE_KM + mean_L) / RE_KM
            avg_sim_dose = const_data['Corrected_Argus_krad'].mean()
            
            doses = [corrected_argus_dose(proxy_L, burst_L, yield_kt, 2.5, t)[1] for t in t_days]
            line, = ax.semilogy(t_days, doses, lw=2.2, alpha=0.8, label=f"Theoretical {const}")
            
            ax.scatter([sim_days], [avg_sim_dose], color=line.get_color(), s=120, marker='X', 
                       zorder=5, label=f"Simulated {const} (Day {sim_days})")

    ax.axvline(sim_days, color="red", ls="--", alpha=0.5, label="Simulation End Time")
    
    ax.set_xlabel("Days After Detonation", fontsize=12)
    ax.set_ylabel("Cumulative TID [krad(Si)]", fontsize=12)
    ax.set_title("Argus Effect Cumulative TID over Time\n(Representative Orbits)", fontsize=12)
    
    # 2. THE FIX: Move the legend outside the plot frame so it doesn't block the data
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "VIZ3_argus_dose_time_empirical.png"), dpi=150)
    plt.close(fig)

def plot_argus_dose_vs_time_FIXED(df, yield_kt=1400, burst_alt_km=400,
                                   sim_days=30, output_dir="results"):
    
    # ── Altitude bounds for constellation validity filtering ──────
    # Satellites whose mean altitude falls outside these ranges are
    # misclassified, in transfer orbits, or have corrupted SGP4 output.
    ALT_BOUNDS = {
        "STARLINK (LEO)": (300, 700),
        "GPS (MEO)":      (19000, 21500),
        "INTELSAT (GEO)": (34000, 37000),
    }
    SHIELDING = {"STARLINK (LEO)": 2.0, "GPS (MEO)": 10.0, "INTELSAT (GEO)": 8.0}
    INCLINATIONS = {"STARLINK (LEO)": 53.0, "GPS (MEO)": 55.0, "INTELSAT (GEO)": 0.1}
    COLORS = {"STARLINK (LEO)": "#3498db", "GPS (MEO)": "#e67e22", "INTELSAT (GEO)": "#2ecc71"}

    os.makedirs(output_dir, exist_ok=True)
    burst_L = (RE_KM + burst_alt_km) / RE_KM
    t_days  = np.linspace(0, 100, 300)
 
    fig, ax = plt.subplots(figsize=(13, 8))
 
    for const in ["STARLINK (LEO)", "GPS (MEO)", "INTELSAT (GEO)"]:
        col   = COLORS[const]
        shield = SHIELDING[const]
        incl  = INCLINATIONS[const]
        lo, hi = ALT_BOUNDS[const]
 
        # ── BUG 1 + BUG 3 FIX: filter by altitude bounds ─────
        # Removes: (a) negative/zero alt from decayed TLEs
        #          (b) misclassified objects (e.g. INTELSAT 2-F1 at MEO)
        #          (c) Starlink in transfer orbits > 700 km
        const_data = df[(df['Constellation'] == const) &
                        (df['Mean_Alt_km'] >= lo) &
                        (df['Mean_Alt_km'] <= hi)].copy()
 
        if const_data.empty:
            print(f"[WARN] No valid data for {const} after filtering")
            continue
 
        n_removed = len(df[df['Constellation']==const]) - len(const_data)
        print(f"{const}: {len(const_data)} satellites "
              f"(removed {n_removed} out-of-range)")
 
        # ── BUG 2 FIX: use MEDIAN altitude to compute proxy_L ─
        # Mean is skewed by the remaining outliers even after filtering.
        # Median is robust and correctly represents the constellation.
        median_alt   = const_data['Mean_Alt_km'].median()
        proxy_L_eq   = (RE_KM + median_alt) / RE_KM
 
        # Orbit-mean L: inclined orbits sweep to higher mag latitudes
        # L(lat) = r/RE / cos²(lat_mag) — sample over ±incl
        incl_rad = np.radians(incl)
        lats_sampled = np.linspace(-incl_rad, incl_rad, 360)
        L_orbit_dist = np.clip(proxy_L_eq / (np.cos(lats_sampled)**2 + 1e-9),
                               proxy_L_eq, proxy_L_eq * 3.0)
        proxy_L_orbit = float(np.mean(L_orbit_dist))
 
        avg_sim_dose = const_data['Corrected_Argus_krad'].median()
 
        print(f"  Median alt:        {median_alt:.0f} km")
        print(f"  Equatorial L:      {proxy_L_eq:.4f}")
        print(f"  Orbit-mean L:      {proxy_L_orbit:.4f}")
        print(f"  Median sim dose:   {avg_sim_dose:.2f} krad")
 
        # Theory at equatorial L (solid)
        doses_eq  = [corrected_argus_dose_krad(proxy_L_eq, burst_L,
                                                yield_kt, shield, t)
                     for t in t_days]
        line, = ax.semilogy(t_days, doses_eq, color=col, lw=2.5, ls="-",
                             label=f"Theory {const}  (eq. L={proxy_L_eq:.2f})")
 
        # Theory at orbit-mean L (dashed) — brackets the simulation
        doses_orb = [corrected_argus_dose_krad(proxy_L_orbit, burst_L,
                                                yield_kt, shield, t)
                     for t in t_days]
        ax.semilogy(t_days, doses_orb, color=col, lw=1.8, ls="--",
                    label=f"Theory {const}  (orbit L={proxy_L_orbit:.2f}, "
                          f"i={incl:.0f}°)")
 
        # Simulated median marker
        ax.scatter([sim_days], [avg_sim_dose],
                   color=col, s=160, marker='X', zorder=6,
                   edgecolors='white', linewidths=1,
                   label=f"Simulated {const}  "
                         f"(Day {sim_days}, median, N={len(const_data)})")
 
    ax.axvline(sim_days, color="red", ls="--", alpha=0.4, lw=1.5,
               label="Simulation end (Day 30)")
    
    ax.axhline(5, color="black", ls='--', label="5 krad Dose")
    ax.axhline(10, color="purple", ls='--', label="10 krad Dose")
 
    # Physics annotation
    # ax.annotate(
    #     "Simulated X should fall between\n"
    #     "solid (equatorial L) and dashed\n"
    #     "(orbit-mean L) curves.\n"
    #     "Residual gap = geomagnetic dipole\n"
    #     "tilt not captured by simple\n"
    #     "latitude sweep model.",
    #     xy=(31, 50), xytext=(48, 20),
    #     fontsize=8, color="#444",
    #     arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    #     bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#ccc")
    # )
 
    ax.set_xlabel("Days After Detonation", fontsize=12)
    ax.set_ylabel("Cumulative TID [krad(Si)]", fontsize=12)
    ax.set_title(
        "Argus Effect: Cumulative TID over Time  (Corrected)\n"
        "Solid = theory at equatorial L  |  "
        "Dashed = theory at orbit-mean L (inclination-adjusted)  |  "
        "X = simulation median",
        fontsize=10
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(0, 100)
    fig.tight_layout()
 
    path = os.path.join(output_dir, "VIZ3_argus_dose_time_FIXED.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] {path}")

def plot_inclination_empirical_scatter(df, output_dir="results"):
    import os; os.makedirs(output_dir, exist_ok=True)
    if df is None or df.empty or 'Inclination_deg' not in df.columns:
        print("[VIZ6] Missing Inclination data. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    
    for const in df['Constellation'].unique():
        subset = df[df['Constellation'] == const]
        ax.scatter(subset['Inclination_deg'], subset['Corrected_Argus_krad'], 
                   alpha=0.7, s=30, edgecolor='white', linewidth=0.5, label=f"{const}")

    ax.axhline(5.0, color="#8e44ad", ls=":", lw=1.5, label="TID limit COTS (5 krad)")
    ax.axhline(100, color="#e74c3c", ls=":", lw=1.5, label="TID limit GPS (100 krad)")

    ax.set_yscale("log")
    ax.set_xlabel("Orbital Inclination [Degrees]", fontsize=12)
    ax.set_ylabel("Corrected 30-Day Argus TID [krad(Si)]", fontsize=12)
    ax.set_title("Radiation Exposure Profile by Orbital Inclination\n", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, 100)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "VIZ6_inclination_empirical.png"), dpi=150)
    plt.close(fig)

def plot_multi_effect_scatter(df, output_dir="results"):
    """
    Enhanced version of pipeline2.py Visualization 1 (survivability scatter).
    Uses corrected Argus dose and multi-effect status from classify_prompt_effects().
    
    df must have columns: XRay_Jm2, Neutron_nm2, Gamma_radsSi,
                           Corrected_Argus_krad, Constellation
    """
    import os; os.makedirs(output_dir, exist_ok=True)

    if df is None or len(df) == 0:
        print("[VIZ5] No data provided, skipping.")
        return

    # Classify with corrected multi-effect logic
    df = df.copy()
    df["Effects"]        = df.apply(lambda r: classify_prompt_effects(
                                r["XRay_Jm2"], r["Neutron_nm2"], r["Gamma_radsSi"]), axis=1)
    df["Prompt_Status2"] = df["Effects"].apply(prompt_overall_status)

    status_colors = {
        "Fatal/Destroyed":   "#c0392b",
        "Severely Degraded": "#e67e22",
        "Nominal":           "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    for status, grp in df.groupby("Prompt_Status2"):
        col = status_colors.get(status, "#aaa")
        for const, cgrp in grp.groupby("Constellation"):
            ax.scatter(cgrp["XRay_Jm2"],
                       cgrp.get("Corrected_Argus_krad", cgrp.get("Accumulated_Argus_Rads", 0)),
                       c=col, alpha=0.6, s=20, edgecolors="none", label=f"{status} – {const}")

    # RAND thresholds
    ax.axvline(0.4,  color="#f1c40f", ls="--", lw=1.5, label="X-ray ionization (0.4 J/m²)")
    ax.axvline(4.0,  color="#e67e22", ls="--", lw=1.5, label="X-ray SGEMP (4 J/m²)")
    ax.axvline(400,  color="#c0392b", ls="--", lw=1.5, label="X-ray thermal (400 J/m²)")
    ax.axhline(5.0,  color="#8e44ad", ls=":",  lw=1.2, label="Argus TID limit COTS (5 krad)")
    ax.axhline(100,  color="#e74c3c", ls=":",  lw=1.2, label="Argus TID limit GPS (100 krad)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-3) 
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel("Prompt X-Ray Fluence [J m⁻²]", fontsize=12)
    ax.set_ylabel("Corrected 30-Day Argus TID [krad(Si)]", fontsize=12)
    ax.set_title("Satellite Survivability — Prompt vs. Delayed Radiation\n"
                 "RAND RR-A3028-3 thresholds from Conrad et al. 2010)",
                 fontsize=12)
    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2,
               bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "VIZ5_survivability_scatter_corrected.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] {path}")

def plot_altitude_governed_survivability(df, output_dir="results"):
    """
    Revised VIZ5: Altitude-Governed Survivability Scatter.
    Color is mapped to Mean_Alt_km to differentiate orbital regimes.
    Marker shapes differentiate between Military (Hardened) and Commercial assets.
    """
    import os
    import matplotlib.cm as cm
    os.makedirs(output_dir, exist_ok=True)

    if df is None or len(df) == 0:
        print("[VIZ5-Rev] No data provided, skipping.")
        return

    df = df.copy()
    
    # 1. Identify Hardened vs Unhardened for marker shapes
    hardened_keywords = ['GPS', 'NAVSTAR', 'GLONASS', 'BEIDOU', 'MILITARY', 'USA-', 'AEHF']
    df['Is_Hardened'] = df['Constellation'].apply(
        lambda x: any(h in str(x).upper() for h in hardened_keywords)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # 2. Scatter plot with color = Altitude
    # We plot unhardened and hardened separately to use different markers
    # but they share the same color scale (cmap)
    sc_comm = ax.scatter(
        df[~df['Is_Hardened']]['XRay_Jm2'], 
        df[~df['Is_Hardened']]['Corrected_Argus_krad'],
        c=df[~df['Is_Hardened']]['Mean_Alt_km'],
        cmap='viridis', marker='o', s=25, alpha=0.6, edgecolors='none', label='Commercial (COTS)'
    )
    
    sc_mil = ax.scatter(
        df[df['Is_Hardened']]['XRay_Jm2'], 
        df[df['Is_Hardened']]['Corrected_Argus_krad'],
        c=df[df['Is_Hardened']]['Mean_Alt_km'],
        cmap='viridis', marker='D', s=35, alpha=0.9, edgecolors='white', linewidth=0.5, label='Military (Hardened)'
    )

    # 3. Add Colorbar for Altitude
    cbar = plt.colorbar(sc_comm, ax=ax)
    cbar.set_label('Mean Orbital Altitude [km]', fontsize=10)

    # 4. Standard RAND/Conrad Thresholds (Keep as reference lines)
    ax.axvline(400,  color="#c0392b", ls="--", lw=1.5, label="X-ray Thermal Limit")
    ax.axhline(5.0,  color="#8e44ad", ls=":",  lw=1.2, label="COTS TID Limit (5krad)")
    ax.axhline(100,  color="#2c3e50", ls="-.",  lw=1.2, label="Mil-Spec TID Limit (100krad)")

    # 5. Aesthetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-3, right=1e4) 
    ax.set_ylim(bottom=1e-2, top=1e7)
    
    ax.set_xlabel("Prompt X-Ray Fluence [J m⁻²]", fontsize=12)
    ax.set_ylabel("Corrected 30-Day Argus TID [krad(Si)]", fontsize=12)
    ax.set_title("The Altitude-Vulnerability Chasm\nPrompt vs. Delayed Radiation Exposure", fontsize=14, pad=15)
    
    ax.grid(True, which="both", ls="-", alpha=0.1)
    ax.legend(loc='lower left', fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "VIZ5_altitude_survivability_scatter.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")

# =============================================================================
# EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    TLE_FILE = '3le.txt'
    OUT = "results4"
    os.makedirs(OUT, exist_ok=True)
    all_results = []
    
    # LOAD THE FILE EXACTLY ONCE HERE
    print("[+] Parsing massive TLE catalog into memory...")
    master_satellite_list = load.tle_file(TLE_FILE)
    
    for const_key in CONSTELLATION_CATALOG.keys():
        try:
            shielding_mm = CONSTELLATION_SHIELDING.get(const_key, (2.5, False, 5.0, ""))[0]
            
            # PASS THE PRE-LOADED LIST INTO THE CLASS
            sim = HANDSimulationAdvanced(TLE_FILE, 
                                         target_constellation=const_key, 
                                         preloaded_sats=master_satellite_list)
            
            if len(sim.target_sats) == 0: continue
            
            # Starfish Prime-esque burst (400km, 1400kT)
            sim.simulate_burst(16.7, -169.5, 400.0, 1400.0, (2026, 5, 5, 12, 0, 0))
            
            df_prompt = sim.calculate_prompt_effects()
            df_delayed = sim.calculate_delayed_argus_effect(days=30, shielding_al_mm=shielding_mm)
            
            # MERGE ON NORAD ID TO PREVENT CARTESIAN EXPLOSION
            df_merged = pd.merge(df_prompt, df_delayed, on=['NORAD_ID', 'Name'])
            
            df_merged['Constellation'] = const_key
            all_results.append(df_merged)
            
        except Exception as e:
            print(f"Failed to process {const_key}: {e}")

    if all_results:
        final_dataset = pd.concat(all_results, ignore_index=True)
        final_dataset.to_csv(f"{OUT}/hand_simulation_final.csv", index=False)
        
        print("[+] Generating Visualizations...")
        plot_xray_fluence_vs_distance_empirical(final_dataset, output_dir=OUT)
        plot_prompt_damage_radius_vs_yield_empirical(final_dataset, output_dir=OUT)
        plot_argus_dose_vs_time_FIXED(final_dataset, output_dir=OUT)
        plot_multi_effect_scatter(final_dataset, output_dir=OUT)
        plot_inclination_empirical_scatter(final_dataset, output_dir=OUT)
        plot_altitude_governed_survivability(final_dataset, output_dir=OUT)
        
        print("[+] Execution Complete.")
    else:
        print("No satellites processed. Ensure your 3LE file is correctly named and populated.")