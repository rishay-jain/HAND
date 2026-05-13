# PUNCH
### Prediction of Unmitigated Nuclear Constellation Hazards

A physics-based simulation pipeline for evaluating prompt and delayed radiation effects on satellite constellations from a high-altitude nuclear detonation (HAND).

**Author:** Rishay Jain
**Date:** May 2026

---

## Overview

PUNCH models the two primary radiation threat regimes produced by a HAND:

**Prompt effects** — X-ray, neutron, and gamma-ray fluences radiated instantaneously from the burst point to all satellites within line of sight, computed at the detonation epoch using inverse-square propagation with atmospheric attenuation.

**Delayed effects (Argus)** — Enhancement of the Van Allen radiation belts by fission electrons injected into Earth's magnetic field. Using a bi-exponential decay model calibrated to the 1962 Starfish Prime test, the pipeline integrates accumulated total ionizing dose (TID) for each satellite over a configurable post-detonation window, accounting for the satellite's full orbital L-shell distribution.

The baseline scenario replicates Starfish Prime at a 2026 epoch: a 1,400 kt detonation at 400 km altitude over Johnston Atoll. All 30,000+ Space Surveillance Network tracked objects are loaded from a live TLE catalog; twelve strategically selected constellations spanning LEO, MEO, and GEO are assessed.

The pipeline produces a per-satellite CSV dataset and a suite of publication-quality figures.

---

## Results Summary

| Metric | Value |
|---|---|
| Total satellites tracked | 11,715 |
| Destroyed instantly (prompt X-ray) | 1,171 (10.0%) |
| Lost within 30 days (Argus TID) | 10,880 (92.9%) |
| Surviving satellites | 835 (7.1%) |

| Constellation | Total | Instant Loss | 30-Day Loss | Survival |
|---|---|---|---|---|
| STARLINK (LEO) | 10,023 | 1,035 (10.3%) | 9,756 (97.3%) | 2.7% |
| ONEWEB | 654 | 93 (14.2%) | 654 (100%) | 0% |
| IRIDIUM | 216 | 12 (5.6%) | 214 (99.1%) | 0.9% |
| PLANET (LEO) | 152 | 19 (12.5%) | 152 (100%) | 0% |
| GALILEO | 30 | 0 (0%) | 30 (100%) | 0% |
| GPS (MEO) | 78 | 0 (0%) | 0 (0%) | 100% |
| GLONASS | 143 | 0 (0%) | 0 (0%) | 100% |
| BEIDOU (MEO) | 56 | 0 (0%) | 0 (0%) | 100% |
| INTELSAT (GEO) | 135 | 1 (0.7%) | 1 (0.7%) | 99.3% |
| MILITARY | 222 | 11 (5.0%) | 73 (32.9%) | 67.1% |
| AEHF | 6 | 0 (0%) | 0 (0%) | 100% |

---

## Physics Models

All formulas are cited to primary literature. Every physics constant is traceable to a specific reference.

### Prompt X-Ray Fluence
From Conrad et al. (2010), as reported in Snyder et al. (2025):

```
Φ_x = 2.3×10¹¹ × Y / r²   [J m⁻²]
```

where `Y` is total yield in kt and `r` is slant range in metres. Atmospheric attenuation follows Glasstone & Dolan (1977): X-rays are fully transmitted above 90 km, linearly attenuated between 90–55 km, and fully absorbed below 55 km. Gamma and neutron radiation is transmitted above 25 km and absorbed below.

### Prompt Damage Thresholds
From Conrad et al. (2010):

| Mechanism | Threshold |
|---|---|
| X-ray ionization upset | Φ_x ≥ 0.4 J m⁻² |
| X-ray SGEMP | Φ_x ≥ 4.0 J m⁻² |
| X-ray thermal/structural | Φ_x ≥ 400 J m⁻² |
| Neutron electronic upset | Φ_n ≥ 10¹⁰ n m⁻² |
| Neutron lattice damage | Φ_n ≥ 10¹⁶ n m⁻² |
| Gamma TID upset | D_γ ≥ 10³ rad(Si) |
| Gamma TID fatal | D_γ ≥ 10⁴ rad(Si) |

All three mechanisms are evaluated independently; a satellite can trigger multiple effects simultaneously.

### L-Shell Computation
Tilted dipole approximation with IGRF pole at 80.8°N, 72.6°W (McIlwain 1961):

```
sin(λ_mag) = sin(φ)sin(φ_p) + cos(φ)cos(φ_p)cos(λ − λ_p)
L = (r / R_E) / cos²(λ_mag)
```

### AE8-MAX Baseline Flux
Two-term Gaussian sum fit to digitised AE8-MAX data (Vette 1991) for E > 1 MeV electrons, capturing the inner belt peak at L ≈ 1.5 and the outer belt peak at L ≈ 4.5.

### HAND Electron Injection and Decay
Bi-exponential decay model from Conrad et al. (2008), calibrated to Starfish Prime:

```
ΔΦ(L, t) = A × exp(−(L − L_burst)² / 2σ_L²) × [α exp(−t/τ₁) + (1−α) exp(−t/τ₂)]
```

- `τ₁ = 40 days` — fast decay (wave-particle pitch-angle scattering)
- `τ₂ = 500 days` — slow decay (Coulomb drag)
- Peak enhancement `A` calibrated to ×100 AE8-MAX at the burst L-shell for a 1,400 kt burst
- Yield scaling: `A ∝ Y^0.7` (Christofilos 1959; Hess 1963)

### Dose Model (SHIELDOSE-2)
From Seltzer (1994), parameterised as a function of areal density `d = ρ_Al × t` [g cm⁻²]:

```
f(d) = D₀ × exp(−d/λ)        for d < 0.05 g/cm²
f(d) = D₀ × (1 + d/λ)^−β    for d ≥ 0.05 g/cm²
```

with `D₀ = 8.5×10⁻⁴ rad(Si)/day per e cm⁻² s⁻¹`, `λ = 0.90 g cm⁻²`, `β = 2.0`.

TID is integrated analytically over the dose window, yielding a closed-form result.

---

## Installation

```bash
git clone https://github.com/yourusername/punch.git
cd punch
pip install numpy pandas matplotlib skyfield
```

The Skyfield DE421 ephemeris file (`de421.bsp`) is downloaded automatically on first run.

---

## TLE Catalog

Download a current 3LE catalog from [Space-Track.org](https://www.space-track.org) (free account required). The pipeline expects the standard 3LE format: name line followed by TLE lines 1 and 2.

Save the file and set its path at the top of `punch.py`:

```python
TLE_FILEPATH = '3le.txt'
```

A full catalog of ~30,000 objects is recommended. Smaller filtered catalogs will work but will reduce constellation sample sizes.

---

## Usage

```bash
python punch.py
```

Output is written to the directory set by `OUTPUT_DIRECTORY` at the top of the file (default: `results4/`).

### Changing the Detonation Scenario

Edit the `simulate_burst_condition()` call in the execution block:

```python
sim.simulate_burst_condition(
    burst_lat=16.7,                        # Sub-burst latitude [degrees]
    burst_lon=-169.5,                      # Sub-burst longitude [degrees]
    burst_alt_km=400.0,                    # Burst altitude [km]
    yield_kt=1400.0,                       # Total yield [kt]
    burst_time_utc=(2026, 5, 5, 12, 0, 0) # UTC detonation time
)
```

### Adding or Removing Constellations

Edit `CONSTELLATION_CATALOG` to add name strings that will be matched against TLE name fields. Add a corresponding entry to `CONSTELLATION_SHIELDING`. Commented-out entries for GLOBALSTAR, ORBCOMM, WORLDVIEW, and INMARSAT are included as starting points.

```python
CONSTELLATION_CATALOG = {
    "MY_CONSTELLATION": ("NAME_STRING_1", "NAME_STRING_2"),
    ...
}

CONSTELLATION_SHIELDING = {
    "MY_CONSTELLATION": (shielding_mm, rad_hardened, tid_krad_limit, "notes"),
    ...
}
```

### Changing the Integration Window

The delayed effects window defaults to 30 days. Change it in the execution block:

```python
df_delayed = sim.calculate_delayed_argus_effect(days=90, shielding_al_mm=shielding_mm)
```

---

## Outputs

### CSV
`hand_simulation_final.csv` — one row per satellite:

| Column | Description |
|---|---|
| NORAD_ID | SSN catalog number |
| Name | TLE name string |
| Distance_km | Slant range from burst at detonation epoch [km] |
| XRay_Jm2 | Prompt X-ray fluence [J m⁻²] |
| Neutron_nm2 | Prompt neutron fluence [n m⁻²] |
| Gamma_radsSi | Prompt gamma TID [rad(Si)] |
| Effects_List | All triggered damage mechanisms |
| Prompt_Status | Worst-case prompt classification |
| Corrected_Argus_krad | Accumulated Argus TID over integration window [krad(Si)] |
| Inclination_deg | Orbital inclination [degrees] |
| Mean_Alt_km | Mean altitude over 24-hr propagation window [km] |
| Constellation | Constellation key |

### Figures

| File | Description |
|---|---|
| `VIZ1_xray_fluence_empirical.png` | Simulated X-ray fluence vs. slant range overlaid on the theoretical 1/r² curve. Primary model verification against Conrad et al. (2010). |
| `VIZ2_prompt_damage_radius_empirical.png` | Threshold damage radius vs. weapon yield for all three X-ray damage levels, with the closest simulated satellite per constellation marked at the simulated yield. |
| `VIZ3_argus_dose_time_FIXED.png` | Cumulative TID vs. days post-detonation for LEO, MEO, and GEO representative constellations. Solid = theory at equatorial L; dashed = theory at orbit-mean L (inclination-adjusted); X marker = simulation median. |
| `VIZ4_survivability_scatter_corrected.png` | Per-satellite prompt X-ray fluence vs. 30-day Argus TID, coloured by survivability status with RAND/Conrad damage thresholds overlaid. |
| `VIZ5_altitude_survivability_scatter.png` | Same axes as VIZ4, with colour mapped to orbital altitude and marker shape distinguishing commercial vs. hardened assets. |
| `VIZ6_inclination_empirical.png` | 30-day Argus TID vs. orbital inclination, illustrating how inclined orbits sweep to higher L-shells and accumulate greater dose. |

---

## Shielding Assumptions

Default values are estimates from published literature (Conrad et al. 2010; Nordin & Kong 1999; Stassinopoulos & Raymond 1988). They are not verified manufacturer specifications and should be treated as representative values for population-level assessment.

| Constellation | Shielding (mm Al) | Rad-Hardened | TID Limit (krad) |
|---|---|---|---|
| STARLINK (LEO) | 2.0 | No | 5 |
| ONEWEB | 3.0 | No | 10 |
| IRIDIUM | 3.0 | No | 10 |
| PLANET (LEO) | 2.5 | No | 5 |
| GALILEO | 8.0 | No | 30 |
| ISS | 15.0 | No | 50 |
| GPS (MEO) | 10.0 | Yes | 100 |
| GLONASS | 8.0 | Yes | 75 |
| BEIDOU (MEO) | 10.0 | Yes | 100 |
| INTELSAT (GEO) | 8.0 | Yes | 50 |
| MILITARY | 15.0 | Yes | 200 |
| AEHF | 20.0 | Yes | 300 |

---

## Known Limitations

**SGP4 propagation artifacts.** A fraction of TLEs in any live catalog correspond to decayed or near-reentry objects. SGP4 evaluation of these objects produces anomalous altitudes. `plot_argus_dose_vs_time` filters each constellation to a physically plausible altitude range before computing representative statistics. The raw CSV retains all rows.

**Tilted dipole L-shell.** The L-shell calculation uses a tilted dipole approximation rather than a full IGRF field model. For high-inclination orbits this underestimates the orbit-mean L-shell by approximately 10–20%, producing a corresponding underestimate of Argus dose for polar-orbit constellations.

**Single-weapon scenario.** The pipeline models a single detonation. Multiple detonations at different L-shells would produce compounding belt enhancements not captured here.

**Static shielding.** All satellites within a constellation are assigned identical shielding parameters. Real constellations have variation across bus generations and mission types.

**Epoch-instantaneous prompt effects.** The prompt calculation evaluates each satellite at a single instant (the detonation epoch). Satellites on the far side of Earth at that epoch receive no prompt dose regardless of orbital geometry.

---

## References

Conrad, E. E., Gurtman, G. A., Kweder, G., Mandell, M. J., & White, W. W. (2010). *Collateral damage to satellites from an EMP attack.* DTRA-IR-10-22.

Conrad, R., Vampola, A. L., & Gaines, E. (2008). Decay of artificial radiation belts. *Journal of Geophysical Research: Space Physics, 113*(A2). https://doi.org/10.1029/2007JA012396

Glasstone, S., & Dolan, P. J. (Eds.). (1977). *The effects of nuclear weapons* (3rd ed.). U.S. Department of Defense and U.S. Department of Energy.

Hess, W. N. (1963). The artificial radiation belt made on July 9, 1962. *Journal of Geophysical Research, 68*(3), 667–683. https://doi.org/10.1029/JZ068i003p00667

McIlwain, C. E. (1961). Coordinates for mapping the distribution of magnetically trapped particles. *Journal of Geophysical Research, 66*(11), 3681–3691. https://doi.org/10.1029/JZ066i011p03681

Nordin, P., & Kong, M. K. (1999). Hardness and survivability requirements. In J. R. Wertz & W. J. Larson (Eds.), *Space mission analysis and design* (3rd ed.). Microcosm Publishing.

Seltzer, S. M. (1994). Updated calculations for routine space-shielding radiation dose estimates: SHIELDOSE-2. *IEEE Transactions on Nuclear Science, 41*(6), 2016–2022. https://doi.org/10.1109/23.340589

Snyder, D., Putney, A., Leidy, E. N., Hartnett, G. S., & Bonomo, J. (2025). *The effects of high-altitude nuclear explosions on non-military satellites.* RAND Corporation. https://www.rand.org/t/RRA3028-3

Stassinopoulos, E. G., & Raymond, J. P. (1988). The space radiation environment for electronics. *Proceedings of the IEEE, 76*(11), 1423–1442. https://doi.org/10.1109/5.90113

van Allen, J. A. (1966). Spatial distribution and time decay of the intensities of geomagnetically trapped electrons from the high altitude nuclear burst of July 1962. In B. M. McCormac (Ed.), *Radiation trapped in the Earth's magnetic field* (Vol. 5, pp. 207–220). D. Reidel Publishing.

Vette, J. I. (1991). *The AE-8 trapped electron model environment.* NSSDC/WDC-A-R&S 91-24.

Zink, J., Nöldeke, C. M., Gaißer, S., & Klinkner, S. (2026). Analysing single event upsets in low Earth orbit considering the geomagnetic field. *Advances in Space Research, 77*(10), 10520–10528. https://doi.org/10.1016/j.asr.2026.03.051

---

## License

MIT License. See `LICENSE` for details.

This repository contains no classified or export-controlled information. All physics models are derived entirely from publicly available literature. TLE data sourced from Space-Track.org is subject to Space-Track's terms of use.