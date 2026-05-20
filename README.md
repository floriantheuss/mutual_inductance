# mutual_inductance

Analysis toolkit for two-coil mutual inductance measurements on superconducting thin films. Given the geometry of a drive/pickup coil pair and a lock-in amplifier measurement of the mutual inductance vs. temperature, this code extracts the London penetration depth λ(T) of the film.

---

## Physics background

A drive coil drives an AC current through a superconducting thin film. A pickup coil on the other side measures the transmitted flux — the mutual inductance M between the two coils. Because the film screens the magnetic field, M depends on the quality of the superconducting screening, which in turn depends on the London penetration depth λ. By comparing the measured M(T) to a theoretical model of M(λ), one can extract λ(T) across the superconducting transition.

---

## Installation

```bash
pip install numpy scipy matplotlib ray psutil lmfit pandas
```

| Package | Purpose |
|---|---|
| `numpy` | Array operations and file I/O |
| `scipy` | Bessel functions, numerical integration, interpolation |
| `matplotlib` | Diagnostic plots at every processing step |
| `ray` | Parallel evaluation of the MI integral across CPU cores |
| `psutil` | Physical CPU count for Ray worker pool sizing |
| `lmfit` | Least-squares fitting of coil distance and penetration depth |
| `pandas` | Rolling-median outlier filter in `MIData` |

---

## Module overview

### `calculate_mutual_inductance.py` — `CalcMutualInductance`

Numerically calculates the theoretical mutual inductance M(λ) for a given coil geometry. The coil is modelled as a stack of individual circular loops determined by winding parameters (number of layers, turns per layer, wire diameter). Ray is used to parallelize the integral evaluation across all unique loop-pair separations.

Key methods:

| Method | What it does |
|---|---|
| `calc_MI_PD_array(lambda_array)` | Main entry point. Computes M and M/M∞ for each λ in the array; saves a lookup table to disk. |
| `fit_coil_distance(MIx_data)` | Fits the coil separation h to a single measured M value (normal-state calibration). |
| `fit_penetration_depth(MIx_data)` | Fits λ to a single measured M value directly (alternative to the lookup table approach, but much slower). |
| `save_MI_array(...)` | Saves the λ → M, M/M∞ lookup table as a CSV with a full header describing the coil geometry. |

---

### `mutual_inductance_calibration.py` — `MICalibration`

Processes a calibration measurement using a **thick** superconducting film (λ ≪ d_film), where the film acts as a perfect magnetic screen at low T. This measurement is used to extract two corrections that are later applied to every sample measurement:

1. **Phase angle** — cable and capacitive coupling mix the X and Y lock-in channels. Deep in the SC phase the real part of M should be zero, so any offset gives the mixing angle directly.
2. **Field leakage** — some flux bypasses the sample ("leaks" around it) and reaches the pickup coil even when the film fully screens. This is estimated from the low-T average of M_x after phase correction and subtracted from all subsequent data.

Key method: `perform_calibration(Tphase, Tleakage)` — runs both corrections and produces diagnostic plots.

---

### `mutual_inductance_data.py` — `MIData`

Processes a sample mutual inductance measurement end-to-end.

Key methods:

| Method | What it does |
|---|---|
| `process_data()` | Applies phase correction → converts to MI → subtracts leakage and residual capacitance → normalizes to normal-state MI. |
| `convert_MI_to_PD(lookup_table_path)` | Inverts the M/M∞(λ) lookup table to get λ(T) by interpolation. |
| `save_results(filename)` | Saves all intermediate and final quantities to a CSV. |

---

### `find_angle.py`

Standalone script. Given a single (X, Y) lock-in reading in the normal state, computes the cable phase mixing angle. Useful for a quick sanity check before running the full calibration pipeline.

---

## Typical analysis workflow

```
Coil geometry
     │
     ▼
1. Build lookup table   CalcMutualInductance.calc_MI_PD_array(lambda_array)
   M/M∞  vs  λ          → saves "lookup_table.csv"
     │
     ▼
2. Calibration run      MICalibration(X_cal, Y_cal, T_cal, f, I)
   (thick SC film)      .perform_calibration(Tphase=[0, Tmax_sc],
                                             Tleakage=[0, Tmax_sc])
                        → angle, leakage
     │
     ▼
3. Sample measurement   MIData(X, Y, T, f, I,
   processing               leakage=leakage,
                             correction_angle=angle)
                        .process_data()
                        → MIx_norm(T)
     │
     ▼
4. Convert to λ(T)      .convert_MI_to_PD("lookup_table.csv")
                        → penetration_depth(T)
                        → plots λ(T) and λ₀²/λ(T)²
```

### Step 1 — generate a lookup table

```python
from mutual_inductance.calculate_mutual_inductance import CalcMutualInductance
import numpy as np

MI = CalcMutualInductance(
    n_dl=4, n_dt=13,        # drive coil: 4 layers, 13 turns/layer
    n_pl=25, n_pt=16,       # pickup coil: 25 layers, 16 turns/layer
    d_film=40e-9,           # film thickness (m)
    r_d=1.5e-4, r_p=1.5e-4,# coil inner radii (m)
    h=1.07e-3,              # coil separation (m)
    d_wire=19e-6,           # wire diameter (m)
)

lambda_array = np.logspace(-8, -4, 200)   # 10 nm → 100 µm
MI_array, MI_norm = MI.calc_MI_PD_array(lambda_array, save_name='lookup_table.csv')
```

### Step 2 — calibration

```python
from mutual_inductance.mutual_inductance_calibration import MICalibration
import numpy as np

# X, Y: lock-in voltages (V); T: temperature (K); f: frequency (Hz); I: drive current (A)
cal = MICalibration(X_cal, Y_cal, T_cal, f=317, I=100e-6)
cal.perform_calibration(
    Tphase=[0, 4],      # temperature range deep in SC phase for phase correction
    Tleakage=[0, 4],    # same range for leakage estimation
    save_name='calibration.csv',
)
angle   = cal.angle     # mixing angle (rad)
leakage = cal.leakage   # MI_x leakage (H)
```

### Step 3 & 4 — process sample and extract λ(T)

```python
from mutual_inductance.mutual_inductance_data import MIData

data = MIData(X, Y, T, f=317, I=100e-6,
              leakage=leakage,
              correction_angle=angle)
data.process_data(save_name='sample_processed.csv')
data.convert_MI_to_PD('lookup_table.csv', save_path='sample_with_lambda.csv')
```

---

## Output files

| File | Contents |
|---|---|
| `lookup_table.csv` | λ (m), M (H), M/M∞ — theoretical lookup table for fixed coil geometry |
| `calibration.csv` | T, X, Y, X_corr, Y_corr, MI_x, MI_y, MI_x_corr, MI_y_corr |
| `sample_processed.csv` | All intermediate quantities + MI_x_norm, MI_y_norm |
| `sample_with_lambda.csv` | All of the above + penetration depth λ(T) (m) |
