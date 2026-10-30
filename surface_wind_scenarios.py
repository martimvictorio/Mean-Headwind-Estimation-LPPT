"""
Surface wind (sensors 3A and 3B) for the four weather scenarios of the thesis
(equivalent of Figures 4.3-4.6 and Table 4.1, but using the anemometers
instead of METAR).

Data assumptions (verified against LPPT METAR):
  - Files WIND_3A_MSG_RAW_<DD>.csv and WIND_3B_MSG_RAW_<DD>.csv are in the
    same folder as this script.
  - Format: ';'-separated, columns: <timestamp>;Dir;Speed
  - Timestamps are UTC, one sample every ~3 s (each sample = mean of last 3 s)
  - Speed in m/s, direction in degrees (true, "wind from"), 1..360
Scenario windows are given in UTC, as in the ASTERIX data ('tod', I062/070, is UTC).

Change of 29-10-2026: the windows were previously treated as local time and converted
to UTC (Europe/Lisbon), which shifted them 1 h earlier than the aircraft data on the
UTC+1 days (29 March, 11 April, 6 May). No time zone conversion is done any more.

Method:
  1. Headwind and crosswind are computed for each sensor separately (Eqs. 3.1-3.2).
  2. Each 3A sample is paired with the nearest 3B sample (|dt| <= 2 s).
  3. Paired samples: arithmetic mean of the two sensors' components.
     Unpaired samples: the available sensor only (thesis rule).
  Because the projection onto the runway axis is linear, this is identical to
  projecting the vector mean of the two sensors, and avoids the 359/001 deg issue.

Dependencies: numpy, pandas, matplotlib.

Usage:
    python surface_wind_scenarios.py          # run checks, process scenarios
    python surface_wind_scenarios.py --test   # run sanity checks only
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Central configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR                              # CSV files next to this script
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "surface_wind_output")
FILE_NAME = "WIND_{sensor}_MSG_RAW_{day}.csv"

RUNWAY_TRUE_BEARING_DEG = 22.72      # Runway 02 true bearing (thesis, Sec. 3.1.2)
MS_TO_KT = 3600.0 / 1852.0           # exact: 1 kt = 1852 m / 3600 s
PAIRING_TOLERANCE_S = 2.0            # max time gap to treat a 3A and a 3B sample as simultaneous
Y_MARGIN_KT = 1.0                    # margin added to the common y-axis limits
AVERAGING_PERIOD = "1min"            # length of the averaging period
MIN_SAMPLES_PER_PERIOD = 10          # ~20 samples expected per minute; fewer -> period discarded


@dataclass(frozen=True)
class Scenario:
    name: str
    file_day: str        # day suffix used in the file names (e.g. "11")
    date: str            # YYYY-MM-DD
    start_utc: str       # HH:MM, UTC
    end_utc: str         # HH:MM, UTC


SCENARIOS = [
    Scenario("High Headwind, High Crosswind", "11", "2026-04-11", "16:00", "20:00"),
    Scenario("High Headwind, Low Crosswind",  "29", "2026-03-29", "08:00", "12:00"),
    Scenario("Low Headwind, High Crosswind",  "06", "2026-05-06", "16:00", "20:00"),
    Scenario("Low Headwind, Low Crosswind",   "16", "2026-03-16", "12:00", "16:00"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def sensor_path(data_dir: str, sensor: str, day: str) -> str:
    path = os.path.join(data_dir, FILE_NAME.format(sensor=sensor, day=day))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Anemometer file not found: {path}")
    return path


def load_sensor(path: str) -> pd.DataFrame:
    """Read one anemometer file -> DataFrame indexed by UTC time, columns dir_deg, speed_ms."""
    raw = pd.read_csv(path, sep=";", dtype={"Dir": str})
    raw.columns = ["time", "dir_deg", "speed_ms"]
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["time"]).dt.tz_localize("UTC"),
        "dir_deg": pd.to_numeric(raw["dir_deg"], errors="coerce"),
        "speed_ms": pd.to_numeric(raw["speed_ms"], errors="coerce"),
    }).dropna()
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    _check_sensor_ranges(df, path)
    return df


def _check_sensor_ranges(df: pd.DataFrame, path: str) -> None:
    """Fail loudly on physically impossible values."""
    if not df["dir_deg"].between(0, 360).all():
        raise ValueError(f"Direction outside [0, 360] in {path}")
    if (df["speed_ms"] < 0).any():
        raise ValueError(f"Negative speed in {path}")


# ---------------------------------------------------------------------------
# Runway components
# ---------------------------------------------------------------------------
def runway_components(speed, dir_deg, runway_deg=RUNWAY_TRUE_BEARING_DEG):
    """
    Headwind and crosswind relative to the runway (thesis Eqs. 3.1-3.2).
    Positive headwind = wind against the aircraft; positive crosswind = from the right.
    Units follow the input speed.
    """
    delta = np.deg2rad(dir_deg - runway_deg)
    return speed * np.cos(delta), speed * np.sin(delta)


def add_components_kt(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with headwind_kt and crosswind_kt for one sensor."""
    hw, cw = runway_components(df["speed_ms"], df["dir_deg"])
    return pd.DataFrame({"headwind_kt": hw * MS_TO_KT,
                         "crosswind_kt": cw * MS_TO_KT}, index=df.index)


# ---------------------------------------------------------------------------
# Combining sensors
# ---------------------------------------------------------------------------
def pair_sensors(a: pd.DataFrame, b: pd.DataFrame,
                 tolerance_s: float = PAIRING_TOLERANCE_S) -> pd.DataFrame:
    """
    Pair component series of 3A (a) and 3B (b), which are sampled at different
    instants (offsets of 1-2 s). Each 3A sample is paired with the nearest 3B
    sample within the tolerance; each 3B sample is used at most once.
    3B samples left without partner are kept as single-sensor rows.
    Output columns: hw_a, cw_a, hw_b, cw_b (NaN where a sensor is missing).
    """
    tol = pd.Timedelta(seconds=tolerance_s)
    a_ = (a.rename_axis("time").reset_index()
            .rename(columns={"headwind_kt": "hw_a", "crosswind_kt": "cw_a"}))
    b_ = (b.rename_axis("time_b").reset_index()
            .rename(columns={"headwind_kt": "hw_b", "crosswind_kt": "cw_b"}))

    merged = pd.merge_asof(a_, b_, left_on="time", right_on="time_b",
                           direction="nearest", tolerance=tol)

    # A 3B sample may be nearest to two 3A samples: keep only its closest partner.
    merged["gap"] = (merged["time"] - merged["time_b"]).abs()
    dup = merged["time_b"].notna() & merged.duplicated("time_b", keep=False)
    if dup.any():
        best = merged[dup].groupby("time_b")["gap"].idxmin()
        drop = merged[dup].index.difference(best.values)
        merged.loc[drop, ["hw_b", "cw_b", "time_b"]] = np.nan

    used_b = set(merged["time_b"].dropna())
    b_only = b_[~b_["time_b"].isin(used_b)].rename(columns={"time_b": "time"})
    combined = pd.concat([merged.drop(columns=["time_b", "gap"]), b_only], ignore_index=True)
    return combined.sort_values("time").set_index("time")


def average_sensors(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean of the two sensors' components; single-sensor rows use that sensor only."""
    out = pd.DataFrame(index=paired.index)
    out["headwind_kt"] = paired[["hw_a", "hw_b"]].mean(axis=1, skipna=True)
    out["crosswind_kt"] = paired[["cw_a", "cw_b"]].mean(axis=1, skipna=True)
    # Absolute value applied AFTER averaging the sensors (see thesis Sec. 4.1.1).
    out["abs_crosswind_kt"] = out["crosswind_kt"].abs()
    out["n_sensors"] = paired[["hw_a", "hw_b"]].notna().sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Temporal averaging
# ---------------------------------------------------------------------------
def period_means(wind: pd.DataFrame, period: str = AVERAGING_PERIOD,
                 min_samples: int = MIN_SAMPLES_PER_PERIOD) -> pd.DataFrame:
    """
    Mean of the PREVIOUS period, labelled at its end: the value at 16:01:00
    uses the samples in ]16:00:00, 16:01:00] (no future information).
    The crosswind is averaged with its sign and only then made absolute.
    Periods with fewer than min_samples samples are discarded.
    """
    grouped = wind[["headwind_kt", "crosswind_kt"]].resample(period, label="right", closed="right")
    out = grouped.mean()
    out["n_samples"] = grouped.size()
    out = out[out["n_samples"] >= min_samples].copy()
    out["abs_crosswind_kt"] = out["crosswind_kt"].abs()
    return out


# ---------------------------------------------------------------------------
# Scenario window and summary
# ---------------------------------------------------------------------------
def utc_window(date: str, start: str, end: str):
    """UTC timestamps of the scenario window (no time zone conversion)."""
    t0 = pd.Timestamp(f"{date} {start}", tz="UTC")
    t1 = pd.Timestamp(f"{date} {end}", tz="UTC")
    return t0, t1


def select_window(df: pd.DataFrame, t0, t1) -> pd.DataFrame:
    """
    Window ]t0, t1] for end-labelled period means: the first value is the
    mean of the first minute inside the window, the last one ends at t1.
    """
    return df[(df.index > t0) & (df.index <= t1)]


def summarise(sc: Scenario, wind: pd.DataFrame, t0, t1) -> dict:
    """Window statistics (thesis Table 4.1 style, plus data-quality counts)."""
    return {
        "scenario": sc.name,
        "date": sc.date,
        "utc_window": f"{t0:%H:%M}-{t1:%H:%M}",
        "n_periods": len(wind),
        "n_samples_3s": int(wind["n_samples"].sum()),
        "mean_HW_kt": round(wind["headwind_kt"].mean(), 1),
        "mean_absCW_kt": round(wind["abs_crosswind_kt"].mean(), 1),
        "std_HW_kt": round(wind["headwind_kt"].std(), 1),
        "std_CW_kt": round(wind["crosswind_kt"].std(), 1),
        "min_HW_kt": round(wind["headwind_kt"].min(), 1),
        "max_HW_kt": round(wind["headwind_kt"].max(), 1),
        "max_absCW_kt": round(wind["abs_crosswind_kt"].max(), 1),
    }


# ---------------------------------------------------------------------------
# Plotting (labels in British English)
# ---------------------------------------------------------------------------
def common_y_limits(windows: list[pd.DataFrame]) -> tuple[float, float]:
    """Same y-axis range for all scenarios, so they can be compared visually."""
    all_values = pd.concat([pd.concat([w["headwind_kt"], w["abs_crosswind_kt"]]) for w in windows])
    y_min = min(0.0, all_values.min()) - Y_MARGIN_KT
    y_max = all_values.max() + Y_MARGIN_KT
    return y_min, y_max


def plot_scenario(sc: Scenario, wind: pd.DataFrame, summary: dict,
                  y_limits: tuple[float, float], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(wind.index, wind["headwind_kt"], color="teal", lw=1.0,
            label="Headwind (1 min mean)")
    ax.axhline(summary["mean_HW_kt"], color="teal", ls="--",
               label=f"Mean headwind ({summary['mean_HW_kt']:.1f} kt)")
    ax.plot(wind.index, wind["abs_crosswind_kt"], color="darkorange", lw=1.0,
            label="Absolute crosswind (1 min mean)")
    ax.axhline(summary["mean_absCW_kt"], color="darkorange", ls="--",
               label=f"Mean absolute crosswind ({summary['mean_absCW_kt']:.1f} kt)")
    ax.axhline(0, color="grey", lw=0.5)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Wind component speed (kt)")
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz="UTC"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, f"surface_wind_{sc.date}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_scenario(sc: Scenario, data_dir: str) -> tuple[pd.DataFrame, dict]:
    """Load both sensors, compute components, average, cut the scenario window."""
    comp_a = add_components_kt(load_sensor(sensor_path(data_dir, "3A", sc.file_day)))
    comp_b = add_components_kt(load_sensor(sensor_path(data_dir, "3B", sc.file_day)))
    wind_3s = average_sensors(pair_sensors(comp_a, comp_b))
    wind = period_means(wind_3s)

    t0, t1 = utc_window(sc.date, sc.start_utc, sc.end_utc)
    window = select_window(wind, t0, t1)
    if window.empty:
        raise ValueError(f"No wind data in window for {sc.date}")
    return window, summarise(sc, window, t0, t1)


def main(data_dir: str = DATA_DIR, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1: compute all scenarios (needed for the common y-axis scale).
    results = [(sc, *process_scenario(sc, data_dir)) for sc in SCENARIOS]
    y_limits = common_y_limits([window for _, window, _ in results])

    # Pass 2: plot and save.
    rows = []
    for sc, window, summary in results:
        summary["figure"] = plot_scenario(sc, window, summary, y_limits, out_dir)
        window.to_csv(os.path.join(out_dir, f"surface_wind_{sc.date}.csv"))
        rows.append(summary)

    table = pd.DataFrame(rows).set_index("scenario")
    table.to_csv(os.path.join(out_dir, "summary_table.csv"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.drop(columns="figure"))
    print(f"\nCommon y-axis limits: {y_limits[0]:.1f} to {y_limits[1]:.1f} kt")
    print(f"Outputs written to: {out_dir}")
    return table


# ---------------------------------------------------------------------------
# Sanity checks with known answers
# ---------------------------------------------------------------------------
def run_self_tests() -> None:
    # 1. Wind aligned with runway -> pure headwind; from the right -> +crosswind;
    #    from behind -> negative headwind.
    hw, cw = runway_components(10.0, RUNWAY_TRUE_BEARING_DEG)
    assert abs(hw - 10) < 1e-9 and abs(cw) < 1e-9
    hw, cw = runway_components(10.0, RUNWAY_TRUE_BEARING_DEG + 90)
    assert abs(hw) < 1e-9 and abs(cw - 10) < 1e-9
    hw, _ = runway_components(10.0, RUNWAY_TRUE_BEARING_DEG + 180)
    assert abs(hw + 10) < 1e-9

    # 2. 359 deg and 1 deg: averaging the components gives the same result
    #    as a wind from 0 deg reduced by cos(1 deg) (no wrap-around problem).
    hw1, cw1 = runway_components(10.0, 359.0)
    hw2, cw2 = runway_components(10.0, 1.0)
    hw0, cw0 = runway_components(10.0 * np.cos(np.deg2rad(1.0)), 0.0)
    assert abs((hw1 + hw2) / 2 - hw0) < 1e-9 and abs((cw1 + cw2) / 2 - cw0) < 1e-9

    # 3. Unit conversion: 1 m/s = 1.943844 kt.
    assert abs(MS_TO_KT - 1.943844) < 1e-6

    # 4. Windows are UTC and are not shifted (also on the UTC+1 days).
    assert utc_window("2026-03-16", "12:00", "16:00")[0].hour == 12
    assert utc_window("2026-03-29", "08:00", "12:00")[0].hour == 8
    assert utc_window("2026-04-11", "16:00", "20:00")[0].hour == 16

    # 5. Pairing: interleaved synthetic sensors, one 3B sample without partner.
    t = pd.date_range("2026-01-01", periods=4, freq="3s", tz="UTC")
    a = add_components_kt(pd.DataFrame({"dir_deg": [359.0] * 4, "speed_ms": [10.0] * 4}, index=t))
    b_times = pd.DatetimeIndex(list(t[:3] + pd.Timedelta(seconds=1)) + [t[-1] + pd.Timedelta(seconds=30)])
    b = add_components_kt(pd.DataFrame({"dir_deg": [1.0] * 4, "speed_ms": [10.0] * 4}, index=b_times))
    out = average_sensors(pair_sensors(a, b))
    assert list(out["n_sensors"]) == [2, 2, 2, 1, 1], list(out["n_sensors"])
    hw_expected = hw0 * MS_TO_KT
    assert abs(out["headwind_kt"].iloc[0] - hw_expected) < 1e-9

    # 6. Absolute crosswind is taken after averaging: opposite crosswinds cancel.
    p = pd.DataFrame({"hw_a": [5.0], "cw_a": [2.0], "hw_b": [5.0], "cw_b": [-2.0]})
    assert average_sensors(p)["abs_crosswind_kt"].iloc[0] == 0.0

    print("All sanity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Run sanity checks only")
    args = parser.parse_args()

    run_self_tests()
    if not args.test:
        main()
