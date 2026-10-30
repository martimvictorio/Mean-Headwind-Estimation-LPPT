# Wind on Final Approach — Thesis code

Code developed for the master's thesis **"Wind on Final Approach"**. It estimates the headwind on the final approach to runway 02 at Lisbon (LPPT) and compares a Kalman filter with the EUROCONTROL baseline and a METAR benchmark.

## Where to find the thesis results

**The thesis results are produced by the `.py` files whose names start with `resultados`.** The number in each name refers to the thesis section:

| Script | Section | What it produces | Output folder |
|---|---|---|---|
| `resultados_4_2_1_METAR.py` | §4.2.1 | Surface wind at LPPT from METAR reports (Mar–Jul 2026), Figs. 4.3 and 4.4 | `METAR_analise_vento/` |
| `resultados_4_2_1_vento_cenarios.py` | §4.2.1 | Surface wind (anemometers 3A/3B) for the four scenarios | `surface_wind_output/` |
| `resultados_4_2_e_4_3_flightdata.py` | §4.1.2 and §4.2 | Raw and pre-processed CAT062 data for 1 July | `figuras_dados_1julho/` |
| `resultados_4_3_headwind.py` | §4.2 | Local wind observations for 1 July | `resultados_4_2/` |
| `resultados_4_4_1_q_tuning.py` | §4.4.1 | Tuning of the Kalman filter parameter q | `resultados_q_1julho/` |
| `resultados_4_4_1_headwind_mean.py` | §4.4.1 | Kalman filter for two values of q (profile and mean headwind) | `resultados_visualizacao_1julho/` |
| `resultados_4_4_2_baseline.py` | §4.4.2 | Verification of the EUROCONTROL baseline | `resultados_baseline_1julho/` |
| `resultados_5_1.py` | §5.1 | Scenario overview (data volume, wind) | `resultados_5_1/` |
| `resultados_5_2.py` | §5.2 | Nowcast evaluation of the three models | `resultados_5_2/` |
| `resultados_5_3.py` | §5.3 | Operational evaluation of the three models | `resultados_5_3/` |

## Other files

**Shared modules**, used by the results scripts:
- `preprocessamento.py`: pre-processing of CAT062 messages and calculation of the observed headwind
- `modelos.py`: Kalman filter, EUROCONTROL baseline and METAR benchmark
- `avaliacao.py`: nowcast and operational evaluation and metrics (bias, RMSE, MAE)
- `cenarios.py`: definition of the evaluation scenarios and data loading
- `estilo_figuras.py`: common figure style

**Exploratory code**, not part of the results:
- `METAR_analise_<day>.py`: METAR analysis for each scenario day

**Data:**
- `dados_<day>_<period>.csv`: CAT062 radar messages for each scenario (16 and 29 March, 11 April, 6 May, 1 July 2026)
- `METAR_<month>.txt`: LPPT METAR reports, March to July 2026
- `WIND_3A_MSG_RAW_<DD>.csv` and `WIND_3B_MSG_RAW_<DD>.csv`: IPMA anemometers (sensors 3A and 3B) on the scenario days

All times are in UTC.

## Important: keep all files in the same folder

The scripts depend on each other. The results scripts import the shared modules, and some import other results scripts (for example, `resultados_4_4_2_baseline.py` uses `resultados_4_4_1_q_tuning.py`). The data files are also read from the scripts' folder.

**All `.py` files and all data files must be in the same folder.** If any file is moved to another folder, the code will stop working.

## How to run

Requirements: Python 3.11, `numpy`, `pandas`, `matplotlib` and `seaborn`.

```
pip install numpy pandas matplotlib seaborn
```

From the repository folder, run the script for the result you want. For example:

```
python resultados_5_2.py
```

Figures and tables are saved in the output folder listed in the table above.