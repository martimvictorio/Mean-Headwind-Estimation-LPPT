"""
§5.2 Nowcast evaluation — baseline EUROCONTROL, filtro de Kalman e benchmark METAR nos
quatro cenários de avaliação.

Avaliação nowcast (T §3.5, Eqs. 3.35–3.38): para cada passagem de teste j, cada modelo corre
sem as observações de j (leave-one-passage-out) e as suas estimativas nos passos em que j
transmitiu são comparadas, nível a nível, com a média das observações de j nesse nível.
O mesmo código (avaliacao.py) avalia os três modelos.

Modelos (modelos.py):
  - baseline EUROCONTROL tal como na §3.4 (janela de 10 min, rejeição a 10 m/s, N ≥ 3,
    retenção ilimitada do último valor válido);
  - filtro de Kalman com q = 0.005 kt²/s, afinado em 1 de julho e NÃO reafinado aqui;
  - benchmark METAR: headwind de superfície (Eq. 3.1) do último METAR, em todos os níveis.
x0 do filtro e do baseline: headwind de superfície IPMA (média dos sensores 3A/3B) no último
minuto completo antes do instante inicial (regra de bloco de 1 min, T §3.1.2).

Saídas em --out:
  tabela_5_2.csv               bias, RMSE e MAE por modelo e cenário (Eq. 3.38), e com
                               todos os cenários juntos (linha "All");
  metricas_por_nivel_5_2.csv   o mesmo por nível de altitude;
  erros_nowcast_5_2.csv        erro e_{j,i} de cada passagem e nível (para a §5.4);
  rmse_nowcast_por_nivel.png   RMSE por nível, um painel por cenário;
  config_usada.json            configuração, x0 de cada cenário e versões.

Uso:
  python resultados_5_2.py --data-dir <pasta com CSV, METAR e IPMA>
  python resultados_5_2.py --scenarios 11abril
  python resultados_5_2.py --selftest
Requer na mesma pasta: preprocessamento.py, cenarios.py, modelos.py, avaliacao.py e
estilo_figuras.py.
"""

import argparse
import csv
import os

import numpy as np

import avaliacao as av
import cenarios
import modelos
import preprocessamento as pp
from estilo_figuras import ALTITUDE_LIM_FT, INK, MODEL_LINEWIDTH, MODEL_STYLE, style_axes


# =============================================================================
# AVALIAÇÃO DE UM CENÁRIO
# =============================================================================
def initial_state_kt(data):
    """x0 = headwind de superfície IPMA no último minuto completo antes do início."""
    return pp.surface_headwind_block_kt(data.surface_wind, data.start_step)


def evaluate_scenario(data, grid):
    """Erros nowcast dos três modelos num cenário. Devolve (x0, minuto do x0, {modelo: erros})."""
    x0, x0_minute = initial_state_kt(data)
    models = modelos.evaluation_models(grid, x0, data.metar, data.cfg["rwy_true_bearing_deg"])
    obs_by_step = pp.group_by(data.obs, lambda o: o.step)
    errors = {m: av.nowcast_errors(m, data.obs, obs_by_step, data.start_step, data.test_ids, grid)
              for m in models}
    return x0, x0_minute, errors


# =============================================================================
# TABELAS
# =============================================================================
def table_rows(sc, errors_by_model, n_test):
    return [{"scenario": sc.key, "name": cenarios.scenario_name(sc), "model": m.key,
             "model_label": m.label, **{f"{k}_kt" if k != "n" else "n_errors": v
                                       for k, v in av.metrics(e).items()},
             "test_passages": n_test}
            for m, e in errors_by_model.items()]


def pooled_rows(errors_by_scenario):
    """Linha "All scenarios" por modelo: métricas sobre os erros de todos os cenários juntos
    (Eq. 3.38). Resume os cenários escolhidos (grelha 2×2), não a climatologia de LPPT.
    errors_by_scenario: {cenário: {modelo: erros}}."""
    pooled, labels = {}, {}
    for sc, errors_by_model in errors_by_scenario.items():
        for m, e in errors_by_model.items():
            labels[m.key] = m.label
            pooled.setdefault(m.key, {}).update({(sc.key, j): d for j, d in e.items()})
    return [{"scenario": "all", "name": "All", "model": key, "model_label": labels[key],
             **{f"{k}_kt" if k != "n" else "n_errors": v for k, v in av.metrics(e).items()},
             "test_passages": len(e)}
            for key, e in pooled.items()]


def level_rows(sc, errors_by_model, grid):
    return [{"scenario": sc.key, "name": cenarios.scenario_name(sc), "model": m.key, **r}
            for m, e in errors_by_model.items() for r in av.per_level_metrics(e, grid)]


def error_rows(sc, errors_by_model, grid):
    return [{"scenario": sc.key, "model": m.key, "passage": j,
             "level_ft": int(grid.levels_ft[i]), "error_kt": err}
            for m, e in errors_by_model.items() for j, d in e.items() for i, err in d.items()]


def print_table(rows):
    print("\n§5.2 — Nowcast evaluation (kt)")
    print(f"{'':12}{'model':33}{'bias':>7}{'RMSE':>7}{'MAE':>7}{'n':>7}")
    for r in rows:
        print(f"{r['name']:12}{r['model_label']:33}{r['bias_kt']:+7.2f}{r['rmse_kt']:7.2f}"
              f"{r['mae_kt']:7.2f}{r['n_errors']:7d}")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# FIGURA: RMSE NOWCAST POR NÍVEL
# =============================================================================
def plot_rmse_by_level(level_rows_by_scenario, labels, path):
    """2×2 pela ordem da Tab. 4.1, eixos partilhados; cenário identificado no topo de cada
    painel (faixa acima de 3950 ft, sem níveis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_max = max(r["rmse"] for rows in level_rows_by_scenario.values() for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, (sc, rows) in zip(axes.flat, level_rows_by_scenario.items()):
        for key, label in labels.items():
            r = [x for x in rows if x["model"] == key]
            ax.plot([x["rmse"] for x in r], [x["level_ft"] for x in r], linewidth=MODEL_LINEWIDTH,
                    marker="o", markersize=3, label=label, zorder=3, **MODEL_STYLE[key])
        ax.text(0.03, 0.99, f"{cenarios.scenario_name(sc)} ({sc.headwind_class.lower()} headwind, "
                f"{sc.crosswind_class.lower()} crosswind)", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_xlim(0, 1.05 * x_max)
        ax.set_ylim(*ALTITUDE_LIM_FT)
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("Nowcast RMSE [kt]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude level [ft]")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    pp.run_selftests()
    cenarios.run_selftests()
    av.run_selftests()

    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    rwy = pp.PREPROCESSING["rwy_true_bearing_deg"]

    # 1. Benchmark METAR: igual em todos os níveis, Eq. 3.1, muda no instante de emissão
    #    (METAR emitido em k já é usado em k) e ignora as observações e o leave-one-out.
    metar = [pp.MetarReport(0, 1013.0, 15.0, rwy, 10.0),            # alinhado: HW = 10 kt
             pp.MetarReport(1800, 1013.0, 15.0, rwy + 60.0, 30.0)]   # a 60°: HW = 15 kt
    model = modelos.metar_model(grid, metar, rwy)
    est = model.estimate({}, 0, [0, 1799, 1800], exclude="X")
    assert np.allclose(est[0], 10.0) and np.allclose(est[1799], 10.0)
    assert np.allclose(est[1800], 15.0) and est[1800].shape == (36,)

    # 2. Sem observações, o filtro e o baseline mantêm x0 em todos os níveis.
    models = modelos.evaluation_models(grid, 7.5, metar, rwy)
    assert [m.key for m in models] == ["baseline", "kalman", "metar"]
    for m in models[:2]:
        assert np.allclose(m.estimate({}, 0, [100])[100], 7.5)

    # 4. Linha "All scenarios": métricas sobre a concatenação dos erros dos cenários; a
    #    mesma passagem em cenários diferentes não se confunde.
    sc1, sc2 = cenarios.SCENARIOS[:2]
    m = models[0]
    rows = pooled_rows({sc1: {m: {"A#1": {0: 1.0, 1: -1.0}}}, sc2: {m: {"A#1": {0: 3.0}}}})
    assert len(rows) == 1 and rows[0]["n_errors"] == 3 and rows[0]["test_passages"] == 2
    assert abs(rows[0]["bias_kt"] - 1.0) < 1e-12 and abs(rows[0]["rmse_kt"] - np.sqrt(11 / 3)) < 1e-12

    # 3. Filtro: q usado é o de modelos.KALMAN (não reafinado).
    assert f"q = {modelos.KALMAN['q_kt2_s']:g}" in models[1].label
    print("Testes de sanidade da §5.2: todos passaram.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="resultados_5_2")
    ap.add_argument("--scenarios", nargs="+", choices=list(cenarios.SCENARIOS_BY_KEY),
                    default=[s.key for s in cenarios.SCENARIOS])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    run_selftests()
    if args.selftest:
        return

    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    table, levels, errors_out, levels_by_sc, x0_info, labels = [], [], [], {}, {}, {}
    errors_by_sc = {}
    for key in args.scenarios:
        sc = cenarios.SCENARIOS_BY_KEY[key]
        data = cenarios.load_scenario(sc, args.data_dir)
        x0, x0_minute, errors = evaluate_scenario(data, grid)
        x0_info[sc.key] = {"x0_kt": x0, "ipma_minute_utc": pp.fmt_hhmm(x0_minute)}
        print(f"{cenarios.scenario_name(sc)}: x0 = {x0:.2f} kt (IPMA, minuto {pp.fmt_hhmm(x0_minute)} UTC), "
              f"{len(data.test_ids)} passagens de teste")
        labels.update({m.key: m.label for m in errors})
        table += table_rows(sc, errors, len(data.test_ids))
        errors_by_sc[sc] = errors
        levels_by_sc[sc] = level_rows(sc, errors, grid)
        levels += levels_by_sc[sc]
        errors_out += error_rows(sc, errors, grid)
    if len(errors_by_sc) > 1:
        table += pooled_rows(errors_by_sc)
    print_table(table)

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "tabela_5_2.csv"), table)
    write_csv(os.path.join(args.out, "metricas_por_nivel_5_2.csv"), levels)
    write_csv(os.path.join(args.out, "erros_nowcast_5_2.csv"), errors_out)
    plot_rmse_by_level(levels_by_sc, labels, os.path.join(args.out, "rmse_nowcast_por_nivel.png"))
    cenarios.save_config({**pp.PREPROCESSING, **cenarios.GRID, **cenarios.EVALUATION,
                          "kalman": modelos.KALMAN, "baseline": modelos.BASELINE,
                          "x0_ipma": x0_info,
                          "scenarios": [vars(cenarios.SCENARIOS_BY_KEY[k]) for k in args.scenarios]},
                         args.out)
    print(f"\nTabelas, figura e configuração guardadas em: {args.out}")


if __name__ == "__main__":
    main()
