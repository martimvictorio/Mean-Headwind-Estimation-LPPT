"""
§5.3 Operational evaluation — headwind médio na final disponível quando cada passagem de
teste inicia a aproximação, para o baseline EUROCONTROL, o filtro de Kalman e o benchmark
METAR nos quatro cenários.

Avaliação operacional (T §3.5, Eq. 3.37): para cada passagem de teste j, a estimativa no
passo k_j (imediatamente antes da 1.ª observação aceite de j, logo sem dados de j por
construção) é média sobre os níveis B_j com observações de j e comparada com a média das
observações de j nesses níveis. Um erro por passagem. Mesmo código (avaliacao.py), mesmos
modelos e mesmo x0 (IPMA, último minuto completo antes do início) da §5.2.

Figura: headwind médio dos 36 níveis ao longo do período (corrida completa, sem
leave-one-out, amostrado a cada 10 s) para os três modelos, com o headwind médio observado
por cada passagem no instante da sua 1.ª observação. A linha média os 36 níveis; o erro da
tabela usa só os níveis B_j de cada passagem (Eq. 3.37). Uma figura por cenário (a escolher
para o texto; as restantes para anexo) e uma figura 2×2 com os quatro.

Figura de perfis: em cada cenário, a primeira passagem de teste que entra na final pelo menos
2 h depois do início (regra fixa, para não escolher o caso). Perfis dos três modelos no
passo k_j dessa passagem e as observações que ela fez a seguir.

Saídas em --out:
  tabela_5_3.csv                     bias, RMSE e MAE por modelo e cenário (Eq. 3.38), e com
                                     todos os cenários juntos (linha "All");
  perfil_operacional_cenarios.png    perfis no instante k_j de uma passagem por cenário;
  casos_perfil_5_3.csv               estimativa, referência e erro dessas passagens;
  erros_operacionais_5_3.csv         e_j, estimativa e referência de cada passagem de teste;
  headwind_medio_<cenário>.png       série temporal de cada cenário;
  headwind_medio_cenarios.png        os quatro cenários numa figura 2×2;
  config_usada.json                  configuração, x0 de cada cenário e versões.

Alteração de 29-10-2026: instantes, eixos e colunas em UTC ('tod' é UTC); removida a
conversão para hora local (k_j_local -> k_j_utc).

Uso:
  python resultados_5_3.py --data-dir <pasta com CSV, METAR e IPMA>
  python resultados_5_3.py --scenarios 11abril
  python resultados_5_3.py --selftest
Requer na mesma pasta: preprocessamento.py, cenarios.py, modelos.py, avaliacao.py,
estilo_figuras.py e resultados_5_2.py (x0 e escrita das tabelas partilhados).
"""

import argparse
import os

import numpy as np

import avaliacao as av
import cenarios
import modelos
import preprocessamento as pp
import resultados_5_2 as r52
from estilo_figuras import (ALTITUDE_LIM_FT, GRID_COLOR, INK, INK_MUTED, MODEL_LINEWIDTH,
                            MODEL_STYLE, style_axes)

SERIES_STEP_S = 10          # resolução temporal das séries da figura
TICK_STEP_S = 30 * 60       # marcas do eixo do tempo a cada 30 min
PROFILE_CASE_AFTER_S = 2 * 3600   # figura de perfis: 1.ª passagem de teste após 2 h


# =============================================================================
# AVALIAÇÃO DE UM CENÁRIO
# =============================================================================
def mean_headwind_series(model, obs_by_step, start_step, end_step, step_s=SERIES_STEP_S):
    """Headwind médio dos 36 níveis ao longo do período (corrida completa do cenário)."""
    t = list(range(start_step, end_step + 1, step_s))
    est = model.estimate(obs_by_step, start_step, t)
    return np.array(t), np.array([est[k].mean() for k in t])


def observed_mean_by_passage(obs, grid):
    """Headwind médio observado por cada passagem (média dos níveis B_j, Eqs. 3.35 e 3.37)
    e instante da sua 1.ª observação aceite."""
    out = {}
    for j, v in pp.group_by(obs, lambda o: o.aircraft).items():
        ref, _ = av.reference_by_level(v, grid)
        out[j] = (min(o.step for o in v), float(np.mean(list(ref.values()))))
    return out


def evaluate_scenario(data, grid):
    """Erros operacionais e séries temporais dos três modelos num cenário."""
    x0, x0_minute = r52.initial_state_kt(data)
    models = modelos.evaluation_models(grid, x0, data.metar, data.cfg["rwy_true_bearing_deg"])
    obs_by_step = pp.group_by(data.obs, lambda o: o.step)
    errors, series = {}, {}
    for m in models:
        errors[m] = av.operational_errors(m, data.obs, obs_by_step, data.start_step,
                                          data.test_ids, grid)
        series[m] = mean_headwind_series(m, obs_by_step, data.start_step, data.end_step)
    case = profile_case(data, models, obs_by_step, grid)
    return x0, x0_minute, errors, series, case


def select_profile_passage(first_step_by_passage, test_ids, start_step, after_s=PROFILE_CASE_AFTER_S):
    """Primeira passagem de teste cuja 1.ª observação aceite ocorre ≥ after_s após o início."""
    later = [j for j in test_ids if first_step_by_passage[j] >= start_step + after_s]
    if not later:
        raise ValueError("Nenhuma passagem de teste depois do instante pedido.")
    return min(later, key=lambda j: (first_step_by_passage[j], j))


def profile_case(data, models, obs_by_step, grid):
    """Perfis dos modelos em k_j e observações da passagem escolhida, com V̂_j e V_j^ref
    (Eq. 3.37) para ligar a figura à tabela."""
    j = select_profile_passage(data.first_step_by_passage, data.test_ids, data.start_step)
    k_j = data.first_step_by_passage[j] - 1
    obs_j = [o for o in data.obs if o.aircraft == j]
    ref, _ = av.reference_by_level(obs_j, grid)
    lv = sorted(ref)
    v_ref = float(np.mean([ref[i] for i in lv]))
    profiles = {m: m.estimate(obs_by_step, data.start_step, [k_j])[k_j] for m in models}
    return {"passage": j, "k_j": k_j, "obs": obs_j, "profiles": profiles,
            "v_ref": v_ref, "v_hat": {m: float(np.mean(x[lv])) for m, x in profiles.items()}}


def passage_rows(sc, errors_by_model, observed):
    rows = []
    for m, e in errors_by_model.items():
        for j, d in e.items():
            t_first, v_ref = observed[j]
            rows.append({"scenario": sc.key, "model": m.key, "passage": j,
                         "k_j_utc": pp.fmt_hhmm(t_first - 1),
                         "estimate_kt": v_ref + d["mean"], "reference_kt": v_ref,
                         "error_kt": d["mean"]})
    return rows


def case_rows(sc, case):
    return [{"scenario": sc.key, "passage": case["passage"],
             "k_j_utc": pp.fmt_hhmm(case["k_j"]),
             "model": m.key, "estimate_kt": v, "reference_kt": case["v_ref"],
             "error_kt": v - case["v_ref"]}
            for m, v in case["v_hat"].items()]


def print_table(rows):
    print("\n§5.3 — Operational evaluation (kt)")
    print(f"{'':12}{'model':33}{'bias':>7}{'RMSE':>7}{'MAE':>7}{'n':>5}")
    for r in rows:
        print(f"{r['name']:12}{r['model_label']:33}{r['bias_kt']:+7.2f}{r['rmse_kt']:7.2f}"
              f"{r['mae_kt']:7.2f}{r['n_errors']:5d}")


# =============================================================================
# FIGURAS: HEADWIND MÉDIO AO LONGO DO TEMPO
# =============================================================================
def draw_series(ax, data, series, observed, show_label=True):
    """Um cenário num eixo: warm-up sombreado, séries dos modelos e médias observadas."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    sc = data.scenario
    t0 = data.start_step
    ax.axvspan(t0, t0 + data.cfg["warmup_s"], color=GRID_COLOR, alpha=0.7, zorder=0,
               label=f"Warm-up period ({data.cfg['warmup_s'] // 60:g} min)")
    for m, (t, v) in series.items():
        ax.plot(t, v, linewidth=MODEL_LINEWIDTH, label=m.label, zorder=3, **MODEL_STYLE[m.key])
    t_obs = [t for t, _ in observed.values()]
    v_obs = [v for _, v in observed.values()]
    ax.scatter(t_obs, v_obs, s=22, color=INK, edgecolors="white", linewidths=0.6,
               zorder=4, label="Mean headwind observed by each passage")
    if show_label:
        ax.text(0.01, 0.98, f"{cenarios.scenario_name(sc)} ({sc.headwind_class.lower()} headwind, "
                f"{sc.crosswind_class.lower()} crosswind)", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP_S))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _pos: pp.fmt_hhmm(s)))
    ax.set_xlabel("Time (UTC)")
    style_axes(ax)


def y_limits(all_values, margin_kt=3.0):
    v = np.concatenate([np.asarray(x, float) for x in all_values])
    # espaço no topo para a identificação do cenário
    return v.min() - margin_kt, v.max() + 2.5 * margin_kt


def plot_scenario(data, series, observed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    draw_series(ax, data, series, observed)
    ax.set_ylim(*y_limits([v for _, v in series.values()] + [[v for _, v in observed.values()]]))
    ax.set_ylabel("Mean headwind of the 36 levels [kt]")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(results, path):
    """2×2 pela ordem da Tab. 4.1, eixo y partilhado (o eixo x é a hora de cada dia)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
    values = [v for data, series, observed in results for _, v in series.values()]
    values += [[v for _, v in observed.values()] for _, _, observed in results]
    for ax, (data, series, observed) in zip(axes.flat, results):
        draw_series(ax, data, series, observed)
    axes[0, 0].set_ylim(*y_limits(values))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean headwind of the 36 levels [kt]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(results, path):
    """2×2 pela ordem da Tab. 4.1, eixos partilhados (mesmos limites da Fig. 5.1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = [o.headwind_kt for data, case in results for o in data.obs]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, (data, case) in zip(axes.flat, results):
        sc = data.scenario
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
        ax.scatter([o.headwind_kt for o in case["obs"]], [o.h_ft for o in case["obs"]], s=18,
                   color=INK, alpha=0.55, edgecolors="none", zorder=2,
                   label="Observations of the test passage")
        for m, x in case["profiles"].items():
            ax.plot(x, data_grid_levels(), linewidth=MODEL_LINEWIDTH, label=m.label, zorder=3,
                    **MODEL_STYLE[m.key])
        start = pp.fmt_hhmm(case["k_j"] + 1)
        ax.text(0.03, 0.99, f"{cenarios.scenario_name(sc)} ({sc.headwind_class.lower()} headwind, "
                f"{sc.crosswind_class.lower()} crosswind)\nTest passage starting at {start} UTC",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_xlim(min(hw) - 2.0, max(hw) + 2.0)
        ax.set_ylim(ALTITUDE_LIM_FT[0], ALTITUDE_LIM_FT[1] + 300)   # 2 linhas de identificação
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("Headwind [kt]")
    for ax in axes[:, 0]:
        ax.set_ylabel("True altitude [ft]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def data_grid_levels():
    return cenarios.AltitudeGrid.from_config(cenarios.GRID).levels_ft


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    r52.run_selftests()
    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    rwy = pp.PREPROCESSING["rwy_true_bearing_deg"]

    # 1. Sem observações, a série do filtro e do baseline é x0; a do METAR é o METAR.
    metar = [pp.MetarReport(0, 1013.0, 15.0, rwy, 12.0)]
    models = modelos.evaluation_models(grid, 7.5, metar, rwy)
    for m, expected in zip(models, (7.5, 7.5, 12.0)):
        t, v = mean_headwind_series(m, {}, 100, 160)
        assert list(t) == [100, 110, 120, 130, 140, 150, 160] and np.allclose(v, expected)

    # 2. Média observada por passagem: média dos níveis com observações (cada nível pesa 1).
    obs = [pp.Observation(5, "A#1", "A", 5.0, 1960.0, 10.0),
           pp.Observation(6, "A#1", "A", 6.0, 1990.0, 14.0),     # mesmo nível: média 12
           pp.Observation(9, "A#1", "A", 9.0, 2501.0, 30.0)]     # outro nível: 30
    assert observed_mean_by_passage(obs, grid) == {"A#1": (5, 21.0)}

    # 3. Passagem da figura de perfis: a primeira de teste a entrar ≥ 2 h após o início
    #    (a fronteira conta); passagens fora do teste não são escolhidas.
    first = {"A#1": 100 + 7199, "B#1": 100 + 7300, "C#1": 100 + 7200, "W#1": 100 + 7250}
    assert select_profile_passage(first, ["A#1", "B#1", "C#1"], 100) == "C#1"
    print("Testes de sanidade da §5.3: todos passaram.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="resultados_5_3")
    ap.add_argument("--scenarios", nargs="+", choices=list(cenarios.SCENARIOS_BY_KEY),
                    default=[s.key for s in cenarios.SCENARIOS])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    run_selftests()
    if args.selftest:
        return

    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    os.makedirs(args.out, exist_ok=True)
    table, passages, x0_info, figures, cases, case_out, errors_by_sc = [], [], {}, [], [], [], {}
    for key in args.scenarios:
        sc = cenarios.SCENARIOS_BY_KEY[key]
        data = cenarios.load_scenario(sc, args.data_dir)
        x0, x0_minute, errors, series, case = evaluate_scenario(data, grid)
        x0_info[sc.key] = {"x0_kt": x0, "ipma_minute_utc": pp.fmt_hhmm(x0_minute)}
        print(f"{cenarios.scenario_name(sc)}: x0 = {x0:.2f} kt (IPMA, minuto {pp.fmt_hhmm(x0_minute)} UTC), "
              f"{len(data.test_ids)} passagens de teste")
        observed = observed_mean_by_passage(data.obs, grid)
        table += r52.table_rows(sc, errors, len(data.test_ids))
        errors_by_sc[sc] = errors
        cases.append((data, case))
        case_out += case_rows(sc, case)
        passages += passage_rows(sc, errors, observed)
        plot_scenario(data, series, observed, os.path.join(args.out, f"headwind_medio_{sc.key}.png"))
        figures.append((data, series, observed))
    if len(errors_by_sc) > 1:
        table += r52.pooled_rows(errors_by_sc)
    print_table(table)

    r52.write_csv(os.path.join(args.out, "tabela_5_3.csv"), table)
    r52.write_csv(os.path.join(args.out, "erros_operacionais_5_3.csv"), passages)
    r52.write_csv(os.path.join(args.out, "casos_perfil_5_3.csv"), case_out)
    if len(figures) == 4:
        plot_all(figures, os.path.join(args.out, "headwind_medio_cenarios.png"))
        plot_profiles(cases, os.path.join(args.out, "perfil_operacional_cenarios.png"))
    cenarios.save_config({**pp.PREPROCESSING, **cenarios.GRID, **cenarios.EVALUATION,
                          "kalman": modelos.KALMAN, "baseline": modelos.BASELINE,
                          "x0_ipma": x0_info, "series_step_s": SERIES_STEP_S,
                          "profile_case_after_s": PROFILE_CASE_AFTER_S,
                          "scenarios": [vars(cenarios.SCENARIOS_BY_KEY[k]) for k in args.scenarios]},
                         args.out)
    print(f"\nTabelas, figuras e configuração guardadas em: {args.out}")


if __name__ == "__main__":
    main()
