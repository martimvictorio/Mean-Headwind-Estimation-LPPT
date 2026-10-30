"""
§5.1 Scenario overview — tabela com o volume de dados de cada cenário de avaliação.

Para cada cenário (cenarios.SCENARIOS, T Tab. 4.1) conta:
  - messages:      mensagens CAT062 lidas (todas as linhas do ficheiro);
  - accepted_obs:  observações aceites pelos 5 critérios de T §3.2.1;
  - aircraft:      aeronaves distintas com pelo menos uma observação aceite, identificadas
                   pelo endereço Mode S (que não muda com o número de pista do radar);
  - go_arounds:    passages − aircraft, i.e. passagens extra da mesma aeronave. Pressupõe
                   que, num período de 4 h, uma aeronave só repete a final após um
                   go-around; avisa se o intervalo entre passagens passar 60 min;
  - passages:      passagens na aproximação final (pp.assign_passages, gap de 300 s);
  - test_passages: passagens de teste, i.e. que entram depois do warm-up de 15 min (T §3.5).
                   São passagens, não aeronaves: a segunda passagem de um go-around é um
                   caso de teste próprio.
Colunas de diagnóstico (só no CSV, não para a tese):
  - warmup_excluded:  passages − test_passages;
  - max_metar_age_min: maior idade do METAR usado na correção de altitude (T Eq. 3.9).

Tabela vento vs duração (tabela_vento_duracao_5_1.csv), uma linha por cenário, só médias:
  - surface_headwind_ipma_kt: headwind médio à superfície (IPMA, sensores 3A/3B) no período
                   do cenário (pp.mean_surface_headwind_kt), como na Tab. 4.1;
  - mean_headwind_36_levels_kt: média dos 36 níveis da média por nível de todas as
                   observações aceites (a linha da figura de vento local);
  - difference_kt: as duas anteriores subtraídas (36 níveis − superfície);
  - mean_approach_min: média, sobre as passagens de teste, do tempo entre a primeira e a
                   última observação aceite da passagem (t_raw). ATENÇÃO: inclui passagens que
                   só cobrem parte da janela de altitudes, que puxam a média para baixo;
                   a mediana (só na consola) é menos sensível a isso.

Figura: headwind observado pelas aeronaves (T Eq. 3.13) em função da altitude verdadeira,
um painel por cenário (grelha 2×2 da Tab. 4.1), com todas as observações aceites do período
e a média por nível da grelha (nível pela Eq. 3.32; média como nas Eqs. 3.34–3.35). A média
agrega as 4 h do cenário: não é o perfil num instante.

Dispersão: desvio-padrão amostral (n − 1) das observações de cada nível, resumido por
cenário pela mediana (e P10–P90) dos 36 níveis. Inclui o ruído de medição E a variação do
vento ao longo das 4 h; não é só ruído.

Saídas em --out: tabela_5_1.csv, tabela_vento_duracao_5_1.csv, vento_local_por_nivel.csv (média, desvio-padrão e n por
nível), vento_local_dispersao.csv (resumo por cenário), vento_local_por_cenario.png e
config_usada.json (configuração e versões).

Uso:
  python resultados_5_1.py --data-dir <pasta com CSV, METAR e IPMA>
  python resultados_5_1.py --scenarios 11abril 29marco
  python resultados_5_1.py --selftest
Requer 'preprocessamento.py', 'cenarios.py' e 'estilo_figuras.py' na mesma pasta.
"""

import argparse
import csv
import os

import numpy as np

import cenarios
import preprocessamento as pp
from estilo_figuras import ALTITUDE_LIM_FT, INK, INK_MUTED, OBS_SCATTER, style_axes

HEADER = ["scenario", "name", "date", "period_utc", "headwind_class", "crosswind_class",
          "messages", "accepted_obs", "aircraft", "go_arounds", "passages", "test_passages",
          "warmup_excluded", "max_metar_age_min"]
GO_AROUND_MAX_INTERVAL_S = 60 * 60   # acima disto, a repetição não parece um go-around


def aircraft_id(passage_key):
    """Aeronave de uma passagem ('modeS|trkNr#n'): o endereço Mode S; sem Mode S, o
    trajeto (modeS|trkNr) como substituto."""
    track_key = passage_key.split("#")[0]
    mode_s = track_key.split("|")[0]
    return track_key if mode_s == "NA" else mode_s


def count_aircraft(first_step_by_passage):
    """Nº de aeronaves distintas e intervalos (s) entre passagens repetidas da mesma."""
    starts = {}
    for j, k in first_step_by_passage.items():
        starts.setdefault(aircraft_id(j), []).append(k)
    intervals = [b - a for ks in starts.values() for a, b in zip(sorted(ks), sorted(ks)[1:])]
    return len(starts), intervals


def approach_durations_min(obs, passage_ids):
    """Tempo (min) entre a primeira e a última observação aceite de cada passagem."""
    by_passage = pp.group_by(obs, lambda o: o.aircraft)
    return np.array([(max(o.t_raw for o in by_passage[j]) - min(o.t_raw for o in by_passage[j])) / 60
                     for j in passage_ids])


def overview_row(data):
    """Linha da tabela para um cenário já carregado."""
    s = data.scenario
    n_passages = len(data.first_step_by_passage)
    n_aircraft, intervals = count_aircraft(data.first_step_by_passage)
    if any(dt > GO_AROUND_MAX_INTERVAL_S for dt in intervals):
        print(f"[AVISO] [{s.key}] aeronave com passagens repetidas a mais de 60 min: "
              "pode não ser um go-around.")
    return {
        "scenario": s.key,
        "name": cenarios.scenario_name(s),
        "date": s.label,
        "period_utc": f"{s.period_utc[0]}–{s.period_utc[1]}",
        "headwind_class": s.headwind_class,
        "crosswind_class": s.crosswind_class,
        "messages": data.n_messages,
        "accepted_obs": len(data.obs),
        "aircraft": n_aircraft,
        "go_arounds": n_passages - n_aircraft,
        "passages": n_passages,
        "test_passages": len(data.test_ids),
        "warmup_excluded": n_passages - len(data.test_ids),
        "max_metar_age_min": round(data.max_metar_age_s / 60),
    }


def wind_duration_rows(data_by_scenario, grid):
    """Headwind à superfície vs na final e duração média da aproximação, por cenário."""
    rows = []
    for sc, data in data_by_scenario.items():
        surface = pp.mean_surface_headwind_kt(data.surface_wind, data.start_step, data.end_step)
        final = float(np.nanmean(mean_by_level(data.obs, grid)[0]))
        dur = approach_durations_min(data.obs, data.test_ids)
        rows.append({"scenario": sc.key, "name": cenarios.scenario_name(sc),
                     "surface_headwind_ipma_kt": round(surface, 1),
                     "mean_headwind_36_levels_kt": round(final, 1),
                     "difference_kt": round(final - surface, 1),
                     "mean_approach_min": round(float(dur.mean()), 2),
                     "median_approach_min_check": round(float(np.median(dur)), 2),
                     "test_passages": len(dur)})
    return rows


def print_wind_duration(rows):
    print("\nHeadwind à superfície vs na final e duração da aproximação (médias):")
    for r in rows:
        print(f"  {r['name']}: superfície {r['surface_headwind_ipma_kt']:5.1f} kt | 36 níveis "
              f"{r['mean_headwind_36_levels_kt']:5.1f} kt | dif. {r['difference_kt']:+5.1f} kt | duração média "
              f"{r['mean_approach_min']:.2f} min (mediana {r['median_approach_min_check']:.2f})")


def check_row(row):
    """Coerência interna das contagens."""
    assert row["test_passages"] <= row["passages"] <= row["accepted_obs"] <= row["messages"], row
    assert row["aircraft"] <= row["passages"], row
    assert row["go_arounds"] == row["passages"] - row["aircraft"], row
    if row["max_metar_age_min"] > 60:
        print(f"[AVISO] [{row['scenario']}] METAR com mais de 60 min usado na correção de altitude.")


def print_table(rows):
    cols = ["name", "date", "period_utc", "messages", "accepted_obs", "aircraft", "go_arounds",
            "passages", "test_passages"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("\n§5.1 — Scenario overview (UTC)")
    for r in [dict(zip(cols, cols))] + rows:
        print("  ".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))


def write_csv(path, rows, fieldnames=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# FIGURA: VENTO LOCAL POR CENÁRIO
# =============================================================================
def stats_by_level(obs, grid):
    """Por nível (Eq. 3.32): média, desvio-padrão amostral (n − 1) e nº de observações.
    NaN onde não há observações (média) ou há menos de 2 (desvio-padrão)."""
    by_level = pp.group_by(obs, lambda o: grid.nearest_level(o.h_ft))
    mean, sd = np.full(grid.n, np.nan), np.full(grid.n, np.nan)
    counts = np.zeros(grid.n, dtype=int)
    for i, v in by_level.items():
        y = np.array([o.headwind_kt for o in v])
        counts[i] = len(y)
        mean[i] = y.mean()
        if len(y) >= 2:
            sd[i] = y.std(ddof=1)
    return mean, sd, counts


def mean_by_level(obs, grid):
    """Média e nº de observações por nível (nível pela Eq. 3.32). NaN sem observações."""
    mean, _, counts = stats_by_level(obs, grid)
    return mean, counts


def local_wind_rows(data_by_scenario, grid):
    rows = []
    for sc, data in data_by_scenario.items():
        mean, sd, n = stats_by_level(data.obs, grid)
        rows += [{"scenario": sc.key, "name": cenarios.scenario_name(sc), "level_ft": int(h),
                  "n_obs": int(c), "mean_headwind_kt": m, "sd_headwind_kt": s}
                 for h, c, m, s in zip(grid.levels_ft, n, mean, sd)]
    return rows


def dispersion_rows(data_by_scenario, grid):
    """Resumo por cenário do desvio-padrão por nível: mediana, P10 e P90 dos 36 níveis."""
    rows = []
    for sc, data in data_by_scenario.items():
        _, sd, n = stats_by_level(data.obs, grid)
        sd = sd[~np.isnan(sd)]
        rows.append({"scenario": sc.key, "name": cenarios.scenario_name(sc),
                     "levels": len(sd), "min_obs_per_level": int(n.min()),
                     "sd_median_kt": float(np.median(sd)),
                     "sd_p10_kt": float(np.percentile(sd, 10)),
                     "sd_p90_kt": float(np.percentile(sd, 90))})
    return rows


def print_dispersion(rows):
    print("\nDesvio-padrão das observações por nível (mediana [P10–P90] dos níveis):")
    for r in rows:
        print(f"  {r['name']}: {r['sd_median_kt']:.1f} kt [{r['sd_p10_kt']:.1f}–{r['sd_p90_kt']:.1f}] "
              f"| {r['levels']} níveis, ≥ {r['min_obs_per_level']} obs. por nível")


def plot_local_wind(data_by_scenario, grid, path):
    """2×2 pela ordem da Tab. 4.1 (linhas: HW alto/baixo; colunas: CW alto/baixo), eixos
    partilhados. Cenário identificado dentro de cada painel, sem títulos por painel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, (sc, data) in zip(axes.flat, data_by_scenario.items()):
        mean, _ = mean_by_level(data.obs, grid)
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
        ax.scatter([o.headwind_kt for o in data.obs], [o.h_ft for o in data.obs], zorder=2,
                   label="Accepted observations", **OBS_SCATTER)
        ax.plot(mean, grid.levels_ft, color=INK, linewidth=2, zorder=3,
                label="Mean per altitude level")
        # Identificação na faixa acima de 4000 ft, sempre vazia (critério 4, Eq. 3.7).
        ax.text(0.03, 0.99, f"{cenarios.scenario_name(sc)} ({sc.headwind_class.lower()} headwind, "
                f"{sc.crosswind_class.lower()} crosswind)", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_ylim(*ALTITUDE_LIM_FT)   # 4000–4400 ft livre para a identificação
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("Headwind [kt]", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("True altitude [ft]", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=INK,
                     markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    pp.run_selftests()
    cenarios.run_selftests()

    # 1. Contagens incoerentes são detetadas.
    row = {"scenario": "x", "messages": 10, "accepted_obs": 8, "aircraft": 2, "go_arounds": 1,
           "passages": 3, "test_passages": 2, "max_metar_age_min": 29}
    check_row(row)
    for bad in ({**row, "test_passages": 4}, {**row, "aircraft": 4}, {**row, "go_arounds": 0}):
        try:
            check_row(bad)
        except AssertionError:
            continue
        raise AssertionError("check_row não detetou contagens incoerentes")

    # 2. Aeronaves: o mesmo Mode S com outro número de pista conta como uma aeronave;
    #    sem Mode S, o trajeto substitui-o. Intervalo entre passagens repetidas.
    first = {"abc|10#1": 0, "abc|10#2": 1000, "abc|22#1": 1500, "NA|7#1": 50, "NA|8#1": 60}
    n, intervals = count_aircraft(first)
    assert n == 3 and sorted(intervals) == [500, 1000]

    # 3. Média por nível: 1960 e 1990 ft caem em 1950 ft (Eq. 3.32); 2001 ft em 2050 ft.
    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    obs = [pp.Observation(0, "A#1", "A", 0.0, h, y) for h, y in ((1960, 10), (1990, 14), (2001, 3))]
    mean, n = mean_by_level(obs, grid)
    i = int(np.where(grid.levels_ft == 1950)[0][0])
    assert n[i] == 2 and mean[i] == 12.0 and n[i + 1] == 1 and mean[i + 1] == 3.0
    assert n.sum() == 3 and np.isnan(mean[0])
    # Desvio-padrão amostral: [10, 14] -> sqrt(8) = 2.83; um só valor -> NaN.
    _, sd, _ = stats_by_level(obs, grid)
    assert abs(sd[i] - np.sqrt(8.0)) < 1e-12 and np.isnan(sd[i + 1])
    # 4. Duração da aproximação: da 1.ª à última observação aceite de cada passagem.
    obs = [pp.Observation(k, j, j, float(k), 2000.0, 0.0)
           for j, ks in (("A#1", (100, 160, 340)), ("B#1", (500, 530))) for k in ks]
    assert np.allclose(approach_durations_min(obs, ["A#1", "B#1"]), [4.0, 0.5])
    print("Testes de sanidade da §5.1: todos passaram.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="resultados_5_1")
    ap.add_argument("--scenarios", nargs="+", choices=list(cenarios.SCENARIOS_BY_KEY),
                    default=[s.key for s in cenarios.SCENARIOS])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    run_selftests()
    if args.selftest:
        return

    rows, data_by_scenario = [], {}
    for key in args.scenarios:
        sc = cenarios.SCENARIOS_BY_KEY[key]
        data_by_scenario[sc] = cenarios.load_scenario(sc, args.data_dir)
        rows.append(overview_row(data_by_scenario[sc]))
        check_row(rows[-1])
    print_table(rows)

    grid = cenarios.AltitudeGrid.from_config(cenarios.GRID)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "tabela_5_1.csv"), rows, HEADER)
    wind_duration = wind_duration_rows(data_by_scenario, grid)
    print_wind_duration(wind_duration)
    write_csv(os.path.join(args.out, "tabela_vento_duracao_5_1.csv"), wind_duration)
    write_csv(os.path.join(args.out, "vento_local_por_nivel.csv"),
              local_wind_rows(data_by_scenario, grid))
    dispersion = dispersion_rows(data_by_scenario, grid)
    print_dispersion(dispersion)
    write_csv(os.path.join(args.out, "vento_local_dispersao.csv"), dispersion)
    plot_local_wind(data_by_scenario, grid, os.path.join(args.out, "vento_local_por_cenario.png"))
    cenarios.save_config({**pp.PREPROCESSING, **cenarios.GRID, **cenarios.EVALUATION,
                          "scenarios": [vars(cenarios.SCENARIOS_BY_KEY[k]) for k in args.scenarios]},
                         args.out)
    print(f"\nTabela, figura e configuração guardadas em: {args.out}")


if __name__ == "__main__":
    main()
