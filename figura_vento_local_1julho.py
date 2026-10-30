"""
Fig. 4.9 (§4.2, Preprocessing validation) — observações de vento local de 1 de julho de 2026.

Dois painéis, com todas as observações aceites pelo pré-processamento (T §3.2.1) no
período 12:00–16:00 UTC ('tod' é UTC; alteração de 29-10-2026):
  - esquerda: headwind (T Eq. 3.13) em função da distância ao limiar 02 (Eq. 3.3), com o
    eixo da distância invertido, no sentido da aproximação;
  - direita: altitude verdadeira (T Eq. 3.9) em função do headwind.
Mesmo estilo da figura da §5.1 (estilo_figuras.py). Substitui a figura gerada por
'visualização_localwind_1julho.py', que tinha um pré-processamento próprio.

Uso:
  python figura_vento_local_1julho.py --data-dir <pasta com CSV e METAR>
Requer 'preprocessamento.py' e 'estilo_figuras.py' na mesma pasta.
"""

import argparse
import os

import preprocessamento as pp
from estilo_figuras import ALTITUDE_LIM_FT, INK_MUTED, OBS_SCATTER, style_axes

DAY = {
    "date": "20260701",
    "aircraft_csv": "dados_1julho_12h16h.csv",
    "metar_file": "METAR_julho.txt",
    # Valores do texto da §4.2: a figura só é gerada se o pré-processamento os reproduzir.
    "expected_messages": 12788,
    "expected_obs": 3775,          # 3772 antes da correção UTC de 29-10-2026
    "expected_passages": 77,
}
DISTANCE_LIM_NM = (12.0, 0.0)     # invertido: a aproximação lê-se da esquerda para a direita


def load_day(data_dir):
    cfg = {**pp.PREPROCESSING, "scenario_date": DAY["date"]}
    metar = pp.load_metar_schedule(os.path.join(data_dir, DAY["metar_file"]), DAY["date"])
    obs, n_messages, _ = pp.load_observations(os.path.join(data_dir, DAY["aircraft_csv"]),
                                              metar, cfg)
    obs = pp.assign_passages(obs, cfg["passage_gap_s"])
    n_passages = len({o.aircraft for o in obs})
    print(f"Mensagens {n_messages} | observações aceites {len(obs)} | passagens {n_passages}")
    found = (n_messages, len(obs), n_passages)
    expected = (DAY["expected_messages"], DAY["expected_obs"], DAY["expected_passages"])
    if found != expected:
        raise ValueError(f"Contagens {found} diferentes das do texto da §4.2 {expected}.")
    return obs


def plot(obs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = [o.headwind_kt for o in obs]
    hw_lim = (min(hw) - 2.0, max(hw) + 2.0)       # mesmo intervalo de headwind nos dois painéis

    fig, (ax_d, ax_h) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax_d.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax_d.scatter([o.distance_nm for o in obs], hw, zorder=2, **OBS_SCATTER)
    ax_d.set_xlim(*DISTANCE_LIM_NM)
    ax_d.set_ylim(*hw_lim)
    ax_d.set_xlabel("Distance to Runway 02 threshold [NM]")
    ax_d.set_ylabel("Headwind [kt]")

    ax_h.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax_h.scatter(hw, [o.h_ft for o in obs], zorder=2, **OBS_SCATTER)
    ax_h.set_xlim(*hw_lim)
    ax_h.set_ylim(*ALTITUDE_LIM_FT)
    ax_h.set_xlabel("Headwind [kt]")
    ax_h.set_ylabel("True altitude [ft]")

    for ax in (ax_d, ax_h):
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="resultados_4_2")
    args = ap.parse_args()

    pp.run_selftests()
    obs = load_day(args.data_dir)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "vento_local_1julho.png")
    plot(obs, path)
    print(f"Figura guardada em: {path}")


if __name__ == "__main__":
    main()
