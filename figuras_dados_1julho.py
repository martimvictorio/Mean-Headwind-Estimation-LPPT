"""
Figs. 4.7 e 4.9 (§4.1.2 e §4.2) — dados CAT062 em bruto e pré-processados, 1 de julho de 2026,
12:00–16:00 UTC.

Substitui 'visualização_dados_raw_1julho.py' (Fig. 4.7) e 'visualização_preprocessing_1julho.py'
(Fig. 4.9), mantendo o estilo das figuras (cores, marcadores, linhas de referência, legendas,
limites dos eixos). Diferenças em relação a esses scripts:
  1. A classificação aceite/rejeitado da Fig. 4.9 usa o pré-processamento partilhado
     (preprocessamento.observation_from_row, T §3.2.1), o mesmo das figuras e resultados
     seguintes. O script antigo tinha critérios próprios (limite inferior de 330 ft em vez de
     400 ft, sem o critério do Mode S nem da IAS), pelo que as contagens não batiam com o texto.
  2. METAR válido em cada mensagem por preprocessamento.metar_valid_at ('tod' em UTC).
  3. Distância com sinal medida ao longo do rumo de aproximação da pista (22.72° + 180°),
     o mesmo dos critérios, em vez de 205° fixos.
  4. Rótulos dos eixos da vista de topo corrigidos: o eixo horizontal é x (Este) e o vertical
     é y (Norte). Os scripts antigos tinham os rótulos trocados ("Y Coordinate" no eixo
     horizontal). Os dados desenhados são os mesmos.
  5. Nas vistas verticais (painéis da direita) é retirado o tráfego local de baixa altitude
     que, projetado no eixo da pista 02, aparece a 3–13 NM e abaixo de 2000 ft sem ter
     relação com a aproximação: (a) tráfego do aeródromo de Cascais (LPCS); (b) quatro
     trajetos HIB101–HIB104 (sem Mode S, 25–450 ft, 40–90 kt) sobre o Tejo junto a
     Belém/Algés, compatíveis com helicópteros. As vistas de topo mantêm todo o tráfego
     (o texto da §4.1.2 refere a atividade em Cascais).
  6. Sem títulos nos painéis (como nas figuras da tese) e sem a janela 3D do script antigo.

Regra de exclusão (DECISÃO EM ABERTO, não definida na tese), zonas em EXCLUSION_ZONES:
  uma aeronave (endereço Mode S; sem Mode S, o track number) é excluída se alguma das suas
  mensagens estiver a menos de radius_nm do centro de uma zona e abaixo de max_alt_ft de
  altitude-pressão. Todas as mensagens dessa aeronave saem das vistas verticais (inclui as
  pernas de chegada/saída, que também caem na zona assinalada pelo orientador).
  - Cascais: centro no ARP de LPCS, 38°43'32"N 009°21'19"W (eAIP Portugal, AD 2 LPCS).
    Teto da CTR de Cascais: 2000 ft ALT (mesma fonte); usa-se 2500 ft de altitude-pressão
    para cobrir a diferença QNH–1013 hPa.
  - Tejo (Belém/Algés): centro e raio escolhidos a partir dos próprios dados de 1 de julho
    (os trajetos HIB ocupam 38.678–38.698N, 9.177–9.252W, abaixo de 450 ft). Teto de
    1000 ft: a final da pista 02 passa por cima desta zona a ≈ 2000 ft. Para retirar só
    Cascais, basta apagar esta zona da lista.
  O script falha se alguma aeronave aceite na final cair numa zona de exclusão.

Uso:
  python figuras_dados_1julho.py --data-dir <pasta com CSV e METAR>
  python figuras_dados_1julho.py --selftest
Requer 'preprocessamento.py' na mesma pasta.
"""

import argparse
import csv
import math
import os

import preprocessamento as pp

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
DAY = {
    "date": "20260701",
    "aircraft_csv": "dados_1julho_12h16h.csv",
    "metar_file": "METAR_julho.txt",
}

EXCLUSION_ZONES = (
    {"name": "Cascais aerodrome (LPCS)",
     "lat_deg": 38 + 43 / 60 + 32 / 3600,         # 38°43'32"N (eAIP AD 2 LPCS)
     "lon_deg": -(9 + 21 / 60 + 19 / 3600),       # 009°21'19"W
     "radius_nm": 2.0,
     "max_alt_ft": 2500.0},                       # altitude-pressão (baroH · 100)
    {"name": "Tagus low-level traffic (Belém/Algés)",
     "lat_deg": 38.688,
     "lon_deg": -9.214,
     "radius_nm": 2.0,
     "max_alt_ft": 1000.0},
)

# Elementos de referência das figuras (iguais aos dos scripts antigos)
GLIDE_PATH = {"dist_nm": (11.3, 0.0), "alt_ft": (4000.0, 330.0)}
RUNWAY_ELEVATION_FT = 330.0
INTERMEDIATE_FIX_NM = 11.3
REJECTED_DIST_RANGE_NM = (-5.0, 15.0)   # Fig. 4.9: rejeitados só desenhados neste intervalo
EARTH_RADIUS_M = 6371008.8              # raio médio (IUGG)


# =============================================================================
# GEOMETRIA
# =============================================================================
def runway_frame_nm(x, y, cfg=pp.PREPROCESSING):
    """Distância com sinal ao longo do eixo de aproximação (positiva antes do limiar, i.e. na
    final; negativa depois) e afastamento lateral (positivo à direita de quem aproxima), em NM."""
    dx, dy = x - cfg["x_thr_m"], y - cfg["y_thr_m"]
    a = math.radians(cfg["rwy_true_bearing_deg"] + 180.0)      # rumo de aproximação (de onde vem)
    along = dx * math.sin(a) + dy * math.cos(a)
    cross = dy * math.sin(a) - dx * math.cos(a)                 # à direita do rumo 22.72° > 0
    return along / pp.NM_TO_M, cross / pp.NM_TO_M


def great_circle_nm(lat1, lon1, lat2, lon2):
    """Distância ortodrómica (haversine, esfera de raio médio) em NM."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h)) / pp.NM_TO_M


# =============================================================================
# DADOS
# =============================================================================
def aircraft_key(row):
    """Aeronave: endereço Mode S; sem ele, o track number (o mesmo critério do texto da §4.1.2)."""
    mode_s = (row.get("modeS") or "").strip()
    return mode_s if mode_s else f"trk{(row.get('trkNr') or '').strip()}"


def _f(row, key):
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def load_messages(csv_path, metar, cfg):
    """Lê o CSV e devolve uma lista de mensagens com posição (x, y, baroH). Para cada uma:
    distância/afastamento no referencial da pista, altitude-pressão, altitude verdadeira
    (Eq. 3.9, METAR válido) e se é aceite pelo pré-processamento (T §3.2.1)."""
    msgs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            x, y, fl = _f(row, "x"), _f(row, "y"), _f(row, "baroH")
            if x is None or y is None or fl is None:
                continue
            t = pp.parse_tod_s(row["tod"])                       # UTC
            report = pp.metar_valid_at(metar, int(t))
            p = pp.static_pressure_hpa(fl * 100.0 * pp.FT_TO_M)
            along, cross = runway_frame_nm(x, y, cfg)
            msgs.append({
                "key": aircraft_key(row), "trk": (row.get("trkNr") or "").strip(),
                "x": x, "y": y, "along_nm": along, "cross_nm": cross,
                "lat": _f(row, "latitude"), "lon": _f(row, "longitude"),
                "h_baro_ft": fl * 100.0,
                "h_true_ft": pp.true_altitude_m(p, report.qnh_hpa, report.temp_c) / pp.FT_TO_M,
                "accepted": pp.observation_from_row(row, report, cfg) is not None,
            })
    return msgs


def zone_aircraft(msgs, zone):
    """Aeronaves com pelo menos uma mensagem a < radius_nm do centro da zona e abaixo de max_alt_ft."""
    out = set()
    for m in msgs:
        if m["lat"] is None or m["lon"] is None or m["h_baro_ft"] >= zone["max_alt_ft"]:
            continue
        if great_circle_nm(m["lat"], m["lon"], zone["lat_deg"], zone["lon_deg"]) < zone["radius_nm"]:
            out.add(m["key"])
    return out


def excluded_aircraft(msgs, zones=EXCLUSION_ZONES):
    """{nome da zona: aeronaves excluídas das vistas verticais}."""
    return {z["name"]: zone_aircraft(msgs, z) for z in zones}


def summary(n_rows, msgs, excluded_by_zone):
    accepted = {m["key"] for m in msgs if m["accepted"]}
    excluded = set().union(*excluded_by_zone.values())
    clash = accepted & excluded
    if clash:
        raise ValueError(f"Aeronaves aceites na final dentro de uma zona de exclusão: {sorted(clash)}")
    print(f"Mensagens no ficheiro: {n_rows} | com posição: {len(msgs)}")
    keys = {m["key"] for m in msgs}
    print(f"Aeronaves com Mode S: {sum(not k.startswith('trk') for k in keys)} | "
          f"sem Mode S (por track number): {sum(k.startswith('trk') for k in keys)} | "
          f"track numbers: {len({m['trk'] for m in msgs})}")
    print(f"Mensagens aceites: {sum(m['accepted'] for m in msgs)} | aeronaves aceites: {len(accepted)}")
    for name, keys in excluded_by_zone.items():
        n = sum(m["key"] in keys for m in msgs)
        print(f"Retirado das vistas verticais — {name}: {len(keys)} aeronaves, {n} mensagens")
    return excluded


# =============================================================================
# FIGURAS (inglês britânico, estilo dos scripts antigos)
# =============================================================================
def _top_view_labels(ax):
    ax.set_xlabel("X Coordinate (metres)")
    ax.set_ylabel("Y Coordinate (metres)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    ax.axis("equal")


def plot_raw(msgs, excluded, cfg, path):
    """Fig. 4.7: vista de topo (todas as mensagens) e vista vertical (sem as zonas excluídas)."""
    import matplotlib.pyplot as plt

    vert = [m for m in msgs if m["key"] not in excluded]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter([m["x"] for m in msgs], [m["y"] for m in msgs], color="grey", s=2, alpha=0.3,
                label="Raw Traffic")
    ax1.scatter([cfg["x_thr_m"]], [cfg["y_thr_m"]], color="blue", marker="*", s=150, label="THR 02")
    _top_view_labels(ax1)

    d = [m["along_nm"] for m in vert]
    ax2.scatter(d, [m["h_baro_ft"] for m in vert], color="grey", s=2, alpha=0.3, label="Raw Traffic")
    ax2.plot(GLIDE_PATH["dist_nm"], GLIDE_PATH["alt_ft"], color="blue", linewidth=2,
             label="Theoretical Glide Path (3º)")
    ax2.set_xlim(max(d) + 1, min(d) - 1)          # invertido: aproximação da esquerda para a direita
    ax2.set_xlabel("Distance to Threshold (NM)")
    ax2.set_ylabel("Barometric Altitude (ft)")
    ax2.axvline(x=0, color="red", linestyle=":", alpha=0.7, label="THR 02")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_preprocessed(msgs, excluded, cfg, path):
    """Fig. 4.9: aceites a verde, rejeitados a vermelho (só entre -5 e 15 NM, como antes);
    vista vertical em altitude verdadeira e sem as zonas excluídas."""
    import matplotlib.pyplot as plt

    lo, hi = REJECTED_DIST_RANGE_NM
    acc = [m for m in msgs if m["accepted"]]
    rej = [m for m in msgs if not m["accepted"] and lo < m["along_nm"] < hi]
    rej_vert = [m for m in rej if m["key"] not in excluded]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter([m["x"] for m in rej], [m["y"] for m in rej], color="lightcoral", s=2, alpha=0.5,
                label="Rejected Traffic")
    ax1.scatter([m["x"] for m in acc], [m["y"] for m in acc], color="green", s=10, alpha=0.8,
                label="Final Approach (Accepted)")
    ax1.scatter([cfg["x_thr_m"]], [cfg["y_thr_m"]], color="blue", marker="*", s=200,
                label="Runway 02 Threshold (THR)")
    _top_view_labels(ax1)

    ax2.scatter([m["along_nm"] for m in rej_vert], [m["h_true_ft"] for m in rej_vert],
                color="lightcoral", s=2, alpha=0.5, label="Rejected Traffic")
    ax2.scatter([m["along_nm"] for m in acc], [m["h_true_ft"] for m in acc], color="green", s=10,
                alpha=0.8, label="Accepted Traffic")
    ax2.plot(GLIDE_PATH["dist_nm"], GLIDE_PATH["alt_ft"], color="blue", linewidth=2,
             label="Theoretical Glide Path (3º)")
    ax2.axhline(y=cfg["h_max_ft"], color="red", linestyle="--", label=f"Upper Limit ({cfg['h_max_ft']:.0f} ft)")
    ax2.axhline(y=RUNWAY_ELEVATION_FT, color="orange", linestyle="--",
                label=f"Runway Elevation ({RUNWAY_ELEVATION_FT:.0f} ft)")
    ax2.axvline(x=INTERMEDIATE_FIX_NM, color="purple", linestyle="--",
                label=f"Intermediate Fix ({INTERMEDIATE_FIX_NM:g} NM)")
    ax2.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    ax2.set_xlim(15, -3)
    ax2.set_ylim(-500, 6000)
    ax2.set_xlabel("Distance to Threshold (NM)")
    ax2.set_ylabel("True Altitude (ft)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    cfg = pp.PREPROCESSING
    # 1. Limiar: distância 0. Ponto a 5 NM no eixo da final: +5 NM, afastamento 0.
    a = math.radians(cfg["rwy_true_bearing_deg"] + 180.0)
    along, cross = runway_frame_nm(cfg["x_thr_m"], cfg["y_thr_m"])
    assert abs(along) < 1e-12 and abs(cross) < 1e-12
    x = cfg["x_thr_m"] + 5 * pp.NM_TO_M * math.sin(a)
    y = cfg["y_thr_m"] + 5 * pp.NM_TO_M * math.cos(a)
    along, cross = runway_frame_nm(x, y)
    assert abs(along - 5.0) < 1e-9 and abs(cross) < 1e-9
    # ... um ponto a Este do limiar (à direita de quem aproxima com rumo 022°) tem afastamento > 0.
    assert runway_frame_nm(cfg["x_thr_m"] + 1000.0, cfg["y_thr_m"])[1] > 0
    # ... depois do limiar (sobre a pista) a distância é negativa.
    assert runway_frame_nm(cfg["x_thr_m"] - 500 * math.sin(a), cfg["y_thr_m"] - 500 * math.cos(a))[0] < 0

    # 2. Haversine: 1' de latitude ≈ 1 NM (diferença < 0.5 % com a esfera de raio médio).
    assert abs(great_circle_nm(38.0, -9.0, 38.0 + 1 / 60, -9.0) - 1.0) < 0.005

    # 3. Zona de Cascais: aeronave baixa junto ao ARP é marcada; alta (≥ 2500 ft) ou a 3 NM não é.
    c = EXCLUSION_ZONES[0]
    m = lambda k, dlat, h: {"key": k, "lat": c["lat_deg"] + dlat, "lon": c["lon_deg"], "h_baro_ft": h}
    msgs = [m("A", 0.0, 800.0), m("A", 0.5, 800.0), m("B", 0.0, 3000.0), m("C", 3 / 60, 500.0)]
    assert zone_aircraft(msgs, c) == {"A"}
    # ... e a aeronave aceite que cair numa zona faz falhar o resumo.
    try:
        summary(0, [{**msgs[0], "accepted": True, "trk": "1"}], {"z": {"A"}})
        raise AssertionError("faltou o erro")
    except ValueError:
        pass
    print("Testes de sanidade de figuras_dados_1julho.py: todos passaram.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="figuras_dados_1julho")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    pp.run_selftests()
    run_selftests()
    if args.selftest:
        return

    import matplotlib
    matplotlib.use("Agg")

    cfg = pp.PREPROCESSING
    csv_path = os.path.join(args.data_dir, DAY["aircraft_csv"])
    metar = pp.load_metar_schedule(os.path.join(args.data_dir, DAY["metar_file"]), DAY["date"])
    with open(csv_path, encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    msgs = load_messages(csv_path, metar, cfg)
    excluded = summary(n_rows, msgs, excluded_aircraft(msgs))

    os.makedirs(args.out, exist_ok=True)
    plot_raw(msgs, excluded, cfg, os.path.join(args.out, "dados_raw_1julho.png"))
    plot_preprocessed(msgs, excluded, cfg, os.path.join(args.out, "dados_preprocessados_1julho.png"))
    print(f"Figuras guardadas em: {args.out}")


if __name__ == "__main__":
    main()
