"""
Pré-processamento das mensagens CAT062 e cálculo do headwind observado pelas aeronaves.

Módulo partilhado por todos os scripts (afinação, baseline, resultados). Não depende de
nenhum modelo nem de nenhum dia em particular: a data do cenário entra em cfg["scenario_date"].

Referências:
  [T] Tese WindOnFinal: §3.1.2 (METAR e anemómetros IPMA), §3.1.3 (replay causal), §3.2.1 (critérios),
      §3.2.2 (headwind), Tab. 3.2 (constantes ISA).

Origem: extraído sem alterações de 'kalman_tuning_1julho.py' (versão de 21-10-2026),
exceto:
  - Observation tem um campo novo, distance_nm (distância ao limiar 02, em NM), no fim e
    com valor por omissão, para não quebrar o código que cria Observation por posição.
  - surface_headwind_kt lança erro com vento VRB ou em falta, em vez de devolver 0 kt
    (decisão de 21-10-2026: o caso não ocorre nos cenários e não está definido na tese).

Referência temporal:
  - 'tod' do CSV, METAR e anemómetros IPMA estão todos em UTC. Não há conversões.
"""

import bisect
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# =============================================================================
# CONFIGURAÇÃO DO PRÉ-PROCESSAMENTO (único sítio onde é definida)
# =============================================================================
PREPROCESSING = {
    # --- Geometria da pista 02 (T §3.2.1) ---
    "x_thr_m": -433.76,
    "y_thr_m": 277.06,
    "rwy_true_bearing_deg": 22.72,
    "angular_margin_deg": 5.0,
    "max_dist_nm": 11.3,
    "h_min_ft": 400.0,                 # T Eq. (3.7)
    "h_max_ft": 4000.0,                # T Eq. (3.7)
    # Separação de passagens da mesma aeronave (borrego ou novo voo), T §3.5. Nos 5 ficheiros,
    # o maior intervalo dentro de uma passagem é 39 s e o menor entre passagens ~11 min.
    "passage_gap_s": 300,
}

# Constantes ISA (T Tab. 3.2) e conversões
P0_HPA = 1013.25
T0_K = 288.15
LAMBDA = -0.0065          # K/m
R_AIR = 287.05287         # J/(kg·K)
G0 = 9.80665              # m/s²
FT_TO_M = 0.3048
NM_TO_M = 1852.0
MS_TO_KT = 3600.0 / NM_TO_M


# =============================================================================
# TEMPO: 'tod', METAR e IPMA EM UTC (ver alterações de 29-10-2026)
# =============================================================================
def _parse_date(date_str):
    return datetime.strptime(date_str, "%Y%m%d")


def fmt_hhmm(seconds):
    seconds = int(round(seconds)) % 86400
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


# =============================================================================
# METAR (T §3.1.2, Tab. 3.2)
# =============================================================================
@dataclass(frozen=True)
class MetarReport:
    t_utc_s: int          # segundos desde 00:00 UTC do dia
    qnh_hpa: float
    temp_c: float
    wind_dir_deg: float   # NaN se VRB ou ausente
    wind_speed_kt: float


def load_metar_schedule(path, date_str):
    """Lê os METAR do dia pedido (prefixo AAAAMMDDhhmm, UTC) e devolve-os ordenados no tempo."""
    reports = []
    line_re = re.compile(rf"^{date_str}(\d{{2}})(\d{{2}})\s+METAR")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            qnh = re.search(r"\bQ(\d{4})\b", line)
            tmp = re.search(r"\s(M?\d{2})/(M?\d{2})\s", line)
            if not (qnh and tmp):
                continue  # METAR incompleto: ignorado, não substituído por ISA
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60
            wnd = re.search(r"\s(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\s", line)
            w_dir, w_spd = float("nan"), 0.0
            if wnd:
                w_spd = float(wnd.group(2))
                if wnd.group(1) != "VRB":
                    w_dir = float(wnd.group(1))
            reports.append(MetarReport(t, float(qnh.group(1)),
                                       float(tmp.group(1).replace("M", "-")), w_dir, w_spd))
    reports.sort(key=lambda r: r.t_utc_s)
    if not reports:
        raise ValueError(f"Nenhum METAR encontrado para {date_str} em {path}")
    return reports


def metar_valid_at(reports, t_utc_s):
    """METAR mais recente emitido até t (causal). Antes do primeiro, usa o primeiro e avisa."""
    times = [r.t_utc_s for r in reports]
    i = bisect.bisect_right(times, t_utc_s) - 1
    if i < 0:
        print(f"[AVISO] Sem METAR antes de t={t_utc_s}s; usado o primeiro disponível.")
        return reports[0]
    return reports[i]


def surface_headwind_kt(report, rwy_bearing_deg):
    """T Eq. (3.1): V_HW = V_w·cos(θ_w − θ_RWY), com o vento médio do METAR.
    Vento VRB ou em falta: erro (convenção não definida na tese; não ocorre nos cenários)."""
    if math.isnan(report.wind_dir_deg):
        raise ValueError(f"METAR das {fmt_hhmm(report.t_utc_s)} UTC com vento VRB ou em falta: "
                         "headwind indefinido.")
    return report.wind_speed_kt * math.cos(math.radians(report.wind_dir_deg - rwy_bearing_deg))


# =============================================================================
# ANEMÓMETROS IPMA 3A/3B (T §3.1.2)
# =============================================================================
# Formato: ";Dir;Speed", uma linha por amostra de 3 s: "AAAA-MM-DD hh:mm:ss;ddd;v.v".
# Hora em UTC e velocidade em m/s: verificado em 21-10-2026 contra os METAR dos quatro
# cenários (a média de 10 min do IPMA coincide com o METAR sem desfasamento horário;
# diferença média < 1 kt após converter m/s -> kt).

@dataclass(frozen=True)
class SurfaceWindSeries:
    sensor: str
    t_utc_s: np.ndarray        # s desde 00:00 UTC do dia do cenário
    headwind_kt: np.ndarray    # T Eq. (3.1) em cada amostra


def load_ipma_headwind(path, date_str, rwy_bearing_deg, sensor=""):
    """Lê um ficheiro IPMA e calcula o headwind (Eq. 3.1) em cada amostra de 3 s, ANTES
    de qualquer média (evita médias de direção). Linhas sem direção ou velocidade são
    ignoradas."""
    day = _parse_date(date_str)
    t, hw = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)                                    # cabeçalho ";Dir;Speed"
        for row in reader:
            if len(row) < 3 or not row[1].strip() or not row[2].strip():
                continue
            ts = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
            speed_kt = float(row[2]) * MS_TO_KT
            t.append((ts - day).total_seconds())
            hw.append(speed_kt * math.cos(math.radians(float(row[1]) - rwy_bearing_deg)))
    if not t:
        raise ValueError(f"Ficheiro IPMA sem amostras válidas: {path}")
    order = np.argsort(t)
    return SurfaceWindSeries(sensor, np.asarray(t)[order], np.asarray(hw)[order])


def mean_surface_headwind_kt(series_list, t0_utc, t1_utc):
    """Headwind médio à superfície no intervalo (t0, t1]: média das amostras de cada
    sensor e depois média dos sensores com dados (T §3.1.2). None se nenhum tiver."""
    means = []
    for s in series_list:
        m = (s.t_utc_s > t0_utc) & (s.t_utc_s <= t1_utc)
        if m.any():
            means.append(float(s.headwind_kt[m].mean()))
    return float(np.mean(means)) if means else None


# Recuo máximo (em minutos) quando nenhum sensor tem amostras no minuto pedido.
SURFACE_BLOCK_MAX_LOOKBACK_MIN = 10


def surface_headwind_block_kt(series_list, t_utc_s, max_lookback_min=SURFACE_BLOCK_MAX_LOOKBACK_MIN):
    """Headwind de superfície no instante t (T §3.1.2, decisão de 21-10-2026): média das
    amostras de 3 s do último minuto de relógio COMPLETO, [m − 60, m) com m = ⌊t/60⌋·60,
    por sensor e depois entre sensores. O valor é constante durante cada minuto (bloco).
    Se nenhum sensor tiver amostras nesse minuto, usa o minuto anterior (até
    max_lookback_min minutos) e avisa. Devolve (headwind_kt, início do minuto usado)."""
    m = math.floor(t_utc_s / 60.0) * 60
    for back in range(1, max_lookback_min + 1):
        t0 = m - 60 * back
        means = []
        for s in series_list:
            sel = (s.t_utc_s >= t0) & (s.t_utc_s < t0 + 60)
            if sel.any():
                means.append(float(s.headwind_kt[sel].mean()))
        if means:
            if back > 1:
                print(f"[AVISO] Sem dados IPMA no minuto {fmt_hhmm(m - 60)} UTC; "
                      f"usado o minuto {fmt_hhmm(t0)} UTC.")
            return float(np.mean(means)), t0
    raise ValueError(f"Sem dados IPMA nos {max_lookback_min} min anteriores a {fmt_hhmm(t_utc_s)} UTC.")


# =============================================================================
# FÍSICA: ALTITUDE E VELOCIDADE (T §3.2.1 e §3.2.2)
# =============================================================================
def static_pressure_hpa(hp_m):
    """T Eq. (3.8): pressão estática a partir da altitude-pressão (ISA)."""
    return P0_HPA * (1.0 + LAMBDA * hp_m / T0_K) ** (-G0 / (LAMBDA * R_AIR))


def true_altitude_m(p_hpa, qnh_hpa, t_local_c):
    """T Eq. (3.9): altitude verdadeira com QNH e temperatura local."""
    t_local_k = t_local_c + 273.15
    return (t_local_k / LAMBDA) * ((p_hpa / qnh_hpa) ** (-LAMBDA * R_AIR / G0) - 1.0)


def true_airspeed_kt(ias_kt, p_hpa, h_true_m, t_local_c):
    """T Eqs. (2.4), (2.6), (3.15): Vtas = Vias·sqrt(ρ0/ρ), com T_aeronave = T_local + λ·h_true."""
    t_aircraft_k = t_local_c + 273.15 + LAMBDA * h_true_m
    rho = (p_hpa * 100.0) / (R_AIR * t_aircraft_k)
    rho0 = (P0_HPA * 100.0) / (R_AIR * T0_K)
    return ias_kt * math.sqrt(rho0 / rho)


def bearing_deg(dx, dy):
    """Ângulo aeronáutico (0° = Norte, sentido horário) em [0, 360)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


# =============================================================================
# OBSERVAÇÕES (T §3.2.1, §3.2.2)
# =============================================================================
@dataclass
class Observation:
    step: int            # passo do relógio virtual (s desde 00:00 UTC) em que a mensagem é recebida
    aircraft: str        # chave da PASSAGEM, "modeS#n" (ver assign_passages)
    mode_s: str          # endereço Mode S (I062/380/ADR): identifica a aeronave
    t_raw: float         # instante da mensagem, s desde 00:00 UTC
    h_ft: float
    headwind_kt: float
    distance_nm: float = float("nan")   # distância horizontal ao limiar 02 (Eq. 3.3)

    @property
    def track_key(self):
        """Compatibilidade com scripts antigos: devolve o endereço Mode S (já sem trkNr)."""
        return self.mode_s


def parse_tod_s(tod):
    """'tod' (hh:mm:ss.sss, UTC) -> s desde 00:00 UTC do dia do cenário."""
    h, m, s = tod.strip().split(":")
    t = int(h) * 3600 + int(m) * 60 + float(s)
    if not 0.0 <= t < 86400.0:
        raise ValueError(f"'tod' fora do dia: {tod!r}")
    return t


def _float(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def observation_from_row(row, metar, cfg):
    """Aplica os 5 critérios de T §3.2.1 e devolve (h_ft, headwind_kt, distance_nm) ou None."""
    vals = {k: _float(row, k) for k in ("x", "y", "baroH", "vx", "vy", "iar")}
    if any(v is None for v in vals.values()) or not (row.get("modeS") or "").strip():
        return None  # critério 1 (o endereço Mode S identifica a aeronave, T §3.2.1)

    dx, dy = vals["x"] - cfg["x_thr_m"], vals["y"] - cfg["y_thr_m"]
    d_nm = math.hypot(dx, dy) / NM_TO_M
    if not (0.0 < d_nm < cfg["max_dist_nm"]):
        return None  # critério 2, Eqs. (3.3)-(3.4)

    app_brg = (cfg["rwy_true_bearing_deg"] + 180.0) % 360.0
    m = cfg["angular_margin_deg"]
    if not (app_brg - m < bearing_deg(dx, dy) < app_brg + m):
        return None  # critério 3, Eqs. (3.5)-(3.6)

    p = static_pressure_hpa(vals["baroH"] * 100.0 * FT_TO_M)
    h_true_m = true_altitude_m(p, metar.qnh_hpa, metar.temp_c)
    h_ft = h_true_m / FT_TO_M
    if not (cfg["h_min_ft"] < h_ft < cfg["h_max_ft"]):
        return None  # critério 4, Eq. (3.7)

    trk = bearing_deg(vals["vx"], vals["vy"])
    rwy = cfg["rwy_true_bearing_deg"]
    if not (rwy - m < trk < rwy + m):
        return None  # critério 5, Eqs. (3.10)-(3.11)

    vg_kt = math.hypot(vals["vx"], vals["vy"]) * MS_TO_KT
    vtas_kt = true_airspeed_kt(vals["iar"], p, h_true_m, metar.temp_c)
    return h_ft, vtas_kt - vg_kt, d_nm  # T Eq. (3.13)


def load_observations(csv_path, metar_reports, cfg):
    """Replay causal: cada mensagem é recebida no passo k = ceil(t_track) (T §3.1.3).
    'tod' já está em UTC (I062/070), tal como o METAR: não há conversão.
    cfg tem de conter as chaves de PREPROCESSING (a data já não é usada aqui).
    Devolve (observações aceites, nº de mensagens lidas, passo inicial)."""
    obs, n_rows, t_first = [], 0, None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            n_rows += 1
            t_utc = parse_tod_s(row["tod"])
            t_first = t_utc if t_first is None else t_first
            res = observation_from_row(row, metar_valid_at(metar_reports, int(t_utc)), cfg)
            if res is None:
                continue
            mode_s = row["modeS"].strip()   # critério 1 garante que existe
            obs.append(Observation(int(math.ceil(t_utc)), "", mode_s, t_utc, *res))
    return obs, n_rows, int(math.ceil(t_first))


def assign_passages(obs, gap_s):
    """T §3.5: uma passagem é uma sequência de observações aceites do mesmo endereço Mode S;
    começa uma nova quando o intervalo entre observações aceites consecutivas > gap_s."""
    last_t, counter = {}, {}
    for o in sorted(obs, key=lambda o: o.t_raw):
        if o.mode_s not in last_t or o.t_raw - last_t[o.mode_s] > gap_s:
            counter[o.mode_s] = counter.get(o.mode_s, 0) + 1
        last_t[o.mode_s] = o.t_raw
        o.aircraft = f"{o.mode_s}#{counter[o.mode_s]}"
    return obs


def group_by(items, key):
    d = {}
    for it in items:
        d.setdefault(key(it), []).append(it)
    return d


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    # 1. Altitude: em ISA (QNH=1013.25, T=15 °C) h_true = h_p
    for hp_ft in (0.0, 1000.0, 3500.0):
        p = static_pressure_hpa(hp_ft * FT_TO_M)
        assert abs(true_altitude_m(p, P0_HPA, 15.0) / FT_TO_M - hp_ft) < 1e-6

    # 2. TAS: ao nível do mar em ISA, TAS = IAS
    assert abs(true_airspeed_kt(140.0, P0_HPA, 0.0, 15.0) - 140.0) < 1e-9

    # 3. 'tod' em UTC, sem conversão: 16:00:00.296 -> 57600.296 s; fora do dia = erro.
    assert abs(parse_tod_s("16:00:00.296") - 57600.296) < 1e-9
    assert fmt_hhmm(parse_tod_s("08:00:00.000")) == "08:00"
    try:
        parse_tod_s("24:00:00.000")
        raise AssertionError("'tod' fora do dia não lançou erro")
    except ValueError:
        pass

    # 4. Headwind de superfície (Eq. 3.1): vento alinhado com a pista = headwind total;
    #    vento a 90° = 0; VRB = erro.
    r = lambda d, v: MetarReport(0, P0_HPA, 15.0, d, v)   # atmosfera ISA
    rwy = PREPROCESSING["rwy_true_bearing_deg"]
    assert abs(surface_headwind_kt(r(rwy, 20.0), rwy) - 20.0) < 1e-12
    assert abs(surface_headwind_kt(r(rwy + 90.0, 20.0), rwy)) < 1e-12
    try:
        surface_headwind_kt(r(float("nan"), 5.0), rwy)
        raise AssertionError("VRB não lançou erro")
    except ValueError:
        pass

    # 5. Mensagem sintética no eixo da final, a 5 NM do limiar, rumo à pista, em ISA:
    #    é aceite, com distância 5 NM e headwind = TAS − GS.
    cfg = PREPROCESSING
    brg = math.radians(rwy + 180.0)
    d_m = 5.0 * NM_TO_M
    gs_ms = 130.0 / MS_TO_KT
    row = {"x": cfg["x_thr_m"] + d_m * math.sin(brg), "y": cfg["y_thr_m"] + d_m * math.cos(brg),
           "baroH": 16.0, "vx": gs_ms * math.sin(math.radians(rwy)),
           "vy": gs_ms * math.cos(math.radians(rwy)), "iar": 140.0, "trkNr": "1", "modeS": "4952a3"}
    row = {k: str(v) for k, v in row.items()}
    h_ft, hw, dist = observation_from_row(row, r(rwy, 0.0), cfg)
    p = static_pressure_hpa(1600.0 * FT_TO_M)
    assert abs(dist - 5.0) < 1e-9 and abs(h_ft - 1600.0) < 1e-6
    assert abs(hw - (true_airspeed_kt(140.0, p, h_ft * FT_TO_M, 15.0) - 130.0)) < 1e-9
    # ... e é rejeitada se o rumo se afastar mais de 5° do da pista (critério 5).
    turn = math.radians(rwy + 6.0)
    row_off = {**row, "vx": str(gs_ms * math.sin(turn)), "vy": str(gs_ms * math.cos(turn))}
    assert observation_from_row(row_off, r(rwy, 0.0), cfg) is None
    # ... e é rejeitada sem endereço Mode S (critério 1), mesmo com trkNr; aceite sem trkNr.
    assert observation_from_row({**row, "modeS": ""}, r(rwy, 0.0), cfg) is None
    assert observation_from_row({**row, "modeS": "  "}, r(rwy, 0.0), cfg) is None
    assert observation_from_row({**row, "trkNr": ""}, r(rwy, 0.0), cfg) is not None

    # 6. IPMA: ficheiro sintético com vento alinhado com a pista (headwind = velocidade) e
    #    uma linha vazia; média por sensor e depois entre sensores.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ipma.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(";Dir;Speed\n2026-04-11 10:00:00;022.72;10\n2026-04-11 10:00:03;;\n"
                    "2026-04-11 10:00:06;202.72;2\n")
        s = load_ipma_headwind(path, "20260411", rwy, "3A")
    assert len(s.t_utc_s) == 2 and s.t_utc_s[0] == 36000.0
    assert abs(s.headwind_kt[0] - 10 * MS_TO_KT) < 1e-9 and abs(s.headwind_kt[1] + 2 * MS_TO_KT) < 1e-9
    s2 = SurfaceWindSeries("3B", np.array([36003.0]), np.array([1.0]))
    expected = 0.5 * ((10 - 2) / 2 * MS_TO_KT + 1.0)
    assert abs(mean_surface_headwind_kt([s, s2], 35999, 36010) - expected) < 1e-9
    assert mean_surface_headwind_kt([s, s2], 36007, 36010) is None

    # 7. Bloco de 1 min: em t = 60.5 s usa-se o minuto [0, 60); a amostra em 60 s pertence
    #    ao minuto seguinte. Sensor sem dados nesse minuto não entra na média. Sem dados em
    #    nenhum sensor, recua um minuto.
    a = SurfaceWindSeries("3A", np.array([0.0, 30.0, 60.0, 120.0]), np.array([2.0, 4.0, 100.0, 7.0]))
    b = SurfaceWindSeries("3B", np.array([3.0, 33.0, 63.0]), np.array([6.0, 8.0, 100.0]))
    assert surface_headwind_block_kt([a, b], 60.5) == (0.5 * (3.0 + 7.0), 0)
    assert surface_headwind_block_kt([a, b], 119.999) == (0.5 * (3.0 + 7.0), 0)  # mesmo bloco
    assert surface_headwind_block_kt([a, b], 120.0) == (100.0, 60)                # muda ao minuto
    assert surface_headwind_block_kt([a, b], 180.0) == (7.0, 120)
    assert surface_headwind_block_kt([b], 245.0)[1] == 60          # recua de 180 para 60
    try:
        surface_headwind_block_kt([b], 3000.0)
        raise AssertionError("faltou o erro sem dados IPMA")
    except ValueError:
        pass

    # 8. Passagens: a mesma aeronave com intervalo > gap gera duas passagens.
    obs = [Observation(t, "", "A", t, 1000.0, 0.0) for t in (0, 10, 400, 410)]
    assign_passages(obs, 300)
    assert [o.aircraft for o in obs] == ["A#1", "A#1", "A#2", "A#2"]
    #    Aeronaves diferentes não se misturam, e track_key (compatibilidade) devolve o Mode S.
    obs = [Observation(0, "", "A", 0, 1000.0, 0.0), Observation(5, "", "B", 5, 1000.0, 0.0),
           Observation(10, "", "A", 10, 1000.0, 0.0)]
    assign_passages(obs, 300)
    assert [o.aircraft for o in obs] == ["A#1", "B#1", "A#1"] and obs[1].track_key == "B"

    print("Testes de sanidade de preprocessamento.py: todos passaram.")


if __name__ == "__main__":
    run_selftests()
