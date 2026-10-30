"""
Cenários de avaliação (Cap. 5) e carregamento dos dados de cada cenário.

Módulo partilhado pelos scripts dos resultados (§5.1, §5.2, §5.3) e pela afinação.
Depende só de 'preprocessamento.py' (pp), não de nenhum modelo. A data do cenário entra
no pré-processamento por cfg["scenario_date"].

Referências:
  [T] Tese WindOnFinal: Tab. 4.1 (cenários), §3.2.1 (pré-processamento), §3.5 (warm-up e
      aeronaves de teste).

Convenções de tempo (iguais às de pp):
  - 'tod' dos ficheiros CAT062, METAR e IPMA estão em UTC; não há conversões.
  - Os períodos dos cenários estão em UTC (os mesmos números de antes: os ficheiros
    cobrem 'tod' 16:00–20:00, etc., e 'tod' é UTC).

Alteração de 29-10-2026: period_local -> period_utc; removida a conversão hora local ->
UTC em read_time_span_utc e period_utc_s (ver preprocessamento.py).
"""

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

import preprocessamento as pp

# =============================================================================
# AVALIAÇÃO (T §3.5) — único sítio onde é definida
# =============================================================================
EVALUATION = {
    # Aeronaves que entram na final durante o warm-up não são de teste (T §3.5).
    "warmup_s": 15 * 60,
}


# =============================================================================
# GRELHA DE ALTITUDE (T §3.3, Eqs. 3.26–3.28, 3.32) — único sítio onde é definida
# =============================================================================
GRID = {
    # 36 níveis, 450 ... 3950 ft, Δh = 100 ft (T §3.3)
    "grid_bottom_ft": 450.0,
    "grid_top_ft": 3950.0,
    "grid_step_ft": 100.0,
}


@dataclass(frozen=True)
class AltitudeGrid:
    levels_ft: np.ndarray

    @classmethod
    def from_config(cls, cfg):
        lv = np.arange(cfg["grid_bottom_ft"], cfg["grid_top_ft"] + 0.5 * cfg["grid_step_ft"],
                       cfg["grid_step_ft"])
        return cls(lv)

    @property
    def n(self):
        return len(self.levels_ft)

    @property
    def step(self):
        return self.levels_ft[1] - self.levels_ft[0]

    def h_row(self, h_ft):
        """T Eqs. (3.26)-(3.28); fora de [450, 3950] ft, peso 1 no nível mais próximo."""
        row = np.zeros(self.n)
        lv = self.levels_ft
        if h_ft <= lv[0]:
            row[0] = 1.0
        elif h_ft >= lv[-1]:
            row[-1] = 1.0
        else:
            i = int(np.searchsorted(lv, h_ft, side="right")) - 1
            w_sup = (h_ft - lv[i]) / (lv[i + 1] - lv[i])
            row[i], row[i + 1] = 1.0 - w_sup, w_sup
        return row

    def nearest_level(self, h_ft):
        """T Eq. (3.32): h_i - Δh/2 < h ≤ h_i + Δh/2."""
        i = int(math.ceil((h_ft - self.levels_ft[0] - 0.5 * self.step) / self.step))
        return min(max(i, 0), self.n - 1)


# =============================================================================
# LISTA DE CENÁRIOS (único sítio onde são definidos) — T Tab. 4.1, pela mesma ordem
# =============================================================================
@dataclass(frozen=True)
class Scenario:
    key: str                      # identificador curto (argumento --scenarios)
    date: str                     # AAAAMMDD
    label: str                    # data por extenso (inglês britânico)
    period_utc: tuple             # ("HH:MM", "HH:MM"), UTC
    headwind_class: str           # "High" / "Low" (T Tab. 4.1)
    crosswind_class: str
    aircraft_csv: str
    metar_file: str
    ipma_files: tuple = field(default_factory=tuple)


SCENARIOS = (
    Scenario("11abril", "20260411", "11 April 2026", ("16:00", "20:00"), "High", "High",
             "dados_11abril_16h20h.csv", "METAR_abril.txt",
             ("WIND_3A_MSG_RAW_11.csv", "WIND_3B_MSG_RAW_11.csv")),
    Scenario("29marco", "20260329", "29 March 2026", ("08:00", "12:00"), "High", "Low",
             "dados_29março_08h12h.csv", "METAR_março.txt",
             ("WIND_3A_MSG_RAW_29.csv", "WIND_3B_MSG_RAW_29.csv")),
    Scenario("06maio", "20260506", "6 May 2026", ("16:00", "20:00"), "Low", "High",
             "dados_06maio_16h20h.csv", "METAR_maio.txt",
             ("WIND_3A_MSG_RAW_06.csv", "WIND_3B_MSG_RAW_06.csv")),
    Scenario("16marco", "20260316", "16 March 2026", ("12:00", "16:00"), "Low", "Low",
             "dados_16março_12h16h.csv", "METAR_março.txt",
             ("WIND_3A_MSG_RAW_16.csv", "WIND_3B_MSG_RAW_16.csv")),
)
SCENARIOS_BY_KEY = {s.key: s for s in SCENARIOS}


def scenario_number(scenario):
    """Número do cenário (1–4) pela ordem da Tab. 4.1, i.e. a ordem de SCENARIOS."""
    return SCENARIOS.index(scenario) + 1


def scenario_name(scenario):
    """Nome usado nas tabelas e figuras: "Scenario n"."""
    return f"Scenario {scenario_number(scenario)}"

# Tolerância para avisar quando os dados não cobrem o período declarado (s).
PERIOD_GAP_WARNING_S = 120


def scenario_config(scenario):
    """Configuração do pré-processamento, da grelha e da avaliação com a data do cenário."""
    return {**pp.PREPROCESSING, **GRID, **EVALUATION, "scenario_date": scenario.date}


def save_config(cfg, out_dir):
    """Guarda a configuração efetivamente usada e as versões, para reprodutibilidade."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_usada.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg, "numpy": np.__version__, "python": sys.version}, f, indent=2)


# =============================================================================
# DADOS CARREGADOS DE UM CENÁRIO
# =============================================================================
@dataclass
class ScenarioData:
    scenario: Scenario
    cfg: dict
    metar: list
    obs: list                     # observações aceites (T §3.2.1), já com passagens
    n_messages: int               # linhas do CSV CAT062
    start_step: int               # s desde 00:00 UTC (primeira mensagem, arredondada para cima)
    end_step: int                 # última mensagem do ficheiro
    first_step_by_passage: dict   # passagem -> 1.º passo com observação aceite
    test_ids: list                # passagens de teste (pós warm-up, T §3.5)
    max_metar_age_s: int          # maior idade do METAR usado durante o período
    surface_wind: list            # pp.SurfaceWindSeries dos sensores IPMA 3A e 3B


def select_test_passages(first_step_by_passage, start_step, warmup_s):
    """T §3.5: são de teste as passagens cuja 1.ª observação aceite ocorre depois do
    warm-up (k ≥ início + warm-up). Mesma regra de kalman_tuning_1julho.sweep_q e
    baseline_1julho.prepare."""
    return sorted(j for j, k in first_step_by_passage.items() if k >= start_step + warmup_s)


def read_time_span_utc(csv_path):
    """Instantes (s desde 00:00 UTC) da primeira e da última mensagem do ficheiro.
    Lê só a coluna 'tod'; serve para verificar o período do cenário."""
    first = last = None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            t = pp.parse_tod_s(row["tod"])                      # 'tod' em UTC
            first = t if first is None else min(first, t)
            last = t if last is None else max(last, t)
    if first is None:
        raise ValueError(f"Ficheiro sem mensagens: {csv_path}")
    return first, last


def period_utc_s(scenario):
    """Período declarado (UTC) em s desde 00:00 UTC."""
    def to_s(hhmm):
        h, m = hhmm.split(":")
        return int(h) * 3600 + int(m) * 60
    return to_s(scenario.period_utc[0]), to_s(scenario.period_utc[1])


def check_period(scenario, first_utc, last_utc):
    """Erro se houver mensagens fora do período declarado; aviso se o ficheiro começar
    ou acabar mais de PERIOD_GAP_WARNING_S longe dos limites."""
    p0, p1 = period_utc_s(scenario)
    if first_utc < p0 or last_utc > p1:
        raise ValueError(f"[{scenario.key}] mensagens fora do período {scenario.period_utc} UTC "
                         f"(primeira {pp.fmt_hhmm(first_utc)} UTC, última {pp.fmt_hhmm(last_utc)} UTC).")
    if first_utc - p0 > PERIOD_GAP_WARNING_S or p1 - last_utc > PERIOD_GAP_WARNING_S:
        print(f"[AVISO] [{scenario.key}] os dados não cobrem todo o período declarado.")


def max_metar_age_s(metar, start_utc, end_utc, sample_s=60):
    """Maior idade (s) do METAR válido (pp.metar_valid_at) durante o período.
    Deteta METAR em falta: em LPPT são emitidos de 30 em 30 min."""
    ages = [t - pp.metar_valid_at(metar, t).t_utc_s
            for t in range(int(start_utc), int(end_utc) + 1, sample_s)]
    return max(ages)


def load_scenario(scenario, data_dir="."):
    """Carrega um cenário (pré-processamento de pp) e define as aeronaves de teste."""
    cfg = scenario_config(scenario)
    path = lambda name: os.path.join(data_dir, name)

    first_utc, last_utc = read_time_span_utc(path(scenario.aircraft_csv))
    check_period(scenario, first_utc, last_utc)

    metar = pp.load_metar_schedule(path(scenario.metar_file), scenario.date)
    obs, n_messages, start_step = pp.load_observations(path(scenario.aircraft_csv), metar, cfg)
    if not obs:
        raise ValueError(f"[{scenario.key}] nenhuma observação aceite no pré-processamento.")
    obs = pp.assign_passages(obs, cfg["passage_gap_s"])

    first_step = {j: min(o.step for o in v)
                  for j, v in pp.group_by(obs, lambda o: o.aircraft).items()}
    return ScenarioData(
        scenario=scenario, cfg=cfg, metar=metar,
        obs=obs, n_messages=n_messages, start_step=start_step,
        end_step=int(math.ceil(last_utc)), first_step_by_passage=first_step,
        test_ids=select_test_passages(first_step, start_step, cfg["warmup_s"]),
        max_metar_age_s=max_metar_age_s(metar, first_utc, last_utc),
        surface_wind=[pp.load_ipma_headwind(path(f), scenario.date, cfg["rwy_true_bearing_deg"],
                                            f.split("_")[1]) for f in scenario.ipma_files])


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    # 1. Lista de cenários coerente: chaves únicas, datas válidas, período com 4 h.
    assert len(SCENARIOS_BY_KEY) == len(SCENARIOS)
    for s in SCENARIOS:
        p0, p1 = period_utc_s(s)
        assert p1 - p0 == 4 * 3600, s.key

    # 2. Período em UTC, sem conversão (29 de março já é UTC+1 em Lisboa, mas é irrelevante).
    assert period_utc_s(SCENARIOS_BY_KEY["29marco"]) == (8 * 3600, 12 * 3600)
    assert period_utc_s(SCENARIOS_BY_KEY["11abril"]) == (16 * 3600, 20 * 3600)
    # Numeração pela ordem da Tab. 4.1.
    assert [scenario_name(s) for s in SCENARIOS] == [f"Scenario {n}" for n in range(1, 5)]
    assert scenario_number(SCENARIOS_BY_KEY["11abril"]) == 1
    assert scenario_number(SCENARIOS_BY_KEY["16marco"]) == 4

    # 3. Aeronaves de teste: fronteira do warm-up incluída (k = início + warm-up conta).
    first = {"A#1": 100, "B#1": 100 + 899, "C#1": 100 + 900, "D#1": 5000}
    assert select_test_passages(first, 100, 900) == ["C#1", "D#1"]

    # 4. Idade do METAR: METAR às 00:00 e 00:30; no período 00:10–00:50 a maior idade é
    #    29 min (às 00:29, antes de o seguinte ser emitido).
    m = [pp.MetarReport(0, 1013.0, 15.0, 0.0, 0.0), pp.MetarReport(1800, 1013.0, 15.0, 0.0, 0.0)]
    assert max_metar_age_s(m, 600, 3000) == 1740

    print("Testes de sanidade de cenarios.py: todos passaram.")
