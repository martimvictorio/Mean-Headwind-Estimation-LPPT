# Wind on Final Approach — Código da tese

Código desenvolvido para a tese de mestrado **"Wind on Final Approach"**. O código estima o headwind na aproximação final à pista 02 de Lisboa (LPPT) e compara um filtro de Kalman com o baseline EUROCONTROL e com um benchmark METAR.

## Onde estão os resultados da tese

**Os resultados da tese são produzidos pelos ficheiros `.py` cujo nome começa por `resultados`.** O número no nome indica a secção da tese:

| Script | Secção | O que produz | Pasta de saída |
|---|---|---|---|
| `resultados_4_2_1_METAR.py` | §4.2.1 | Vento à superfície em LPPT a partir dos METAR (mar–jul 2026), Figs. 4.3 e 4.4 | `METAR_analise_vento/` |
| `resultados_4_2_1_vento_cenarios.py` | §4.2.1 | Vento à superfície (anemómetros 3A/3B) nos quatro cenários | `surface_wind_output/` |
| `resultados_4_2_e_4_3_flightdata.py` | §4.1.2 e §4.2 | Dados CAT062 em bruto e pré-processados de 1 de julho | `figuras_dados_1julho/` |
| `resultados_4_3_headwind.py` | §4.2 | Observações de vento local de 1 de julho | `resultados_4_2/` |
| `resultados_4_4_1_q_tuning.py` | §4.4.1 | Afinação do parâmetro q do filtro de Kalman | `resultados_q_1julho/` |
| `resultados_4_4_1_headwind_mean.py` | §4.4.1 | Filtro de Kalman para dois valores de q (perfil e headwind médio) | `resultados_visualizacao_1julho/` |
| `resultados_4_4_2_baseline.py` | §4.4.2 | Verificação do baseline EUROCONTROL | `resultados_baseline_1julho/` |
| `resultados_5_1.py` | §5.1 | Resumo dos cenários (volume de dados, vento) | `resultados_5_1/` |
| `resultados_5_2.py` | §5.2 | Avaliação nowcast dos três modelos | `resultados_5_2/` |
| `resultados_5_3.py` | §5.3 | Avaliação operacional dos três modelos | `resultados_5_3/` |

## Restantes ficheiros

**Módulos partilhados**, usados pelos scripts de resultados:
- `preprocessamento.py`: pré-processamento das mensagens CAT062 e cálculo do headwind observado
- `modelos.py`: filtro de Kalman, baseline EUROCONTROL e benchmark METAR
- `avaliacao.py`: avaliação nowcast e operacional e métricas (bias, RMSE, MAE)
- `cenarios.py`: definição dos cenários de avaliação e carregamento dos dados
- `estilo_figuras.py`: estilo comum das figuras

**Código exploratório**, que não faz parte dos resultados:
- `METAR_analise_<dia>.py`: análise do METAR de cada dia de cenário

**Dados:**
- `dados_<dia>_<período>.csv`: mensagens radar CAT062 de cada cenário (16 e 29 de março, 11 de abril, 6 de maio, 1 de julho de 2026)
- `METAR_<mês>.txt`: relatórios METAR de LPPT, de março a julho de 2026
- `WIND_3A_MSG_RAW_<DD>.csv` e `WIND_3B_MSG_RAW_<DD>.csv`: anemómetros IPMA (sensores 3A e 3B) nos dias dos cenários

Todos os tempos estão em UTC.

## Importante: manter todos os ficheiros na mesma pasta

Os scripts dependem uns dos outros. Os de resultados importam os módulos partilhados, e alguns importam outros scripts de resultados (por exemplo, `resultados_4_4_2_baseline.py` usa `resultados_4_4_1_q_tuning.py`). Os dados também são lidos da pasta onde estão os scripts.

**Todos os ficheiros `.py` e todos os ficheiros de dados têm de estar na mesma pasta.** Se mover algum ficheiro para outra pasta, o código deixa de funcionar.

## Como correr

Requisitos: Python 3.11, `numpy`, `pandas`, `matplotlib` e `seaborn`.

```
pip install numpy pandas matplotlib seaborn
```

Na pasta do repositório, corra o script do resultado que quer ver. Por exemplo:

```
python resultados_5_2.py
```

As figuras e as tabelas são guardadas na pasta de saída indicada na tabela acima.