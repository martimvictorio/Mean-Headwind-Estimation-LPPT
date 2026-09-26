import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyse_lppt_wind():
  # 1. Lista com os nomes dos ficheiros METAR
  files = [
      "METAR_março.txt",
      "METAR_abril.txt",
      "METAR_maio.txt",
      "METAR_junho.txt",
      "METAR_julho.txt",
  ]
  wind_data = []

  # 2. Leitura sequencial dos ficheiros de texto
  for file_name in files:
    if not os.path.exists(file_name):
      print(f"File missing in directory: {file_name}")
      continue

    with open(file_name, "r") as file:
      for line in file:
        # 3. Filtrar linhas de observação do LPPT com dados de vento (KT)
        if "LPPT" in line and "KT" in line:
          parts = line.split()
          date_str = parts[0]

          if len(date_str) == 12 and date_str.isdigit():
            date_formatted = date_str[0:8]  # Formato YYYYMMDD (Data)
            hour_str = date_str[8:10]  # Extração da hora (HH)

            try:
              hour = int(hour_str)
            except ValueError:
              continue

            # Descartar dados da meia-noite até às 06:00 (00:00 a 05:59)
            # Período em que o aeroporto se encontra encerrado/sem operações relevantes
            if 0 <= hour < 6:
              continue

            wind_str = next(
                (p for p in parts if p.endswith("KT") and len(p) >= 7), None
            )

            if wind_str:
              direction_str = wind_str[0:3]
              speed_str = wind_str[3:5]

              try:
                speed = int(speed_str)
              except ValueError:
                continue

              # 4. Classificação operacional do vento
              if direction_str == "VRB":
                runway = "Unknown"
              else:
                try:
                  direction = int(direction_str)
                  if 115 <= direction <= 295:
                    runway = "South-Southwest Predominant Wind"
                  else:
                    runway = "North-Northeast Predominant Wind"
                except ValueError:
                  runway = "Unknown"

              wind_data.append({
                  "Date": date_formatted,
                  "Wind Speed": speed,
                  "Runway": runway,
              })

  # 5. Estruturação no Pandas
  df = pd.DataFrame(wind_data)

  if df.empty:
    print(
        "No data available after filtering night hours and valid runways."
    )
    return

  # Calcular a velocidade média global do vento à superfície para o período operacional (06h - 24h)
  overall_mean_speed = df["Wind Speed"].mean()
  print(
      "Overall Surface Mean Wind Speed (March - July, 06:00-24:00 UTC):"
      f" {overall_mean_speed:.2f} kt"
  )

  # Limpeza de direções variáveis desconhecidas e conversão da data
  df = df[df["Runway"] != "Unknown"].copy()
  df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

  # 6. Agrupamento diário avançado (.agg para Média, Mínimo, Máximo e Moda)
  df_daily = (
      df.groupby("Date")
      .agg(
          Mean_Speed=("Wind Speed", "mean"),
          Min_Speed=("Wind Speed", "min"),
          Max_Speed=("Wind Speed", "max"),
          Runway=("Runway", lambda x: x.mode()[0]),
      )
      .reset_index()
  )

  # 7. Renderização do Gráfico com Média, Intervalo Min-Max e Categorização por Cores
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(16, 6))

  # Banda sombreada para o intervalo diário (Mínimo a Máximo)
  plt.fill_between(
      df_daily["Date"],
      df_daily["Min_Speed"],
      df_daily["Max_Speed"],
      color="grey",
      alpha=0.2,
      label="Daily Min-Max Wind Speed Range",
      zorder=1,
  )

  # Linha de tendência diária (Média)
  sns.lineplot(
      data=df_daily,
      x="Date",
      y="Mean_Speed",
      color="grey",
      alpha=0.6,
      linewidth=1.5,
      zorder=2,
      label="Daily Average Surface Wind Speed",
  )

  # Linha horizontal estática indicando a média global do período operacional
  plt.axhline(
      y=overall_mean_speed,
      color="purple",
      linestyle="--",
      linewidth=2,
      label=f"Overall Mean Surface Wind ({overall_mean_speed:.1f} kt)",
      zorder=3,
  )

  # Dispersão colorida de acordo com a direção predominante (Azul e Laranja)
  sns.scatterplot(
      data=df_daily,
      x="Date",
      y="Mean_Speed",
      hue="Runway",
      palette={
          "North-Northeast Predominant Wind": "royalblue",
          "South-Southwest Predominant Wind": "darkorange",
      },
      s=80,
      zorder=4,
  )

  # Rótulos em Inglês Britânico
  plt.title(
      (
          "Daily Average, Minimum, and Maximum Surface Wind Speed and Wind"
          " Direction Categorisation (06:00 - 24:00 UTC)"
      ),
      fontsize=15,
      fontweight="bold",
  )
  plt.xlabel("Date (Continuous Timeline)", fontsize=12)
  plt.ylabel("Wind Speed (kt)", fontsize=12)

  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
  plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
  plt.xticks(rotation=45)

  # Legenda posicionada no canto superior direito
  plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  analyse_lppt_wind()