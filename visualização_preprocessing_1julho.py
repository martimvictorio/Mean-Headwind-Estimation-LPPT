import csv
import math
import re
import matplotlib.pyplot as plt
import os

# ==========================================
# CONSTANTES DA TESE (Secção 3.2.1)
# ==========================================
X_THR = -433.76     # Coordenada X da cabeceira 02 (m)
Y_THR = 277.06      # Coordenada Y da cabeceira 02 (m)
M_PARA_NM = 1852.0  # Fator de conversão (metros para NM)
FT_PARA_M = 0.3048  # Fator de conversão (pés para metros)

# ==========================================
# CONSTANTES DA ATMOSFERA PADRÃO (ISA)
# ==========================================
P_ISA = 1013.25      # Pressão ao nível do mar (hPa)
T_ISA = 288.15       # Temperatura ao nível do mar (K)
LAMBDA_ISA = -0.0065 # Gradiente térmico troposférico (K/m)
R_GAS = 287.05287    # Constante específica do gás para o ar seco (J/(kg·K))
G0 = 9.80665         # Aceleração da gravidade (m/s²)

def carregar_cronograma_metar(caminho_metar, dia_alvo):
    """
    Lê o ficheiro METAR e cria um dicionário cronológico para o dia especificado.
    """
    cronograma = {}
    if not os.path.exists(caminho_metar):
        print(f"Warning: File {caminho_metar} not found. Using default weather.")
        return cronograma
        
    with open(caminho_metar, 'r', encoding='utf-8') as f:
        for linha in f:
            padrao = rf'\b{dia_alvo}(\d{{2}})(\d{{2}})Z\b'
            match = re.search(padrao, linha)
            if match:
                hora = int(match.group(1))
                minuto = int(match.group(2))
                cronograma[(hora, minuto)] = linha.strip()
                
    return dict(sorted(cronograma.items()))

def processar_metar(metar_str):
    """
    Analisa a string METAR para extrair o QNH e a temperatura local.
    """
    match_qnh = re.search(r'Q(\d{4})', metar_str)
    qnh = float(match_qnh.group(1)) if match_qnh else 1013.25
    
    match_temp = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar_str)
    if match_temp:
        temp_str = match_temp.group(1)
        temp_celsius = float(temp_str.replace('M', '-'))
    else:
        temp_celsius = 15.0

    return qnh, temp_celsius

def extrair_hora_minuto_msg(msg):
    """
    Extrai a hora e o minuto do campo 'tod' (Time Of Track Information).
    """
    tod = msg.get('tod', '')
    if not tod:
        return 0, 0
        
    if ':' in str(tod):
        partes = str(tod).split(':')
        return int(partes[0]), int(partes[1])
    
    try:
        segundos = float(tod)
        hora = int(segundos // 3600)
        minuto = int((segundos % 3600) // 60)
        return hora, minuto
    except ValueError:
        return 0, 0

def mensagem_cumpre_requisitos_grafico(msg, qnh_local, temp_local_celsius):
    """
    Aplica os filtros espaciais definidos na tese e calcula a altitude verdadeira.
    Calcula a distância projetada para diferenciar aeronaves a chegar e aeronaves na pista.
    """
    try:
        campos_obrigatorios = ['x', 'y', 'baroH', 'vx', 'vy']
        for campo in campos_obrigatorios:
            if not msg.get(campo) or str(msg[campo]).strip() == '':
                return False, None, None, None, None, None

        x, y = float(msg['x']), float(msg['y'])
        baroH = float(msg['baroH'])
        vx, vy = float(msg['vx']), float(msg['vy'])
        
        # Obter o Track Number para contagem de voos únicos
        trk_nr = msg.get('trkNr', None)

        # 1. Distância Euclidiana (Usada apenas para os limites absolutos da tese)
        distancia_euclidiana_nm = math.sqrt((x - X_THR)**2 + (y - Y_THR)**2) / M_PARA_NM
        
        # 1.1 Distância Projetada com Sinal (Para os gráficos)
        # Projeção no eixo de 205º (Azimute de aproximação RWY 02)
        # Valores positivos = Em aproximação | Valores negativos = Sobre a pista / Descolagem
        rad_205 = math.radians(205)
        distancia_projetada_nm = ((x - X_THR) * math.sin(rad_205) + (y - Y_THR) * math.cos(rad_205)) / M_PARA_NM

        # 2. Cálculo da Altitude Verdadeira
        h_p_ft = baroH * 100 
        h_p_m = h_p_ft * FT_PARA_M
        
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p = P_ISA * (1 + (LAMBDA_ISA * h_p_m) / T_ISA) ** expoente_p
        
        t_local_k = temp_local_celsius + 273.15
        expoente_h = -(LAMBDA_ISA * R_GAS) / G0
        h_true_m = (t_local_k / LAMBDA_ISA) * ((p / qnh_local) ** expoente_h - 1)
        
        h_real = h_true_m / FT_PARA_M

        # Filtro 1: Distância (0 a 11.3 NM) baseado na distância pura
        if not (0 < distancia_euclidiana_nm < 11.3):
            return False, x, y, distancia_projetada_nm, h_real, trk_nr

        # Filtro 2: Azimute Aeronáutico (200º a 210º)
        bearing = math.degrees(math.atan2(x - X_THR, y - Y_THR))
        if bearing < 0: bearing += 360
        if not (200 <= bearing <= 210):
            return False, x, y, distancia_projetada_nm, h_real, trk_nr

        # Filtro 3: Altitude Real (330 ft a 4000 ft)
        if not (330 < h_real < 4000):
            return False, x, y, distancia_projetada_nm, h_real, trk_nr

        # Filtro 4: Track Angle (20º a 30º)
        chi_g = math.degrees(math.atan2(vx, vy))
        if chi_g < 0: chi_g += 360
        if not (20 < chi_g < 30):
            return False, x, y, distancia_projetada_nm, h_real, trk_nr

        return True, x, y, distancia_projetada_nm, h_real, trk_nr

    except ValueError:
        return False, None, None, None, None, None

def gerar_graficos_tese(caminho_csv, caminho_metar, dia_alvo):
    """
    Simula a reprodução dos dados, contabiliza voos e gera os gráficos corrigidos.
    """
    cronograma_metar = carregar_cronograma_metar(caminho_metar, dia_alvo)
    chaves_tempo = list(cronograma_metar.keys())
    
    qnh_atual, temp_atual = 1013.25, 15.0
    metar_ativo = "Default ISA Conditions"
    
    if chaves_tempo:
        metar_ativo = cronograma_metar[chaves_tempo[0]]
        qnh_atual, temp_atual = processar_metar(metar_ativo)
    
    print("="*60)
    print("INITIALISING DATA REPLAY SIMULATOR...")
    print(f"Starting weather -> {metar_ativo}")
    print("="*60)
    
    x_rejeitados, y_rejeitados = [], []
    x_aceites, y_aceites = [], []
    distancias_aceites, altitudes_aceites = [], []
    distancias_rejeitadas, altitudes_rejeitadas = [], []
    
    total_mensagens = 0
    voos_aceites_unicos = set() # Estrutura que guarda apenas os números de track únicos
    indice_proximo_metar = 1 if len(chaves_tempo) > 1 else 0

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        
        for msg in leitor_csv:
            total_mensagens += 1
            
            hora_msg, min_msg = extrair_hora_minuto_msg(msg)
            
            if indice_proximo_metar < len(chaves_tempo):
                hora_proximo, min_proximo = chaves_tempo[indice_proximo_metar]
                if (hora_msg > hora_proximo) or (hora_msg == hora_proximo and min_msg >= min_proximo):
                    metar_ativo = cronograma_metar[(hora_proximo, min_proximo)]
                    qnh_atual, temp_atual = processar_metar(metar_ativo)
                    
                    print(f"[CLOCK UPDATE] {hora_msg:02d}:{min_msg:02d} -> Applied new METAR: {metar_ativo}")
                    indice_proximo_metar += 1
            
            valida, x, y, dist_nm, alt_ft, trk_nr = mensagem_cumpre_requisitos_grafico(msg, qnh_atual, temp_atual)
            
            if dist_nm is not None and alt_ft is not None:
                if valida:
                    x_aceites.append(x)
                    y_aceites.append(y)
                    distancias_aceites.append(dist_nm)
                    altitudes_aceites.append(alt_ft)
                    
                    if trk_nr: # Se existir um Track Number, adiciona ao nosso conjunto
                        voos_aceites_unicos.add(trk_nr)
                else:
                    # Limitar apenas o que queremos mostrar visualmente (de -5 NM a 15 NM)
                    if -5 < dist_nm < 15: 
                        x_rejeitados.append(x)
                        y_rejeitados.append(y)
                        distancias_rejeitadas.append(dist_nm)
                        altitudes_rejeitadas.append(alt_ft)

    print("="*60)
    print(f"EXTRACTION RESULTS:")
    print(f"-> Total messages received: {total_mensagens}")
    print(f"-> Total accepted observations (messages): {len(distancias_aceites)}")
    print(f"-> Number of unique aircraft on Final Approach: {len(voos_aceites_unicos)}")
    print("="*60)
    
    # =========================================================
    # CONSTRUÇÃO DO PAINEL GRÁFICO DUPLO (INGLÊS BRITÂNICO)
    # =========================================================
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(x_rejeitados, y_rejeitados, color='lightcoral', s=2, alpha=0.5, label='Rejected Traffic')
    plt.scatter(x_aceites, y_aceites, color='green', s=10, alpha=0.8, label='Final Approach (Accepted)')
    plt.scatter([X_THR], [Y_THR], color='blue', marker='*', s=200, label='Runway 02 Threshold (THR)')
    plt.title("Top View (XY Plane) - Spatial Validation", fontweight='bold')
    plt.xlabel("Y Coordinate (metres)")
    plt.ylabel("X Coordinate (metres)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.axis('equal') 

    plt.subplot(1, 2, 2)
    plt.scatter(distancias_rejeitadas, altitudes_rejeitadas, color='lightcoral', s=2, alpha=0.5, label='Rejected Traffic')
    plt.scatter(distancias_aceites, altitudes_aceites, color='green', s=10, alpha=0.8, label='Accepted Traffic')
    
    plt.plot([11.3, 0], [4000, 330], color='blue', linewidth=2, label='Theoretical Glide Path (3º)')
    plt.axhline(y=4000, color='red', linestyle='--', label='Upper Limit (4000 ft)')
    plt.axhline(y=330, color='orange', linestyle='--', label='Runway Elevation (330 ft)')
    plt.axvline(x=11.3, color='purple', linestyle='--', label='Intermediate Fix (11.3 NM)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5) # Linha que marca a cabeceira
    
    # Ajustar os limites para mostrar claramente as distâncias negativas (aeronaves na pista)
    plt.xlim(15, -3) 
    plt.ylim(-500, 6000) 
    plt.title("Final Approach Vertical Profile (True Altitude)", fontweight='bold')
    plt.xlabel("Distance to Threshold (NM)")
    plt.ylabel("True Altitude (ft)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nome_do_ficheiro_csv = "dados_1julho_12h16h.csv" 
    nome_do_ficheiro_metar = "METAR_julho.txt"
    dia_alvo = "01"
    
    gerar_graficos_tese(nome_do_ficheiro_csv, nome_do_ficheiro_metar, dia_alvo)