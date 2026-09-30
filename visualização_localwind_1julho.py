import csv
import math
import re
import matplotlib.pyplot as plt
import os

# ==========================================
# THESIS CONSTANTS (Secção 3.2.1)
# ==========================================
X_THR = -433.76     
Y_THR = 277.06      
M_PARA_NM = 1852.0  
FT_PARA_M = 0.3048  

# ==========================================
# STANDARD ATMOSPHERE (ISA) CONSTANTS
# ==========================================
P_ISA = 1013.25      
T_ISA = 288.15       
LAMBDA_ISA = -0.0065 
R_GAS = 287.05287    
G0 = 9.80665         

def carregar_cronograma_metar(caminho_metar, dia_alvo):
    """
    Lê o ficheiro METAR e cria um dicionário cronológico para o dia especificado.
    """
    cronograma = {}
    if not os.path.exists(caminho_metar):
        print(f"[WARNING] File {caminho_metar} not found. Using default weather.")
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
    qnh = float(match_qnh.group(1)) if match_qnh else P_ISA
    
    match_temp = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar_str)
    if match_temp:
        temp_str = match_temp.group(1)
        temp_celsius = float(temp_str.replace('M', '-'))
    else:
        temp_celsius = 15.0

    return qnh, temp_celsius

def extrair_hora_minuto_msg(msg):
    """
    Extrai a hora e o minuto do campo 'tod'.
    """
    tod = msg.get('tod', '')
    if not tod: return 0, 0
    if ':' in str(tod):
        partes = str(tod).split(':')
        return int(partes[0]), int(partes[1])
    try:
        segundos = float(tod)
        return int(segundos // 3600), int((segundos % 3600) // 60)
    except ValueError:
        return 0, 0

def mensagem_cumpre_requisitos_grafico(msg, qnh_local, temp_local_celsius):
    """
    Aplica os filtros espaciais e calcula a altitude verdadeira e a distância.
    """
    try:
        campos_obrigatorios = ['x', 'y', 'baroH', 'vx', 'vy']
        for campo in campos_obrigatorios:
            if not msg.get(campo) or str(msg[campo]).strip() == '':
                return False, None, None

        x, y = float(msg['x']), float(msg['y'])
        baroH = float(msg['baroH'])
        vx, vy = float(msg['vx']), float(msg['vy'])

        # Distância Euclidiana e Distância Projetada
        distancia_euclidiana_nm = math.sqrt((x - X_THR)**2 + (y - Y_THR)**2) / M_PARA_NM
        rad_205 = math.radians(205)
        distancia_projetada_nm = ((x - X_THR) * math.sin(rad_205) + (y - Y_THR) * math.cos(rad_205)) / M_PARA_NM

        # Cálculo da Altitude Verdadeira (Termodinâmica)
        h_p_ft = baroH * 100 
        h_p_m = h_p_ft * FT_PARA_M
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p = P_ISA * (1 + (LAMBDA_ISA * h_p_m) / T_ISA) ** expoente_p
        
        t_local_k = temp_local_celsius + 273.15
        expoente_h = -(LAMBDA_ISA * R_GAS) / G0
        h_true_m = (t_local_k / LAMBDA_ISA) * ((p / qnh_local) ** expoente_h - 1)
        h_real = h_true_m / FT_PARA_M

        # Filtros espaciais
        if not (0 < distancia_euclidiana_nm < 11.3): return False, distancia_projetada_nm, h_real
        bearing = math.degrees(math.atan2(x - X_THR, y - Y_THR))
        if bearing < 0: bearing += 360
        if not (200 <= bearing <= 210): return False, distancia_projetada_nm, h_real
        if not (330 < h_real < 4000): return False, distancia_projetada_nm, h_real
        chi_g = math.degrees(math.atan2(vx, vy))
        if chi_g < 0: chi_g += 360
        if not (20 < chi_g < 30): return False, distancia_projetada_nm, h_real

        return True, distancia_projetada_nm, h_real

    except ValueError:
        return False, None, None

def calcular_vento_local(msg, temp_local_celsius):
    """
    Calcula o headwind aproximado usando a True Airspeed e a Ground Speed.
    Retorna o valor em nós (kt).
    """
    try:
        if not msg.get('iar') or str(msg['iar']).strip() == '':
            return None 

        iar_kt = float(msg['iar'])
        vx = float(msg['vx'])
        vy = float(msg['vy'])
        baroH = float(msg['baroH'])

        # 1. Vg (Ground Speed)
        vg_ms = math.sqrt(vx**2 + vy**2)
        vg_kt = vg_ms * (3600 / M_PARA_NM) 

        # 2. Pressão e Densidade Local
        p0_pa = P_ISA * 100 
        hp_m = baroH * 100 * FT_PARA_M 
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p_pa = p0_pa * (1 + (LAMBDA_ISA * hp_m) / T_ISA) ** expoente_p

        t_local_k = temp_local_celsius + 273.15
        rho_local = p_pa / (R_GAS * t_local_k)
        rho_0 = p0_pa / (R_GAS * T_ISA)

        # 3. True Airspeed e Headwind
        vtas_kt = iar_kt * math.sqrt(rho_0 / rho_local)
        headwind_kt = vtas_kt - vg_kt

        return headwind_kt

    except ValueError:
        return None

def gerar_graficos_vento(caminho_csv, caminho_metar, dia_alvo):
    """
    Lê o ficheiro CSV, calcula o vento para mensagens válidas e plota os gráficos de análise.
    """
    cronograma_metar = carregar_cronograma_metar(caminho_metar, dia_alvo)
    chaves_tempo = list(cronograma_metar.keys())
    
    qnh_atual, temp_atual = P_ISA, 15.0
    if chaves_tempo:
        qnh_atual, temp_atual = processar_metar(cronograma_metar[chaves_tempo[0]])
    
    # Listas para guardar dados que vão ser plotados
    distancias_vento = []
    altitudes_vento = []
    valores_vento = []
    
    indice_proximo_metar = 1 if len(chaves_tempo) > 1 else 0

    print("="*60)
    print("EXTRACTING WIND OBSERVATIONS FOR VISUALISATION...")
    print("="*60)

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        
        for msg in leitor_csv:
            hora_msg, min_msg = extrair_hora_minuto_msg(msg)
            
            # Atualização dinâmica do tempo e temperatura do METAR
            if indice_proximo_metar < len(chaves_tempo):
                hora_proximo, min_proximo = chaves_tempo[indice_proximo_metar]
                if (hora_msg > hora_proximo) or (hora_msg == hora_proximo and min_msg >= min_proximo):
                    metar_ativo = cronograma_metar[(hora_proximo, min_proximo)]
                    qnh_atual, temp_atual = processar_metar(metar_ativo)
                    indice_proximo_metar += 1
            
            # Validar apenas se está no percurso (0 a 11.3 NM, altitude correta, etc.)
            valida, dist_nm, alt_ft = mensagem_cumpre_requisitos_grafico(msg, qnh_atual, temp_atual)
            
            if valida:
                vento_kt = calcular_vento_local(msg, temp_atual)
                if vento_kt is not None:
                    distancias_vento.append(dist_nm)
                    altitudes_vento.append(alt_ft)
                    valores_vento.append(vento_kt)

    print(f"Total valid wind observations extracted: {len(valores_vento)}")
    
    # =========================================================
    # CONSTRUÇÃO DO PAINEL GRÁFICO DUPLO (INGLÊS BRITÂNICO)
    # =========================================================
    plt.figure(figsize=(16, 7))

    # Gráfico 1: Headwind vs. Distância à Cabeceira
    plt.subplot(1, 2, 1)
    plt.scatter(distancias_vento, valores_vento, color='dodgerblue', s=8, alpha=0.6, label='Observed Headwind')
    plt.axvline(x=11.3, color='purple', linestyle='--', label='Intermediate Fix (11.3 NM)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Runway Threshold')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3) # Linha de vento zero
    
    plt.xlim(12, -1) # Invertido para simular a aproximação
    plt.title("Headwind Profile along Final Approach", fontweight='bold')
    plt.xlabel("Distance to Threshold (NM)")
    plt.ylabel("Calculated Headwind (kt)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Gráfico 2: Altitude vs. Headwind (Perfil Vertical)
    plt.subplot(1, 2, 2)
    plt.scatter(valores_vento, altitudes_vento, color='mediumseagreen', s=8, alpha=0.6, label='Observed Headwind')
    plt.axhline(y=4000, color='purple', linestyle='--', label='Upper Limit (4000 ft)')
    plt.axhline(y=330, color='orange', linestyle='--', label='Runway Elevation (330 ft)')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
    
    plt.ylim(0, 4500)
    plt.title("Vertical Headwind Profile", fontweight='bold')
    plt.xlabel("Calculated Headwind (kt)")
    plt.ylabel("True Altitude (ft)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nome_do_ficheiro_csv = "dados_1julho_12h16h.csv" 
    nome_do_ficheiro_metar = "METAR_julho.txt"
    dia_alvo = "01"
    
    gerar_graficos_vento(nome_do_ficheiro_csv, nome_do_ficheiro_metar, dia_alvo)