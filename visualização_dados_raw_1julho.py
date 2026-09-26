import csv
import math
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTES BÁSICAS
# ==========================================
X_THR = -433.76  # Coordenada X da cabeceira 02 (m)
Y_THR = 277.06   # Coordenada Y da cabeceira 02 (m)
M_PARA_NM = 1852.0  # Fator de conversão (metros para NM)

# Parâmetros da Direção de Aproximação (Baseado na Tese)
# A pista 02 tem direção 25º, logo a aproximação vem da direção 205º.
ANGULO_APROXIMACAO_GRAUS = 205
ANGULO_APROX_RAD = math.radians(ANGULO_APROXIMACAO_GRAUS)

# Componentes do vetor unitário da direção de aproximação
# Usamos sin para X e cos para Y assumindo que 0º é o Norte (Eixo Y)
VETOR_DIR_X = math.sin(ANGULO_APROX_RAD)
VETOR_DIR_Y = math.cos(ANGULO_APROX_RAD)

def explorar_dados_e_contar(caminho_csv):
    """
    Lê os dados em bruto, conta o número total de mensagens e aviões,
    e separa a visualização em duas etapas (2D e depois 3D).
    Calcula a distância projetada para diferenciar tráfego em aproximação do tráfego divergente.
    """
    x_coords = []
    y_coords = []
    altitudes_ft = []
    distancias_nm = []

    # Variáveis para a nossa funcionalidade de contagem
    total_mensagens = 0
    track_numbers = set()  # O set() garante que não há aviões duplicados

    print("A ler os dados em bruto do ficheiro CSV...")

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        
        for msg in leitor_csv:
            # 1. Contar a mensagem lida
            total_mensagens += 1
            
            # 2. Detetar o avião (track number)
            trk_nr = msg.get('trkNr')
            if trk_nr and trk_nr.strip() != '':
                track_numbers.add(trk_nr.strip())

            # 3. Extrair coordenadas para os gráficos (ignorando mensagens sem posição)
            if msg.get('x') and msg.get('y') and msg.get('baroH'):
                try:
                    x = float(msg['x'])
                    y = float(msg['y'])
                    baroH = float(msg['baroH'])
                    
                    # Conversão direta do Flight Level para pés (feet)
                    alt_ft = baroH * 100
                    
                    # ----------------------------------------------------
                    # NOVO CÁLCULO DE DISTÂNCIA COM SINAL (PROJEÇÃO VETORIAL)
                    # ----------------------------------------------------
                    # Posição relativa da aeronave à cabeceira
                    dx = x - X_THR
                    dy = y - Y_THR
                    
                    # Produto escalar: projeta a posição no vetor da final (205º)
                    dist_m_com_sinal = (dx * VETOR_DIR_X) + (dy * VETOR_DIR_Y)
                    
                    # Conversão para Milhas Náuticas (NM)
                    dist_nm = dist_m_com_sinal / M_PARA_NM
                    
                    x_coords.append(x)
                    y_coords.append(y)
                    altitudes_ft.append(alt_ft)
                    distancias_nm.append(dist_nm)
                    
                except ValueError:
                    # Ignorar linhas com valores numéricos corrompidos
                    continue

    # =========================================================
    # IMPRESSÃO DOS RESULTADOS DE CONTAGEM
    # =========================================================
    print("-" * 50)
    print("RESUMO DOS DADOS LIDOS:")
    print(f"-> Total de mensagens processadas: {total_mensagens}")
    print(f"-> Total de aviões detetados (track numbers únicos): {len(track_numbers)}")
    print(f"-> Observações espaciais válidas extraídas: {len(x_coords)}")
    print("-" * 50)

    if not x_coords:
        print("Aviso: Não foram encontrados dados válidos para desenhar os gráficos.")
        return

    # =========================================================
    # ETAPA 1: JANELA DE GRÁFICOS 2D
    # =========================================================
    print("A abrir a janela com os gráficos 2D. Fecha a janela para avançar para o gráfico 3D...")
    
    plt.figure(figsize=(14, 6))

    # Gráfico 1.1: Vista de Topo (XY Plane)
    plt.subplot(1, 2, 1)
    plt.scatter(x_coords, y_coords, color='grey', s=2, alpha=0.3, label='Raw Traffic')
    plt.scatter([X_THR], [Y_THR], color='blue', marker='*', s=150, label='THR 02')
    plt.title("Top View (XY Plane) - Raw Data", fontweight='bold')
    plt.xlabel("Y Coordinate (metres)")
    plt.ylabel("X Coordinate (metres)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.axis('equal')

    # Gráfico 1.2: Perfil Vertical
    plt.subplot(1, 2, 2)
    plt.scatter(distancias_nm, altitudes_ft, color='grey', s=2, alpha=0.3, label='Raw Traffic')
    plt.plot([11.3, 0], [4000, 330], color='blue', linewidth=2, label='Theoretical Glide Path (3º)')
    
    # Ajuste dinâmico do eixo X para suportar distâncias negativas
    max_dist = max(distancias_nm) if distancias_nm else 20
    min_dist = min(distancias_nm) if distancias_nm else -5
    
    # Invertemos os limites (maior para menor) para simular a aproximação da esquerda para a direita.
    # Adicionamos uma margem de 1 NM em ambos os lados para melhor visualização.
    plt.xlim(max_dist + 1, min_dist - 1)
    
    plt.title("Vertical Profile - Raw Barometric Altitude", fontweight='bold')
    plt.xlabel("Distance to Threshold (NM)")
    plt.ylabel("Barometric Altitude (ft)")
    
    # Adicionar uma linha vertical no 0 para marcar claramente a cabeceira
    plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='THR 02')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()  # Pausa a execução até fechares a janela 2D

    # =========================================================
    # ETAPA 2: JANELA DE GRÁFICO 3D
    # =========================================================
    print("A abrir a janela com a visualização espacial 3D...")
    
    fig_3d = plt.figure(figsize=(9, 7))
    ax3 = fig_3d.add_subplot(1, 1, 1, projection='3d')
    
    ax3.scatter(x_coords, y_coords, altitudes_ft, color='purple', s=2, alpha=0.2, label='Raw Traffic')
    ax3.scatter([X_THR], [Y_THR], [330], color='blue', marker='*', s=150, label='THR 02')
    
    ax3.set_title("3D Trajectory Visualisation", fontweight='bold')
    ax3.set_xlabel("Y Coordinate (metres)")
    ax3.set_ylabel("X Coordinate (metres)")
    ax3.set_zlabel("Altitude (ft)")
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nome_do_ficheiro = "dados_1julho_12h16h.csv" 
    explorar_dados_e_contar(nome_do_ficheiro)