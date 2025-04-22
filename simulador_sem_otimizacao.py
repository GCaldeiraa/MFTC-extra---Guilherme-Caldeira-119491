import numpy as np
from scipy.optimize import fsolve

# Parâmetros do sistema
area_deposito = 185  # m²
altura_inicial = 4  # m
altura_max = 7  # m
altura_min = 2  # m
tempo_total = 24  # horas
dist_bomba_dep = 150  # m
altura_atual = altura_inicial  # m (altura atual da água no reservatório)
tarifa_energia = np.array([
    # 00h-02h (0h e 1h)
    0.0713, 0.0713,  
    # 02h-04h (2h e 3h)
    0.0651, 0.0651,
    # 04h-06h (4h e 5h)
    0.0593, 0.0593,
    # 06h-08h (6h e 7h)
    0.0778, 0.0778,
    # 08h-10h (8h e 9h)
    0.0851, 0.0851,
    # 10h-12h (10h e 11h)
    0.0923, 0.0923,
    # 12h-14h (12h e 13h)
    0.0968, 0.0968,
    # 14h-16h (14h e 15h)
    0.10094, 0.10094,
    # 16h-18h (16h e 17h)
    0.10132, 0.10132,
    # 18h-20h (18h e 19h)
    0.10230, 0.10230,
    # 20h-22h (20h e 21h)
    0.10189, 0.10189,
    # 22h-24h (22h e 23h)
    0.10132, 0.10132
])

# Constante para cálculo de perdas de carga
coef_perdas = (32 * 0.02) / (0.3**5 * 9.81 * np.pi**2)  #falta multiplicar pelo comprimento da tubulação (em m)

# Funções dos caudais
def calcular_Q_r(t):
    return (-0.004 * t**3 + 0.09 * t**2 + 0.1335 * t + 20)/3600  # Convertendo de m³/h para m³/s

def calcular_Q_vc_max(t):
    return (-1.19333e-7 * t**7 - 4.90754e-5 * t**6 + 3.733e-3 * t**5 - 0.09621 * t**4 + 1.03965 * t**3 - 3.8645 * t**2 - 1.0124 * t + 75.393)/3600  # Convertendo de m³/h para m³/s

# Função para calcular o Qpump
def calcular_Q_pump(Q_r,altura_atual, dist_bomba_dep):
    # Função para resolver numericamente
    def equation(Qp):
        h_pump = 260 - 0.002 * Qp**2*3600**2  # Altura de instalação da bomba (em m, em que Qp está em m³/s)
        perdas_totais = coef_perdas*5000*(Qp - Q_r)**2 + coef_perdas*2500*Q_r**2  # Perdas nas tubulações
        h_required = altura_atual + dist_bomba_dep + perdas_totais
        return h_pump - h_required
    
    # Resolver a equação numericamente
    Qp_initial_guess = 1 #palpite inicial
    Qp_solution = fsolve(equation, Qp_initial_guess)[0]
    return Qp_solution

def calcular_custo(Q_pump, preço_por_hora):
    eficiencia=0.65 # Eficiência da bomba
    h_pump = 260 - 0.002 * Q_pump**2*3600**2 # Altura de instalação da bomba
    densidade = 1000 # kg/m³ (densidade da água)
    aceleracao_gravidade = 9.81 # m/s² (aceleração da gravidade)
    custo_bomba = (Q_pump *densidade*aceleracao_gravidade*h_pump)/(eficiencia*1000)*preço_por_hora # Exemplo de cálculo de custo. Foi dividido por 1000 para passar potencia para kw
    return custo_bomba

# Simulação
horas = np.arange(0, tempo_total)
resultados_Q_r = []
resultados_Q_vc_max = []
resultados_Q_pump = []
resultados_h_atual = [altura_inicial]  # Inicializa com a altura inicial
preco_total=0

for t in horas:
    Q_r = calcular_Q_r(t)
    Q_vc_max = calcular_Q_vc_max(t)
    Q_pump = calcular_Q_pump(Q_r,resultados_h_atual[t], dist_bomba_dep)
    resultados_Q_r.append(Q_r)
    resultados_Q_vc_max.append(Q_vc_max)
    resultados_Q_pump.append(Q_pump)
    
    # Calcula a nova altura
    delta_h = ((Q_pump - Q_r - Q_vc_max) / area_deposito)*3600 ##delta_h em m/h
    nova_altura = resultados_h_atual[t] + delta_h
    resultados_h_atual.append(nova_altura)
    preco_total += calcular_custo(Q_pump, tarifa_energia[t])   

print("\nResultados completos:")
print("Hora | Q_r (m³/s) | Q_vc_max (m³/s) | Q_pump (m³/s) | altura_atual (m)")
print("-----|------------|-----------------|---------------|------------------")

for t in horas:
    print(f"{t+1:4} | {resultados_Q_r[t]:10.6f} | {resultados_Q_vc_max[t]:15.6f} | {resultados_Q_pump[t]:13.6f} | {resultados_h_atual[t+1]:16.4f}")

# Verificação final
print(f"\nAltura final do reservatório: {resultados_h_atual[-1]:.4f} m")
print(f"Preço total da energia: {preco_total:.2f} €")
if resultados_h_atual[-1] > altura_max:
    print("AVISO: Altura excedeu o máximo permitido!")
elif resultados_h_atual[-1] < altura_min:
    print("AVISO: Altura ficou abaixo do mínimo permitido!")