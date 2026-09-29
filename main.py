import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import lambertw

def solucion_exacta_friis(kappa, sigma=1.0):
    """
    Calcula la potencia de transmision exacta Pt utilizando la funcion W de Lambert
    basado en el colapso del presupuesto de enlace fisico: Pt + sigma * ln(Pt) = kappa.
    """
    # Transformacion algebraica para la entrada de Lambert
    val = (1.0 / sigma) * np.exp(kappa / sigma)
    W = np.real(lambertw(val))
    return sigma * W

def solucion_analitica_estupinan(kappa, sigma=1.0):
    """
    Solucion analitica de Juan Estupinan aplicada al modelo fisico de Friis.
    Complejidad estricta O(1) sin bucles iterativos.
    """
    ratio = kappa / sigma
    ln_ratio = np.log(ratio)
    return sigma * (ratio - ln_ratio + (ln_ratio / ratio))

def newton_raphson_friis(kappa, sigma=1.0, tol=1e-7, max_iter=100):
    """Metodo numerico iterativo tradicional de Newton-Raphson para el modelo de Friis."""
    # Estimacion inicial basada en el componente dominante
    Pt = kappa / sigma
    if Pt <= 0:
        Pt = 1.0
        
    for i in range(max_iter):
        f = Pt + sigma * np.log(Pt) - kappa
        df = 1.0 + (sigma / Pt)
        Pt_new = Pt - f / df
        if abs(Pt_new - Pt) < tol:
            return Pt_new, i + 1
        Pt = Pt_new
    return Pt, max_iter


# 1. ESCENARIO FISICO: PARAMETROS DE TELEMETRIA SATELITAL (IEEE/NASA)

# Distancia intersatelital d = 50 km (red enjambre LEO)
d = 50000.0 
# Longitud de onda del laser lambda = 1550 nm (Laser infrarrojo estandar ISL)
lambd = 1550e-9 
# Ganancias de antenas opticas acopladas (en decibelios)
Gt = 30.0 
Gr = 30.0 
# Potencia de recepcion objetivo en el nodo sensor
Prx = -40.0 
# Coeficiente de atenuacion estocastica por ruido de fase
sigma_canal = 1.5

# Calculo del presupuesto de enlace neto (Atenuacion de Friis constante)
pérdida_espacio_libre = 20.0 * np.log10(lambd / (4.0 * np.pi * d))
kappa_neto = Prx - Gt - Gr - pérdida_espacio_libre


# 2. BENCHMARKING DE RENDIMIENTO COMPUTACIONAL
print("Ejecutando ciclos de procesamiento masivo para el modelo de Friis...")
num_pruebas = 500000

# Medicion de tiempo: Newton-Raphson
t0 = time.time()
for _ in range(num_pruebas):
    _ = newton_raphson_friis(kappa_neto, sigma_canal)
t_newton = time.time() - t0

# Medicion de tiempo: Solucion propuesta
t0 = time.time()
for _ in range(num_pruebas):
    _ = solucion_analitica_estupinan(kappa_neto, sigma_canal)
t_propuesta = time.time() - t0

reduccion_porcentaje = ((t_newton - t_propuesta) / t_newton) * 100

print("\n" + "="*60)
print("RESULTADOS EMPIRICOS DE RENDIMIENTO (MODELO DE FRIIS-ESTUPINAN)")
print("="*60)
print(f"Tiempo total Newton-Raphson       : {t_newton:.4f} segundos")
print(f"Tiempo total Solucion propuesta   : {t_propuesta:.4f} segundos")
print(f"Porcentaje de reduccion de tiempo : {reduccion_porcentaje:.2f}%")
print("="*60 + "\n")


# 3. GENERACION DE LA GRAFICA CIENTIFICA PARA EL PAPER
print("Generando grafica de convergencia fisica y transicion de fase...")
valores_kappa = np.linspace(5, 100, 500)
errores_absolutos = []

for k in valores_kappa:
    exacta = solucion_exacta_friis(k, sigma_canal)
    aproximada = solucion_analitica_estupinan(k, sigma_canal)
    errores_absolutos.append(abs(exacta - aproximada))

# Configuracion de estilo bajo estandar editorial IEEE
plt.figure(figsize=(7, 4.5))
plt.plot(valores_kappa, errores_absolutos, label='Error Residual Exacto E', color='#d32f2f', linewidth=2.5)
plt.axvspan(5, 15, color='#ffebee', alpha=0.7, label='Frontera Critica (Transicion de Fase)')
plt.axvspan(15, 100, color='#e8f5e9', alpha=0.5, label='Estabilidad Asintotica del Enlace')

plt.title('Decaimiento del Error Absoluto E vs. Presupuesto de Enlace Kappa', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Parametro del Presupuesto de Enlace Neto (Kappa)', fontsize=10)
plt.ylabel('Magnitud del Error Absoluto |E| (Vatios)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(5, 100)

# Almacenamiento local de la figura en alta definicion
plt.savefig('grafica_friis_estupinan.png', dpi=300, bbox_inches='tight')
plt.show()
print("Grafica guardada exitosamente como 'grafica_friis_estupinan.png' en los archivos de Colab.")
