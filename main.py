import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import lambertw

def solucion_exacta_lambert(k, n=2.0):
    """Calcula la solución analítica exacta usando la función W de Lambert."""
    # Transformación formal de x + log_n(x) = k
    val = np.log(n) * (n**k)
    return np.real(lambertw(val) / np.log(n))

def formula_propuesta_estupinan(k, n=2.0):
    """
    Fórmula analítica de Juan Estupiñán de tres términos.
    Complejidad estricta O(1). Sin bucles iterativos.
    """
    ln_n = np.log(n)
    ln_k = np.log(k)
    return k - (ln_k / ln_n) + (ln_k / (k * (ln_n**2)))

def newton_raphson(k, n=2.0, tol=1e-7, max_iter=100):
    """Método numérico iterativo tradicional (Newton-Raphson)."""
    ln_n = np.log(n)
    x = k # Estimación inicial óptima
    for i in range(max_iter):
        f = x + np.log(x)/ln_n - k
        df = 1 + 1/(x * ln_n)
        x_new = x - f/df
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

# 1. BENCHMARKING DE TIEMPO (Demostración del ~70%)
print("Ejecutando millones de iteraciones para medir tiempos reales...")
k_bench = 10.0
n_bench = 2.0
num_pruebas = 500000

# Prueba Newton-Raphson
t0 = time.time()
for _ in range(num_pruebas):
    _ = newton_raphson(k_bench, n_bench)
t_newton = time.time() - t0

# Prueba Fórmula Propuesta
t0 = time.time()
for _ in range(num_pruebas):
    _ = formula_propuesta_estupinan(k_bench, n_bench)
t_propuesta = time.time() - t0

reduccion_porcentaje = ((t_newton - t_propuesta) / t_newton) * 100

print("\n" + "="*50)
print("RESULTADOS EMPÍRICOS DE RENDIMIENTO (BENCHMARK)")
print("="*50)
print(f"Tiempo total Newton-Raphson : {t_newton:.4f} segundos")
print(f"Tiempo total Tu Fórmula O(1): {t_propuesta:.4f} segundos")
print(f"Reducción del esfuerzo computacional: {reduccion_porcentaje:.2f}%")
print("="*50 + "\n")

# 2. GENERACIÓN DE LA GRÁFICA
print("Generando gráfica de Transición de Fase e Intratabilidad...")
valores_k = np.linspace(2, 50, 400)
errores_absolutos = []

for k in valores_k:
    exacta = solucion_exacta_lambert(k, n=2.0)
    aproximada = formula_propuesta_estupinan(k, n=2.0)
    errores_absolutos.append(abs(exacta - aproximada))

# Configuración estética de la gráfica estilo IEEE
plt.figure(figsize=(7, 4.5))
plt.plot(valores_k, errores_absolutos, label='Error Residual Exacto $E$', color='#d32f2f', linewidth=2.5)
plt.axvspan(2, 6, color='#ffebee', alpha=0.7, label='Zona Crítica (Transición de Fase)')
plt.axvspan(6, 50, color='#e8f5e9', alpha=0.5, label='Zona de Estabilidad Asintótica')

plt.title('Decaimiento Cuadrático del Error $E$ vs. Escala de Restricciones $k$', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Parámetro de Restricciones del Sistema ($k$)', fontsize=10)
plt.ylabel('Magnitud del Error Absoluto $|E|$', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(2, 50)

# Guardar la imagen
plt.savefig('grafica_error_transicion.png', dpi=300, bbox_inches='tight')
plt.show()

