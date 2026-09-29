import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import lambertw

def solucion_exacta_friis(kappa, sigma=1.0):
    """Calcula la potencia de transmision exacta Pt utilizando la funcion W de Lambert."""
    val = (1.0 / sigma) * np.exp(kappa / sigma)
    W = np.real(lambertw(val))
    return sigma * W

def solucion_analitica_estupinan(kappa, sigma=1.0):
    """Solucion analitica de Juan Estupinan con el factor cuadratico de acoplamiento."""
    ratio = kappa / sigma
    ln_ratio = np.log(ratio)
    return sigma * (ratio - ln_ratio + (ln_ratio / (ratio * np.log(sigma + 1.0 if sigma > 1 else 2.0))))

def newton_raphson_friis(kappa, sigma=1.0, tol=1e-7, max_iter=100):
    """Metodo numerico iterativo tradicional de Newton-Raphson para el modelo de Friis."""
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


# 1. PARAMETROS DE TELEMETRIA SATELITAL Y BENCHMARKING DE RENDIMIENTO

d = 50000.0 
lambd = 1550e-9 
Gt, Gr = 30.0, 30.0 
Prx = -40.0 
sigma_canal = 1.5


pérdida_espacio_libre = 20.0 * np.log10(lambd / (4.0 * np.pi * d))
kappa_neto = Prx - Gt - Gr - pérdida_espacio_libre

print("Ejecutando ciclos de procesamiento masivo para el modelo de Friis...")
num_pruebas = 500000

t0 = time.time()
for _ in range(num_pruebas):
    _ = newton_raphson_friis(kappa_neto, sigma_canal)
t_newton = time.time() - t0

t0 = time.time()
for _ in range(num_pruebas):
    _ = solucion_analitica_estupinan(kappa_neto, sigma_canal)
t_propuesta = time.time() - t0

reduccion_porcentaje = ((t_newton - t_propuesta) / t_newton) * 100

print("\n" + "="*60)
print("RESULTADOS EMPIRICOS DE RENDIMIENTO (MODELO DE FRIIS-ESTUPINAN)")
print("="*60)
print(f"Tiempo total Newton-Raphson       : {t_newton:.4f} segundos")
print(f"Tiempo total Solucion Estupinan   : {t_propuesta:.4f} segundos")
print(f"Porcentaje de reduccion de tiempo : {reduccion_porcentaje:.2f}%")
print("="*60 + "\n")


# 2. GENERACION DE FIGURAS ANALITICAS Y DE HARDWARE
valores_kappa = np.linspace(15, 150, 500)
errores_absolutos = []
errores_relativos = []

for k in valores_kappa:
    exacta = solucion_exacta_friis(k, sigma_canal)
    aproximada = solucion_analitica_estupinan(k, sigma_canal)
    errores_absolutos.append(abs(exacta - aproximada))
    errores_relativos.append(abs(exacta - aproximada) / exacta)

# FIGURA A: Grafica fisica de estabilizacion (Hardware)
plt.figure(figsize=(7, 4.5))
plt.plot(valores_kappa, errores_absolutos, label='Error Residual Optimizacion |E|', color='#d32f2f', linewidth=2.5)
plt.axvspan(15, 40, color='#ffebee', alpha=0.7, label='Frontera Critica (Alta Incerteza)')
plt.axvspan(40, 150, color='#e8f5e9', alpha=0.5, label='Zona de Estabilidad Asintotica O(1/k²)')
plt.title('Estabilizacion del Error Absoluto E vs. Presupuesto de Enlace Kappa', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Presupuesto de Enlace Intersatelital Neto (Kappa)', fontsize=10)
plt.ylabel('Magnitud del Error Absoluto |E| (Vatios)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(15, 150)
plt.savefig('grafica_friis_estupinan.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\n" * 2) 
# FIGURA B: Grafica analitica pura (Teoria Matematica)
plt.figure(figsize=(7, 4.5))
plt.plot(valores_kappa, errores_relativos, label='Error Relativo Teorico $E_r$', color='#1565c0', linewidth=2.5)
plt.axvspan(15, 40, color='#e3f2fd', alpha=0.7, label='Zona Critica de Transicion')
plt.axvspan(40, 150, color='#e8f5e9', alpha=0.5, label='Regimen de Convergencia $O(1/\\kappa^2)$')
plt.title('Decaimiento Asintotico del Error Relativo Normalizado', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Parametro del Presupuesto de Enlace Neto (Kappa)', fontsize=10)
plt.ylabel('Magnitud del Error Relativo |E_r|', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(15, 150)
plt.savefig('grafica_decaimiento_puro.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Procesamiento completo. Graficas exportadas exitosamente.")
