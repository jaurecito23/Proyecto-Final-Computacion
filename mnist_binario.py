"""
Proyecto Final Computación — Red Neuronal para Dígitos (MNIST)
Versión Inicial: Clasificación binaria entre dos dígitos (0 y 1) usando Regresión Logística.
"""

import os

# Configuramos carpetas locales dentro del proyecto para caché y datos
CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(CARPETA_PROYECTO, ".mpl_cache")
RUTA_DATOS = os.path.join(CARPETA_PROYECTO, "datos_mnist")

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("=== Paso 1: Entorno verificado correctamente ===")
print(f"NumPy versión:        {np.__version__}")
print(f"scikit-learn versión: {sklearn.__version__}")
print(f"Matplotlib versión:   {plt.matplotlib.__version__}\n")

# =====================================================================
# PASO 2: Cargar, filtrar, normalizar y explorar los datos de MNIST
# =====================================================================
print("=== Paso 2: Cargando y preparando datos de MNIST ===")
print("Descargando/cargando MNIST (puede tardar unos segundos la primera vez)...")

# 1. Descargamos las 70.000 imágenes de 28x28 = 784 píxeles (y guardamos caché binario rápido)
os.makedirs(RUTA_DATOS, exist_ok=True)
ARCHIVO_CACHE_NPZ = os.path.join(RUTA_DATOS, "mnist_784_cache.npz")

if os.path.exists(ARCHIVO_CACHE_NPZ):
    datos_cache = np.load(ARCHIVO_CACHE_NPZ, allow_pickle=True)
    X_total = datos_cache["X"]
    y_total = datos_cache["y"]
else:
    X_raw, y_raw = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
        data_home=RUTA_DATOS,
    )
    X_total = X_raw.astype(np.uint8)
    y_total = y_raw.astype(str)
    np.savez_compressed(ARCHIVO_CACHE_NPZ, X=X_total, y=y_total)
print(f"Dataset MNIST completo: X_total = {X_total.shape}, y_total = {y_total.shape}")

# 2. Filtramos únicamente los dígitos '0' y '1' (clasificación binaria)
mascara = (y_total == "0") | (y_total == "1")
X = X_total[mascara]
y = y_total[mascara].astype(int)  # Convertimos '0' y '1' de texto a enteros 0 y 1

print(f"Datos filtrados (solo 0 y 1): X = {X.shape}, y = {y.shape}")
print(f"Cantidad de '0's: {np.sum(y == 0)} | Cantidad de '1's: {np.sum(y == 1)}")

# 3. Normalizamos los píxeles del rango [0, 255] al rango [0.0, 1.0]
print(f"Rango de píxeles ANTES de normalizar: [{X.min()}, {X.max()}]")
X = X / 255.0
print(f"Rango de píxeles DESPUÉS de normalizar: [{X.min():.1f}, {X.max():.1f}]")

# 4. Graficamos las primeras 8 imágenes para ver cómo son los datos
fig, ejes = plt.subplots(1, 8, figsize=(12, 2.2))
fig.suptitle("Paso 2: Ejemplos reales de MNIST filtrados (28x28 píxeles)", fontsize=12, fontweight="bold")

for i, ax in enumerate(ejes):
    # Reconstruimos la fila de 784 números en una cuadrícula de 28x28
    imagen_28x28 = X[i].reshape(28, 28)
    ax.imshow(imagen_28x28, cmap="gray")
    ax.set_title(f"Etiqueta: {y[i]}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
ruta_grafico_ejemplos = os.path.join(CARPETA_PROYECTO, "paso2_ejemplos_mnist.png")
plt.savefig(ruta_grafico_ejemplos, dpi=150)
plt.close()
print(f"Gráfico de ejemplos guardado en: {ruta_grafico_ejemplos}\n")

# =====================================================================
# PASO 3: Separar en conjunto de Entrenamiento (80%) y Prueba (20%)
# =====================================================================
print("=== Paso 3: Separando datos en Entrenamiento (Train) y Prueba (Test) ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Imágenes de ENTRENAMIENTO (80%): X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"  -> Cantidad de '0's en Train: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
print(f"  -> Cantidad de '1's en Train: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")

print(f"Imágenes de PRUEBA / TEST (20%): X_test  = {X_test.shape},  y_test  = {y_test.shape}")
print(f"  -> Cantidad de '0's en Test:  {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
print(f"  -> Cantidad de '1's en Test:  {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)\n")

# =====================================================================
# PASO 4: Entrenar la Regresión Logística e inspeccionar sus partes clave
# =====================================================================
print("=== Paso 4: Entrenando la Regresión Logística (1 Neurona) ===")

# 1. Definimos la neurona con Descenso por Gradiente y función de pérdida Log-Loss (Sigmoide)
neurona = SGDClassifier(
    loss="log_loss",          # Regresión logística (usa sigmoide y entropía cruzada)
    learning_rate="constant", # Tasa de aprendizaje constante
    eta0=0.01,                # Valor de alfa (tamaño del paso = 0.01)
    max_iter=100,             # Máximo de épocas de entrenamiento
    random_state=42,
)

# 2. Entrenamos el modelo con las 11.824 imágenes de entrenamiento
neurona.fit(X_train, y_train)

# 3. Extraemos los 784 pesos (w) y el sesgo (b) que aprendió la neurona
pesos_w = neurona.coef_[0]
sesgo_b = neurona.intercept_[0]

print(f"Cantidad de pesos aprendidos (w): {pesos_w.shape[0]} (uno por cada píxel de 28x28)")
print(f"  -> Peso más negativo (empuja hacia el '0'): {pesos_w.min():.4f}")
print(f"  -> Peso más positivo (empuja hacia el '1'): {pesos_w.max():.4f}")
print(f"  -> Sesgo aprendido (b):                     {sesgo_b:.4f}\n")

# 4. Demostración en vivo: cuenta matemática manual vs. scikit-learn en una imagen de prueba
idx_ejemplo = 0
imagen_ejemplo = X_test[idx_ejemplo]
etiqueta_real = y_test[idx_ejemplo]

# Paso A: Producto punto z = (w_1*x_1 + ... + w_784*x_784) + b
z_manual = np.dot(pesos_w, imagen_ejemplo) + sesgo_b

# Paso B: Función sigmoide que aplasta entre 0 y 1 -> sigma(z) = 1 / (1 + e^(-z))
prob_manual = 1.0 / (1.0 + np.exp(-z_manual))

# Paso C: Lo que calcula scikit-learn internamente con predict_proba()
prob_sklearn = neurona.predict_proba(imagen_ejemplo.reshape(1, -1))[0, 1]
prediccion_final = neurona.predict(imagen_ejemplo.reshape(1, -1))[0]

print("--- Verificación de la matemática dentro de la neurona (Imagen de Test #0) ---")
print(f"Etiqueta real de la imagen:                y = {etiqueta_real}")
print(f"Suma ponderada (producto punto + sesgo):   z = {z_manual:.4f}")
print(f"Probabilidad calculada a mano sigma(z):    {prob_manual:.8f}")
print(f"Probabilidad devuelta por scikit-learn:    {prob_sklearn:.8f}")
print(f"Decisión final de la neurona (>= 0.5 -> 1): {prediccion_final}\n")

# =====================================================================
# PASO 5: Descenso por Gradiente explícito en NumPy (Caja Transparente)
# =====================================================================
print("=== Paso 5: Entrenando nuestra propia neurona paso a paso con NumPy ===")

# 1. Inicializamos los 784 pesos y el sesgo en 0 (la neurona empieza sin saber nada)
w_propio = np.zeros(X_train.shape[1])
b_propio = 0.0
alfa = 0.5          # Tasa de aprendizaje (learning rate)
epocas = 100        # Cantidad de iteraciones de descenso por gradiente
N_train = len(y_train)
historial_error = []

for epoca in range(epocas):
    # A. Suma ponderada z = X * w + b y función sigmoide sigma(z)
    z = np.dot(X_train, w_propio) + b_propio
    y_hat = 1.0 / (1.0 + np.exp(-z))

    # B. Función de pérdida (Entropía Cruzada Binaria)
    eps = 1e-15
    error_epoca = -np.mean(
        y_train * np.log(y_hat + eps) + (1 - y_train) * np.log(1 - y_hat + eps)
    )
    historial_error.append(error_epoca)

    # C. Gradientes por la Regla de la Cadena: dL/dw = (y_hat - y)*x , dL/db = (y_hat - y)
    grad_w = np.dot(X_train.T, (y_hat - y_train)) / N_train
    grad_b = np.mean(y_hat - y_train)

    # D. Paso de actualización del Descenso por Gradiente
    w_propio = w_propio - alfa * grad_w
    b_propio = b_propio - alfa * grad_b

    if epoca in [0, 1, 5, 10, 25, 50, 99]:
        aciertos_epoca = np.mean((y_hat >= 0.5).astype(int) == y_train) * 100
        print(f"  Época {epoca:2d} | Error (Log-Loss): {error_epoca:.4f} | Exactitud en Train: {aciertos_epoca:.2f}%")
print()

# =====================================================================
# PASO 6: Verificación del Modelo y Generación de Gráficos Finales
# =====================================================================
print("=== Paso 6: Verificación Científica en el Conjunto de Prueba (Test) ===")

# 1. Exactitud (Accuracy) en datos de Prueba (2.956 imágenes nunca vistas)
y_pred_sklearn = neurona.predict(X_test)
exactitud_sklearn = accuracy_score(y_test, y_pred_sklearn) * 100

z_test_propio = np.dot(X_test, w_propio) + b_propio
y_prob_propio = 1.0 / (1.0 + np.exp(-z_test_propio))
y_pred_propio = (y_prob_propio >= 0.5).astype(int)
exactitud_propio = accuracy_score(y_test, y_pred_propio) * 100

print(f"1. Comparación con valores publicados (>99% esperado para '0' vs '1'):")
print(f"   -> Exactitud en Test (scikit-learn SGDClassifier): {exactitud_sklearn:.2f}%")
print(f"   -> Exactitud en Test (Nuestra neurona en NumPy):   {exactitud_propio:.2f}%\n")

# 2. Matriz de Confusión detallada
matriz_conf = confusion_matrix(y_test, y_pred_sklearn)
ceros_correctos = matriz_conf[0, 0]
ceros_confundidos = matriz_conf[0, 1]
unos_confundidos = matriz_conf[1, 0]
unos_correctos = matriz_conf[1, 1]

print("2. Desglose de aciertos y errores en las 2.956 imágenes de Test:")
print(f"   -> Dígitos '0' bien clasificados: {ceros_correctos} de {ceros_correctos + ceros_confundidos} (errores: {ceros_confundidos})")
print(f"   -> Dígitos '1' bien clasificados: {unos_correctos} de {unos_correctos + unos_confundidos} (errores: {unos_confundidos})")
print(f"   -> Total de errores en todo el examen: {ceros_confundidos + unos_confundidos} de {len(y_test)} imágenes\n")

# 3. Verificación del caso sintético conocido: Imagen totalmente negra (x = 0)
imagen_negra = np.zeros((1, 784))
prob_negra_sklearn = neurona.predict_proba(imagen_negra)[0, 1]
prob_negra_teorica = 1.0 / (1.0 + np.exp(-sesgo_b))
print("3. Verificación con entrada de resultado conocido (Imagen toda negra x = 0):")
print(f"   -> Valor teórico esperado sigma(b): {prob_negra_teorica:.8f}")
print(f"   -> Valor devuelto por el programa:  {prob_negra_sklearn:.8f}\n")

# 4. Gráfico Principal de Verificación: Curva de Error + Mapas de Pesos 28x28
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Paso 6: Verificación Matemática y Física de la Neurona Artificial", fontsize=13, fontweight="bold")

# Panel A: Descenso monótono del error (Propiedad matemática)
ejes[0].plot(range(epocas), historial_error, color="#1f77b4", linewidth=2.5)
ejes[0].scatter([0, 1, 10, 99], [historial_error[0], historial_error[1], historial_error[10], historial_error[99]], color="#d62728", zorder=5)
ejes[0].set_title("A. Descenso del Error (Entropía Cruzada)", fontsize=11, fontweight="bold")
ejes[0].set_xlabel("Época (Iteración de Descenso por Gradiente)")
ejes[0].set_ylabel("Error promedio L (Log-Loss)")
ejes[0].grid(True, linestyle="--", alpha=0.5)

# Panel B: Mapa de calor de los 784 pesos de scikit-learn (28x28)
max_abs_sk = np.max(np.abs(pesos_w))
im1 = ejes[1].imshow(pesos_w.reshape(28, 28), cmap="RdBu_r", vmin=-max_abs_sk, vmax=max_abs_sk)
ejes[1].set_title("B. Mapa de 784 Pesos (scikit-learn)\nRojo > 0 (vota '1') | Azul < 0 (vota '0')", fontsize=10, fontweight="bold")
ejes[1].axis("off")
plt.colorbar(im1, ax=ejes[1], fraction=0.046, pad=0.04)

# Panel C: Mapa de calor de los 784 pesos de nuestra neurona NumPy (28x28)
max_abs_np = np.max(np.abs(w_propio))
im2 = ejes[2].imshow(w_propio.reshape(28, 28), cmap="RdBu_r", vmin=-max_abs_np, vmax=max_abs_np)
ejes[2].set_title("C. Mapa de 784 Pesos (Nuestro NumPy)\nRojo > 0 (vota '1') | Azul < 0 (vota '0')", fontsize=10, fontweight="bold")
ejes[2].axis("off")
plt.colorbar(im2, ax=ejes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
ruta_verificacion = os.path.join(CARPETA_PROYECTO, "paso6_verificacion_modelo.png")
plt.savefig(ruta_verificacion, dpi=150)
plt.close()
print(f"Gráfico de verificación guardado en: {ruta_verificacion}")

# 5. Gráfico de inspección de los pocos errores cometidos en Test
indices_errores = np.where(y_pred_sklearn != y_test)[0]
probs_test = neurona.predict_proba(X_test)[:, 1]
n_mostrar = min(len(indices_errores), 6)

if n_mostrar > 0:
    fig, ejes_err = plt.subplots(1, n_mostrar, figsize=(2.5 * n_mostrar, 3))
    if n_mostrar == 1:
        ejes_err = [ejes_err]
    fig.suptitle(f"Inspección de Errores en Test ({len(indices_errores)} casos ambiguos de {len(y_test)})", fontsize=12, fontweight="bold")
    for i in range(n_mostrar):
        idx = indices_errores[i]
        ejes_err[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ejes_err[i].set_title(
            f"Real: {y_test[idx]} | Pred: {y_pred_sklearn[idx]}\nProb '1': {probs_test[idx]:.2f}",
            fontsize=9,
        )
        ejes_err[i].axis("off")
    plt.tight_layout()
    ruta_errores = os.path.join(CARPETA_PROYECTO, "paso6_errores_test.png")
    plt.savefig(ruta_errores, dpi=150)
    plt.close()
    print(f"Gráfico de errores en Test guardado en: {ruta_errores}")




