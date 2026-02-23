# ===============================
# PCA - Laboratorio Mineria de Datos
# ===============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity


# ===============================
# 1. Verificar archivos en carpeta
# ===============================
print("Archivos en la carpeta actual:")
print(os.listdir())

# ===============================
# 2. Cargar dataset
# ===============================
df = pd.read_csv("Mineria-de-Datos/movies_2026.csv", encoding="latin1")
print("\nDataset cargado correctamente.")
print(df.head())

# ===============================
# 3. Seleccionar variables numéricas
# ===============================
df_num = df.select_dtypes(include=['int64', 'float64'])

if 'id' in df_num.columns:
    df_num = df_num.drop(columns=['id'])

print("\nVariables numéricas utilizadas:")
print(df_num.columns)

# ===============================
# 4. Manejo de valores nulos
# ===============================
print("\nValores nulos por variable:")
print(df_num.isnull().sum())

df_num = df_num.dropna()
print("\nDataset después de eliminar nulos:", df_num.shape)

# ===============================
# 5. Matriz de correlación
# ===============================
plt.figure(figsize=(12,10))
sns.heatmap(df_num.corr(), cmap='coolwarm')
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("matriz_correlacion.png")
plt.close()

print("\nMatriz de correlación guardada como imagen.")

# ===============================
# 6. Prueba KMO
# ===============================
kmo_all, kmo_model = calculate_kmo(df_num)
print("\nKMO:", kmo_model)

# ===============================
# 7. Test de Bartlett
# ===============================
chi_square_value, p_value = calculate_bartlett_sphericity(df_num)
print("Chi-cuadrado:", chi_square_value)
print("p-value:", p_value)

# ===============================
# 8. Escalar datos
# ===============================
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)

# ===============================
# 9. Aplicar PCA
# ===============================
pca = PCA()
pca.fit(df_scaled)

explained_variance = pca.explained_variance_ratio_

# ===============================
# 10. Scree Plot
# ===============================
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada')
plt.title('Scree Plot')
plt.tight_layout()
plt.savefig("scree_plot.png")
plt.close()

print("\nScree plot guardado.")

# ===============================
# 11. Varianza acumulada
# ===============================
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Acumulada')
plt.tight_layout()
plt.savefig("varianza_acumulada.png")
plt.close()

print("Gráfica de varianza acumulada guardada.")

# ===============================
# 12. Selección de componentes
# ===============================
var_acumulada = np.cumsum(explained_variance)
n_componentes = np.argmax(var_acumulada >= 0.80) + 1

print("\nNúmero de componentes que explican al menos 80%:", n_componentes)

pca_final = PCA(n_components=n_componentes)
principal_components = pca_final.fit_transform(df_scaled)

# ===============================
# 13. Cargas (Loadings)
# ===============================
loadings = pd.DataFrame(
    pca_final.components_.T,
    columns=[f'PC{i+1}' for i in range(n_componentes)],
    index=df_num.columns
)

print("\nCargas de los componentes:")
print(loadings)

# Guardar loadings
loadings.to_csv("loadings_pca.csv")

print("\nProceso de PCA finalizado correctamente.")