import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import umap

# Cargar dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

# Graficar
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=y)
plt.title("UMAP - Breast Cancer Dataset")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(scatter)
plt.show()