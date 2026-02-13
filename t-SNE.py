import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# 1. Cargar dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Estandarizar (MUY IMPORTANTE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicar t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

X_embedded = tsne.fit_transform(X_scaled)

# 4. Graficar
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)
plt.title("t-SNE - Breast Cancer Dataset")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(scatter)
plt.show()
