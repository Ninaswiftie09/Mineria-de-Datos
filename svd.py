import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

def load_movielens_100k(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):   
        raise FileNotFoundError(f"No se encuentra el archivo: {data_path}")

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    return df

def build_user_item_matrix(df: pd.DataFrame):
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
    item_id_to_idx = {iid: j for j, iid in enumerate(sorted(item_ids))}

    rows = df["user_id"].map(user_id_to_idx).to_numpy()
    cols = df["item_id"].map(item_id_to_idx).to_numpy()
    data = df["rating"].astype(float).to_numpy()

    n_users = len(user_id_to_idx)
    n_items = len(item_id_to_idx)

    R = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()
    return R, user_id_to_idx, item_id_to_idx

def center_by_user(R):
    R = R.tocsr()
    n_users = R.shape[0]
    user_means = np.zeros(n_users, dtype=float)

    R_centered = R.copy().astype(float)
    for u in range(n_users):
        start, end = R_centered.indptr[u], R_centered.indptr[u + 1]
        if start == end:
            continue
        vals = R_centered.data[start:end]
        mu = vals.mean()
        user_means[u] = mu
        R_centered.data[start:end] = vals - mu

    return R_centered, user_means

def explained_energy(singular_values):
    s2 = singular_values**2
    return s2 / s2.sum()

def main():
    data_path = os.path.join("ml-100k", "u.data")

    print("Cargando...")
    df = load_movielens_100k(data_path)

    print("\n== Resumen del dataset ==")
    n_ratings = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"Ratings totales: {n_ratings}")
    print(f"Usuarios únicos: {n_users}")
    print(f"Películas únicas: {n_items}")
    print(f"Rating min/max: {df['rating'].min()} / {df['rating'].max()}")

    density = n_ratings / (n_users * n_items)
    print(f"Densidad: {density:.4f} (~{density*100:.2f}%)")

    print("\n== Construyendo matriz usuario–película (dispersa) ==")
    R, user_map, item_map = build_user_item_matrix(df)
    print(f"Shape de R: {R.shape}")
    print(f"Valores no-cero (nnz): {R.nnz}")

    print("\n== Centrando por usuario (opcional, recomendado) ==")
    R_centered, user_means = center_by_user(R)

    k = 20
    print(f"\n== Ejecutando SVD truncado con k={k} ==")
    U, s, Vt = svds(R_centered, k=k)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]

    print("\n== Resultados ==")
    print("Top 10 singular values:")
    print(np.round(s[:10], 4))

    energy = explained_energy(s)
    print("\nEnergía aproximada capturada:")
    print(np.round(energy[:10], 4))
    print(f"Suma energía (k componentes): {energy.sum():.4f}")

    print("\n== Ejemplo de predicción ==")
    u = 0
    i = 0
    pred_centered = (U[u, :] * s) @ Vt[:, i]
    pred_rating = user_means[u] + pred_centered
    print(f"Predicción aproximada: {pred_rating:.3f}")

if __name__ == "__main__":
    main()
