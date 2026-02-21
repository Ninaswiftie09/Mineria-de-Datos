import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import wfdb

def load_physionet_ecg(record="100", db="mitdb", seconds=10):
    sig, fields = wfdb.rdsamp(record_name=record, pn_dir=db)

    fs = fields["fs"] 
    n = int(seconds * fs)

    X = sig[:n, :2]
    t = np.arange(n) / fs
    return X, t, fs, fields

def mix_signals(S):
    A = np.array([[1.0, 0.6],
                  [0.4, 1.0]])
    X = S @ A.T
    return X, A

def plot_signals(t, signals, title, labels):
    plt.figure(figsize=(10, 5))
    for i in range(signals.shape[1]):
        plt.plot(t, signals[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    S, t, fs, fields = load_physionet_ecg(record="100", db="mitdb", seconds=10)

    S = S - S.mean(axis=0)

    X, A = mix_signals(S)

    ica = FastICA(n_components=2, random_state=42, max_iter=2000, tol=1e-4)
    S_hat = ica.fit_transform(X)  

    plot_signals(t, S, "Señales originales (2 canales ECG de PhysioNet)", ["Canal 1", "Canal 2"])
    plot_signals(t, X, "Señales mezcladas (mezcla artificial)", ["Mezcla 1", "Mezcla 2"])
    plot_signals(t, S_hat, "Componentes recuperadas con ICA (FastICA)", ["ICA 1", "ICA 2"])

    print("Frecuencia de muestreo (fs):", fs)
    print("Canales:", fields.get("sig_name", ["ch1", "ch2"]))
    print("Matriz de mezcla usada A:\n", A)

if __name__ == "__main__":
    main()