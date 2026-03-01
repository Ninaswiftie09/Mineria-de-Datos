import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

INPUT_CSV = "movies_2026.csv"

COLS = ["budget", "revenue", "runtime", "popularity", "voteAvg", "voteCount", "releaseYear"]

SETTINGS = [
    (0.05, 0.60),
    (0.03, 0.70),
    (0.02, 0.80),
]

LIFT_MIN = 1.10
TOP_N = 15

FILTER_TOO_COMMON = True
TOO_COMMON_THRESHOLD = 0.80  


def discretize_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    bins = pd.qcut(series, q=q, duplicates="drop")
    return pd.Series([f"{prefix}={str(b)}" for b in bins], index=series.index)


def run_apriori(onehot_df: pd.DataFrame, min_support: float, min_conf: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = apriori(onehot_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        rules = rules[rules["lift"] >= LIFT_MIN]
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    return freq, rules


def main():
    df = pd.read_csv(INPUT_CSV, encoding="latin1")

    data = df[COLS].copy().dropna()

    disc = pd.DataFrame(index=data.index)
    for c in COLS:
        disc[c] = discretize_qcut(data[c], q=3, prefix=c)

    onehot = pd.get_dummies(disc)

    disc.head(10).to_csv("apriori_discretizacion_muestra.csv", index=False)

    removed_items = []
    onehot_used = onehot

    if FILTER_TOO_COMMON:
        item_freq = onehot.mean().sort_values(ascending=False)
        removed_items = item_freq[item_freq > TOO_COMMON_THRESHOLD].index.tolist()
        if removed_items:
            onehot_used = onehot.drop(columns=removed_items)
            pd.DataFrame({
                "item": removed_items,
                "frecuencia": [float(item_freq[i]) for i in removed_items]
            }).to_csv("apriori_items_muy_frecuentes.csv", index=False)

    for sup, conf in SETTINGS:
        freq, rules = run_apriori(onehot_used, sup, conf)

        tag = f"sup{str(sup).replace('.','_')}_conf{str(conf).replace('.','_')}"
        freq.to_csv(f"apriori_itemsets_{tag}.csv", index=False)

        if rules.empty:
            rules.to_csv(f"apriori_reglas_{tag}.csv", index=False)
        else:
            top_rules = rules.head(TOP_N).copy()

            top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
            top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

            cols_out = ["antecedents", "consequents", "support", "confidence", "lift"]
            top_rules[cols_out].to_csv(f"apriori_reglas_top_{tag}.csv", index=False)

    print("Archivos generados:")
    print("- apriori_discretizacion_muestra.csv")
    if FILTER_TOO_COMMON and removed_items:
        print("- apriori_items_muy_frecuentes.csv")
    for sup, conf in SETTINGS:
        tag = f"sup{str(sup).replace('.','_')}_conf{str(conf).replace('.','_')}"
        print(f"- apriori_itemsets_{tag}.csv")
        print(f"- apriori_reglas_top_{tag}.csv")


if __name__ == "__main__":
    main()