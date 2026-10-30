"""Entraînement d'un détecteur d'attaques réseau (IDS) : normal (0) / attaque (1)."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flux_train.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

TARGET = "attaque"
CATEGORICAL = ["protocol"]
SEED = 42


def build_pipeline(classifier):
    """Prétraitement (imputation, échelle, encodage) + classifieur dans un seul Pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, CATEGORICAL),
        ])),
        ("clf", classifier),
    ])


def explore(df):
    print("=== Exploration ===")
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Répartition de la cible :\n", df[TARGET].value_counts(normalize=True).round(3))
    print()


def main():
    df = pd.read_csv(DATA)
    explore(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    }

    results = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf).fit(X_train, y_train)
        pred = pipe.predict(X_val)
        results[name] = {
            "f1": f1_score(y_val, pred),
            "recall": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred),
        }
        print(f"=== {name} ===")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"]))
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"Matrice de confusion : VN={tn} FP={fp} FN={fn} VP={tp}\n")

    print("=== Tableau récapitulatif (classe attaque) ===")
    print("| Modèle | F1 | Rappel | Précision |\n|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m['f1']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} |")

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nModèle retenu : {best}")

    final = build_pipeline(candidates[best]).fit(X, y)  # ré-entraîné sur toutes les données
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Modèle enregistré dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
