"""
MODULE — Modèle IA : Diagnostic multiclasse + Identification de la maladie
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

CHEMIN_RF       = "models/random_forest.pkl"
CHEMIN_LR       = "models/logistic_regression.pkl"
CHEMIN_SCALER   = "models/scaler.pkl"
CHEMIN_FEATURES = "models/features.pkl"
CHEMIN_ENCODER  = "models/label_encoder.pkl"


def entrainer(df: pd.DataFrame) -> dict:
    os.makedirs("models", exist_ok=True)

    cols = [c for c in df.columns if c not in ["maladie", "label"]]
    X = df[cols].values
    y = df["label"].values


    le = LabelEncoder()
    le.fit(y)
    le.classes_noms_ = (
        df[["label", "maladie"]]
        .drop_duplicates()
        .sort_values("label")["maladie"]
        .values
    )

    with open(CHEMIN_FEATURES, "wb") as f:
        pickle.dump(cols, f)
    with open(CHEMIN_ENCODER, "wb") as f:
        pickle.dump(le, f)


    df_aug = _augmenter(df, cols, n=5)
    X_aug = df_aug[cols].values
    y_aug = df_aug["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug,
        test_size=0.2,
        random_state=42
    )


    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    acc_rf = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 1)
    with open(CHEMIN_RF, "wb") as f:
        pickle.dump(rf, f)


    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    lr = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        C=0.1
    )
    lr.fit(X_tr_sc, y_train)
    acc_lr = round(accuracy_score(y_test, lr.predict(X_te_sc)) * 100, 1)
    with open(CHEMIN_LR, "wb") as f:
        pickle.dump(lr, f)
    with open(CHEMIN_SCALER, "wb") as f:
        pickle.dump(scaler, f)

    return {
        "random_forest":       {"accuracy": acc_rf},
        "logistic_regression": {"accuracy": acc_lr},
        "nb_features":         len(cols),
        "nb_classes":          int(len(np.unique(y))),
        "nb_lignes_original":  len(df),
        "nb_lignes_augmente":  len(df_aug),
        "nb_malades":          int((df["maladie"] != "Sain").sum()),
        "nb_sains":            int((df["maladie"] == "Sain").sum()),
    }


def _augmenter(df: pd.DataFrame, cols: list, n: int = 5) -> pd.DataFrame:
    """
    Augmentation simple : pour chaque ligne, crée n variantes
    en faisant du bruit binaire (flip aléatoire de 1-2 symptômes mineurs).
    """
    import random
    random.seed(42)
    np.random.seed(42)

    nouvelles = []
    for _, row in df.iterrows():
        for _ in range(n):
            nouvelle = row.copy()

            for col in cols:
                if np.random.random() < 0.08:
                    nouvelle[col] = 1 - int(nouvelle[col])
            nouvelles.append(nouvelle)

    df_aug = pd.concat([df, pd.DataFrame(nouvelles)], ignore_index=True)
    return df_aug


def predire(symptomes: dict, modele: str) -> dict:
    with open(CHEMIN_FEATURES, "rb") as f:
        features = pickle.load(f)
    with open(CHEMIN_ENCODER, "rb") as f:
        le = pickle.load(f)

    X = np.array([[symptomes.get(f, 0) for f in features]])

    if modele == "random_forest":
        with open(CHEMIN_RF, "rb") as f:
            clf = pickle.load(f)
        probas = clf.predict_proba(X)[0]
    else:
        with open(CHEMIN_LR, "rb") as f:
            clf = pickle.load(f)
        with open(CHEMIN_SCALER, "rb") as f:
            scaler = pickle.load(f)
        probas = clf.predict_proba(scaler.transform(X))[0]


    noms = list(le.classes_noms_)
    idx_sain = noms.index("Sain")
    prob_sain = float(probas[idx_sain])


    probas_maladies = probas.copy()
    probas_maladies[idx_sain] = 0
    idx_meilleure_maladie = int(np.argmax(probas_maladies))
    prob_meilleure_maladie = float(probas[idx_meilleure_maladie])
    nom_meilleure_maladie = noms[idx_meilleure_maladie]


    nb_symptomes = sum(symptomes.values())

    if prob_sain > 0.50 and nb_symptomes <= 3:
        est_sain   = True
        nom_predit = "Sain"
        proba_max  = prob_sain
    else:

        est_sain   = False
        nom_predit = nom_meilleure_maladie
        proba_max  = prob_meilleure_maladie


    top3_idx = np.argsort(probas_maladies)[::-1][:3]
    top3 = [
        {
            "maladie":     noms[i],
            "probabilite": round(float(probas[i]) * 100, 1)
        }
        for i in top3_idx
    ]


    if proba_max >= 0.30 and nb_symptomes >= 3:
        confiance = "Haute"
    elif proba_max >= 0.15 or nb_symptomes >= 2:
        confiance = "Moyenne"
    else:
        confiance = "Faible"

    return {
        "diagnostic":       "Sain" if est_sain else "Malade",
        "maladie_probable": nom_predit,
        "probabilite":      round(proba_max * 100, 1),
        "confiance":        confiance,
        "top3":             top3,
        "est_sain":         est_sain,
        "prob_sain":        round(prob_sain * 100, 1),
    }


def modeles_prets() -> bool:
    return all(os.path.exists(p) for p in [
        CHEMIN_RF, CHEMIN_LR, CHEMIN_SCALER,
        CHEMIN_FEATURES, CHEMIN_ENCODER
    ])