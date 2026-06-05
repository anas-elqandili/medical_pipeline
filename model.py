"""
MODULE — Modèle IA : Diagnostic + Identification de la maladie
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

CHEMIN_RF       = "models/random_forest.pkl"
CHEMIN_LR       = "models/logistic_regression.pkl"
CHEMIN_SCALER   = "models/scaler.pkl"
CHEMIN_FEATURES = "models/features.pkl"
CHEMIN_MALADIES = "models/maladies_profils.pkl"


def entrainer(df: pd.DataFrame) -> dict:
    os.makedirs("models", exist_ok=True)

    cols = [c for c in df.columns if c not in ["maladie","label"]]
    X = df[cols].values
    y = df["label"].values

    with open(CHEMIN_FEATURES, "wb") as f:
        pickle.dump(cols, f)

    # Sauvegarder les profils de maladies pour l'identification
    profils = {}
    for _, row in df[df["label"]==1].iterrows():
        profils[row["maladie"]] = {c: row[c] for c in cols}
    with open(CHEMIN_MALADIES, "wb") as f:
        pickle.dump(profils, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    acc_rf = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 1)
    with open(CHEMIN_RF, "wb") as f:
        pickle.dump(rf, f)

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42,
                            class_weight="balanced")
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
        "nb_malades":          int((y==1).sum()),
        "nb_sains":            int((y==0).sum()),
    }


def predire(symptomes: dict, modele: str ) -> dict:
    with open(CHEMIN_FEATURES, "rb") as f:
        features = pickle.load(f)
    with open(CHEMIN_MALADIES, "rb") as f:
        profils = pickle.load(f)

    X = np.array([[symptomes.get(f, 0) for f in features]])

    if modele == "random_forest":
        with open(CHEMIN_RF, "rb") as f:
            clf = pickle.load(f)
        proba = clf.predict_proba(X)[0]
    else:
        with open(CHEMIN_LR, "rb") as f:
            clf = pickle.load(f)
        with open(CHEMIN_SCALER, "rb") as f:
            scaler = pickle.load(f)
        proba = clf.predict_proba(scaler.transform(X))[0]

    prob_malade = round(float(proba[1]), 3)
    prob_sain   = round(float(proba[0]), 3)
    diagnostic  = "Malade" if prob_malade >= 0.5 else "Sain"

    if max(prob_malade, prob_sain) >= 0.80:
        confiance = "Haute"
    elif max(prob_malade, prob_sain) >= 0.60:
        confiance = "Moyenne"
    else:
        confiance = "Faible"

    # Identifier la maladie la plus probable (similarité Jaccard)
    maladie_probable = None
    maladie_score    = -1
    if diagnostic == "Malade":
        symptomes_actifs = {k for k, v in symptomes.items() if v == 1}
        for nom, profil in profils.items():
            symp_maladie = {k for k, v in profil.items() if v == 1}
            if not symp_maladie:
                continue
            inter = len(symptomes_actifs & symp_maladie)
            union = len(symptomes_actifs | symp_maladie)
            score = inter / union if union > 0 else 0
            if score > maladie_score:
                maladie_score    = score
                maladie_probable = nom

    return {
        "diagnostic":         diagnostic,
        "probabilite_malade": prob_malade,
        "probabilite_sain":   prob_sain,
        "confiance":          confiance,
        "maladie_probable":   maladie_probable,
        "similarite":         round(maladie_score * 100, 1) if maladie_score >= 0 else 0,
    }


def modeles_prets() -> bool:
    return all(os.path.exists(p) for p in [
        CHEMIN_RF, CHEMIN_LR, CHEMIN_SCALER,
        CHEMIN_FEATURES, CHEMIN_MALADIES
    ])
