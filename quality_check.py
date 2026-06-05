"""
MODULE — Vérification qualité des données
"""
import numpy as np
import pandas as pd


def verifier_qualite(df: pd.DataFrame) -> dict:
    resultats = {
        "manquants":   _check_manquants(df),
        "doublons":    _check_doublons(df),
        "equilibre":   _check_equilibre(df),
        "completude":  _check_completude(df),
    }
    nb_ok = sum(1 for v in resultats.values() if v["passe"])
    resultats["score"] = round((nb_ok / len(resultats)) * 100)
    return resultats


def _check_manquants(df):
    problemes = [f"{c}: {df[c].isnull().mean():.1%}"
                 for c in df.columns if df[c].isnull().mean() > 0.15]
    passe = len(problemes) == 0
    return {"nom": "Valeurs manquantes < 15%", "passe": passe,
            "statut": "✅" if passe else "⚠️",
            "details": problemes or ["Toutes colonnes OK"]}


def _check_doublons(df):
    nb = int(df.duplicated().sum())
    return {"nom": "Absence de doublons", "passe": nb == 0,
            "statut": "✅" if nb == 0 else "⚠️",
            "details": [f"{nb} doublon(s)" if nb > 0 else "Aucun doublon"]}


def _check_equilibre(df):
    if "label" not in df.columns:
        return {"nom": "Équilibre classes", "passe": True, "statut": "✅", "details": ["Pas de label"]}
    ratio = df["label"].mean()
    passe = 0.3 <= ratio <= 0.7
    return {"nom": "Équilibre malades/sains", "passe": passe,
            "statut": "✅" if passe else "⚠️",
            "details": [f"Malades: {ratio:.1%} | Sains: {1-ratio:.1%}"]}


def _check_completude(df):
    cols_symptomes = [c for c in df.columns if c not in ["maladie", "label"]]
    lignes_vides = int((df[cols_symptomes].sum(axis=1) == 0).sum())
    passe = lignes_vides == 0
    return {"nom": "Lignes avec au moins 1 symptôme", "passe": passe,
            "statut": "✅" if passe else "⚠️",
            "details": [f"{lignes_vides} ligne(s) sans aucun symptôme"
                        if lignes_vides > 0 else "Toutes les lignes ont des symptômes"]}
