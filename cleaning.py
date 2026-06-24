"""
MODULE — Nettoyage des données symptômes NHS
"""
import numpy as np
import pandas as pd


def nettoyer_dataset(df: pd.DataFrame) -> tuple:
    rapport = {}
    df = df.copy()
    df, rapport["doublons"]  = supprimer_doublons(df)
    df, rapport["manquants"] = gerer_manquants(df)
    df, rapport["types"]     = corriger_types(df)
    return df, rapport


def supprimer_doublons(df):
    nb = int(df.duplicated().sum())
    df = df.drop_duplicates()
    return df, {"nb": nb, "statut": f" {nb} doublon(s) supprimé(s)"}


def gerer_manquants(df):
    nb = int(df.isnull().sum().sum())
    # Colonnes binaires symptômes → remplir par 0
    cols_symptomes = [c for c in df.columns if c not in ["maladie", "label"]]
    df[cols_symptomes] = df[cols_symptomes].fillna(0)
    return df, {"nb_initial": nb, "statut": f" {nb} valeur(s) manquante(s) traitée(s)"}


def corriger_types(df):
    cols = [c for c in df.columns if c not in ["maladie"]]
    for col in cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    return df, {"statut": "Types corrigés"}
