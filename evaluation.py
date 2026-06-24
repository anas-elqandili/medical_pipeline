import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from model import _augmenter

# ── Chargement ────────────────────────────────────────────────────────────────
file_path  = r"C:\Users\user\medical_pipeline\data\raw\symptoms_dataset.csv"
model_path = r"C:\Users\user\medical_pipeline\models\random_forest.pkl"
enc_path   = r"C:\Users\user\medical_pipeline\models\label_encoder.pkl"
feat_path  = r"C:\Users\user\medical_pipeline\models\features.pkl"

df = pd.read_csv(file_path)

with open(model_path, "rb") as f: modele_rf = pickle.load(f)
with open(enc_path,   "rb") as f: le        = pickle.load(f)
with open(feat_path,  "rb") as f: features  = pickle.load(f)

# ── Même augmentation que l'entraînement ─────────────────────────────────────
df_aug = _augmenter(df, features, n=5)
X = df_aug[features].values
y = df_aug["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Prédiction ────────────────────────────────────────────────────────────────
y_pred = modele_rf.predict(X_test)

# ── Noms lisibles — uniquement les classes présentes dans le test ─────────────
classes_presentes = np.unique(np.concatenate([y_test, y_pred]))
noms_presents = [le.classes_noms_[i] for i in classes_presentes]

print("\n" + "="*50)
print("   RÉSULTATS — RANDOM FOREST MULTICLASSE")
print("="*50)
print(f"Nombre de classes total  : {len(np.unique(y))} maladies")
print(f"Classes vues au test     : {len(classes_presentes)}")
print(f"Baseline aléatoire       : {100/len(np.unique(y)):.2f}%")
print(f"Accuracy du modèle       : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score (weighted)      : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score (macro)         : {f1_score(y_test, y_pred, average='macro'):.4f}")
print("-"*50)

# ── Rapport détaillé ──────────────────────────────────────────────────────────
report = classification_report(
    y_test, y_pred,
    labels=classes_presentes,       # ✅ uniquement les classes présentes
    target_names=noms_presents,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T
report_df = report_df[report_df["support"] > 0].sort_values("f1-score", ascending=False)

print("\nTOP 10 — Maladies les mieux détectées :")
print(report_df[["precision","recall","f1-score","support"]].head(10).to_string())
print("\nBAS 5 — Maladies les moins bien détectées :")
print(report_df[["precision","recall","f1-score","support"]].tail(5).to_string())
print("="*50)

# ── Résumé pour ta soutenance ─────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred) * 100
baseline = 100 / len(np.unique(y))
print(f"\n RÉSUMÉ")
print(f"   Le modèle atteint {acc:.1f}% d'accuracy")
print(f"   soit {acc/baseline:.0f}x mieux qu'un choix aléatoire ({baseline:.2f}%)")