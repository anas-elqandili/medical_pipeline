import os
import pandas as pd
import streamlit as st

from scraper_nhs   import construire_dataset, SYMPTOMES_CIBLES, LABELS_FR
from cleaning      import nettoyer_dataset
from quality_check import verifier_qualite
from model         import entrainer, predire, modeles_prets


st.set_page_config(page_title="Diagnostic IA — NHS", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0a0d14; color: #e0e6f0; }
    .titre-section {
        font-size: 1.1rem; font-weight: 700;
        color: #fff; letter-spacing: 2px;
        padding: 0.5rem 0; margin-bottom: 1rem;
        border-bottom: 1px solid #1e2740;
    }
    .carte-resultat {
        border-radius: 14px; padding: 1.5rem 2rem;
        margin-top: 1rem; text-align: center;
    }
    .malade {
        background: linear-gradient(135deg, #2d0a14, #1a0a0a);
        border: 2px solid #ff4757;
    }
    .sain {
        background: linear-gradient(135deg, #0a2d1a, #0a1a0d);
        border: 2px solid #00e5c3;
    }
    .badge-maladie {
        background: #ff475722; color: #ff6b81;
        border: 1px solid #ff475744;
        border-radius: 20px; padding: 0.3rem 1rem;
        font-size: 0.8rem; font-weight: 700;
        display: inline-block; margin-top: 0.5rem;
    }
    .badge-sain {
        background: #00e5c322; color: #00e5c3;
        border: 1px solid #00e5c344;
        border-radius: 20px; padding: 0.3rem 1rem;
        font-size: 0.8rem; font-weight: 700;
        display: inline-block; margin-top: 0.5rem;
    }
    div[data-testid="stMetric"] {
        background: #161b27; border-radius: 10px;
        padding: 0.8rem; border: 1px solid #1e2740;
    }
</style>
""", unsafe_allow_html=True)


for cle in ["df_brut","df_clean","rapport_clean",
            "rapport_qualite","rapport_modele","pipeline_ok"]:
    if cle not in st.session_state:
        st.session_state[cle] = None
if st.session_state["pipeline_ok"] is None:
    st.session_state["pipeline_ok"] = False


st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117,#161b27);
            padding:2rem 2.5rem; border-bottom:1px solid #1e2740;
            margin-bottom:2rem;">
    <div style="font-size:2rem; font-weight:900; color:#fff; letter-spacing:3px;">
        DIAGNOSTIC IA
    </div>
    <div style="font-size:0.8rem; color:#5a6a8a; letter-spacing:3px; margin-top:0.3rem;">
        NHS SYMPTOMS SCRAPER — MACHINE LEARNING — CLASSIFICATION
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Scraping & Données",
    "Pipeline & Modèle IA",
    "Diagnostic Patient",
])


with tab1:
    st.markdown('<div class="titre-section">SOURCE DE DONNÉES NHS</div>',
                unsafe_allow_html=True)

    col_url, col_btn = st.columns([4, 1])
    with col_url:
        url_input = st.text_input(
            "URL du site NHS :",
            value="https://www.nhs.uk/conditions/",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        nb_maladies = st.slider("Nb maladies", 10, 100, 40, step=10)

    lancer = st.button("Lancer le scraping", type="primary",
                       use_container_width=True)

    if os.path.exists("data/raw/symptoms_dataset.csv"):
        st.info("Dataset existant détecté — vous pouvez relancer ou continuer.")
        if st.button("Charger le dataset existant"):
            st.session_state["df_brut"] = pd.read_csv(
                "data/raw/symptoms_dataset.csv"
            )
            st.success("Dataset chargé !")

    if lancer:
        if not url_input.startswith("http"):
            st.error("L'URL doit commencer par http://")
        else:
            progress = st.progress(0)
            status   = st.empty()
            log_box  = st.empty()
            logs     = []

            def callback(i, total, nom):
                progress.progress(int((i / total) * 100))
                status.markdown(f"**Scraping ({i}/{total}) : {nom}**")
                logs.append(nom)
                log_box.text_area("Journal", "\n".join(logs[-8:]),
                                  height=140, key=f"log_{i}")

            with st.spinner("Scraping NHS en cours..."):
                try:
                    df = construire_dataset(url_input, nb_maladies, callback)
                    st.session_state["df_brut"] = df
                    progress.progress(100)
                    status.markdown("**Scraping terminé !**")
                except Exception as e:
                    st.error(f"Erreur scraping : {e}")

    if st.session_state["df_brut"] is not None:
        df = st.session_state["df_brut"]
        st.markdown('<div class="titre-section">DATASET SCRAPÉ</div>',
                    unsafe_allow_html=True)


        nb_sains   = int((df["maladie"] == "Sain").sum())
        nb_malades = int((df["maladie"] != "Sain").sum())
        nb_classes = int(df["maladie"].nunique())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total patients",  len(df))
        m2.metric("Malades",         nb_malades)
        m3.metric("Sains",           nb_sains)
        m4.metric("Maladies uniques", nb_classes)

        cols_affichage = ["maladie"] + SYMPTOMES_CIBLES[:8] + ["label"]
        st.dataframe(
            df[[c for c in cols_affichage if c in df.columns]],
            use_container_width=True, height=250
        )

        st.markdown("**Top 10 symptômes les plus fréquents chez les malades :**")
        malades = df[df["maladie"] != "Sain"]
        freq = malades[SYMPTOMES_CIBLES].sum().sort_values(ascending=False).head(10)
        freq.index = [LABELS_FR.get(c, c) for c in freq.index]
        st.bar_chart(freq)

        st.success("Dataset prêt — allez dans l'onglet **2 — Pipeline & Modèle**")


with tab2:
    st.markdown('<div class="titre-section">PIPELINE AUTOMATIQUE</div>',
                unsafe_allow_html=True)

    if st.session_state["df_brut"] is None:
        st.warning("Veuillez d'abord scraper les données dans l'onglet 1.")
    else:
        if st.button(" Lancer le Pipeline complet", type="primary",
                     use_container_width=True):

            df_brut = st.session_state["df_brut"]

            with st.spinner("Nettoyage des données..."):
                df_clean, rapport_clean = nettoyer_dataset(df_brut)
                st.session_state["df_clean"]      = df_clean
                st.session_state["rapport_clean"] = rapport_clean
            st.success("Nettoyage terminé")

            with st.spinner("Vérification qualité..."):
                rapport_q = verifier_qualite(df_clean)
                st.session_state["rapport_qualite"] = rapport_q
            st.success(f"Qualité vérifiée — Score : {rapport_q['score']}/100")

            with st.spinner("Entraînement des modèles IA..."):
                rapport_m = entrainer(df_clean)
                st.session_state["rapport_modele"] = rapport_m
                st.session_state["pipeline_ok"]    = True

            rf_acc = rapport_m['random_forest']['accuracy']
            lr_acc = rapport_m['logistic_regression']['accuracy']
            st.success(
                f"Modèles entraînés — RF : {rf_acc}% | LR : {lr_acc}%"
            )

        if st.session_state["rapport_clean"] is not None:
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Nettoyage**")
                for v in st.session_state["rapport_clean"].values():
                    st.markdown(f"- {v.get('statut','')}")

            with col_b:
                st.markdown("**Qualité**")
                r = st.session_state["rapport_qualite"]
                st.metric("Score global", f"{r.get('score', 0)}/100")
                for k, v in r.items():
                    if k != "score":
                        st.markdown(f"{v['statut']} {v['nom']}")
                        for d in v["details"]:
                            st.caption(f"→ {d}")

            with col_c:
                st.markdown("**Modèles IA**")
                r = st.session_state["rapport_modele"]
                if r:
                    st.metric("Random Forest",
                              f"{r['random_forest']['accuracy']}%")
                    st.metric("Régression Logistique",
                              f"{r['logistic_regression']['accuracy']}%")
                    st.caption(f"Features  : {r['nb_features']}")
                    st.caption(f"Classes   : {r['nb_classes']}")
                    st.caption(f"Malades   : {r['nb_malades']}")
                    st.caption(f"Sains     : {r['nb_sains']}")

            if st.session_state["pipeline_ok"]:
                st.success("Pipeline complet — allez dans l'onglet **3 — Diagnostic**")



with tab3:
    st.markdown('<div class="titre-section">DIAGNOSTIC PATIENT</div>',
                unsafe_allow_html=True)

    if not st.session_state["pipeline_ok"] and not modeles_prets():
        st.warning("Veuillez d'abord lancer le pipeline dans l'onglet 2.")
    else:
        modele_choisi = "random_forest"

        st.markdown("**Cochez les symptômes du patient :**")
        symptomes_coches = {}
        cols_sym = st.columns(5)

        for i, sym in enumerate(SYMPTOMES_CIBLES):
            with cols_sym[i % 5]:
                val = st.checkbox(LABELS_FR.get(sym, sym), key=f"sym_{sym}")
                symptomes_coches[sym] = 1 if val else 0

        nb_coches = sum(symptomes_coches.values())
        st.caption(f"Symptômes sélectionnés : **{nb_coches}**")

        if st.button("Diagnostiquer le patient",
                     type="primary", use_container_width=True):

            if nb_coches == 0:
                st.warning("Cochez au moins un symptôme.")
            else:
                with st.spinner("Analyse en cours..."):
                    #  Nouveau format de predire()
                    resultat = predire(symptomes_coches, modele_choisi)

                st.markdown("---")
                st.markdown("### Résultat du diagnostic")

                est_sain   = resultat["est_sain"]
                maladie    = resultat["maladie_probable"]
                probabilite = resultat["probabilite"]   # déjà en %
                confiance  = resultat["confiance"]
                top3       = resultat["top3"]

                if not est_sain:
                    st.markdown(f"""
                    <div class="carte-resultat malade">
                        <div style="font-size:3rem;"></div>
                        <div style="font-size:2.2rem; font-weight:900;
                                    color:#ff4757; letter-spacing:3px;">MALADE</div>
                        <div style="font-size:1rem; color:#ff6b81; margin-top:0.5rem;">
                            Probabilité : <strong>{probabilite}%</strong>
                        </div>
                        <div style="font-size:0.85rem; color:#8892a4; margin-top:0.3rem;">
                            Confiance : <strong>{confiance}</strong>
                        </div>
                        <div style="margin-top:1rem;">
                            <span class="badge-maladie">
                                Maladie probable : {maladie}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="carte-resultat sain">
                        <div style="font-size:3rem;"></div>
                        <div style="font-size:2.2rem; font-weight:900;
                                    color:#00e5c3; letter-spacing:3px;">SAIN</div>
                        <div style="font-size:1rem; color:#00e5c3; margin-top:0.5rem;">
                            Probabilité d'être sain : <strong>{probabilite}%</strong>
                        </div>
                        <div style="font-size:0.85rem; color:#8892a4; margin-top:0.3rem;">
                            Confiance : <strong>{confiance}</strong>
                        </div>
                        <div style="margin-top:1rem;">
                            <span class="badge-sain">Aucune maladie détectée</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


                
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilité",  f"{probabilite}%")
                c2.metric("Confiance",    confiance)
                c3.metric("Symptômes",    f"{nb_coches}/{len(SYMPTOMES_CIBLES)}")

                
                st.markdown("**Symptômes analysés :**")
                syms_actifs = [LABELS_FR.get(s, s)
                               for s, v in symptomes_coches.items() if v == 1]
                st.markdown(" ".join([
                    f'<span style="background:#1e2740; color:#c9d1e0; '
                    f'border-radius:6px; padding:0.2rem 0.6rem; '
                    f'margin:0.2rem; font-size:0.78rem;">{s}</span>'
                    for s in syms_actifs
                ]), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.warning(
                    "**Avertissement médical** : Ce diagnostic est généré par une IA "
                    "à des fins éducatives uniquement. Consultez toujours un médecin "
                    "pour un diagnostic médical réel."
                )


st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#3a4560; font-size:0.75rem;">'
    'Diagnostic IA — NHS Scraper · Random Forest · Logistic Regression · Streamlit'
    '</div>',
    unsafe_allow_html=True
)
