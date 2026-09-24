"""
App Streamlit - Prédiction du score de performance d'un élève (Student_Performance)

Fichiers attendus dans le MÊME dossier que ce script (générés par le notebook
Reg_Multiple_Linear_Ridge_Lasso_Decision_Tree_complet.ipynb, section "Tester le modèle
le plus performant") :
    - best_model.joblib   (le meilleur modèle, quel que soit son algorithme réel)
    - scaler.joblib
    - encoders.joblib

Lancer en local :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
import os

st.set_page_config(page_title="Prédiction score étudiant", page_icon="📚")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = jb.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = jb.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoder = jb.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    return model, scaler, encoder


try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(
        "Fichiers .joblib introuvables dans le dossier de l'app. "
        "Copie best_model.joblib, scaler.joblib et encoders.joblib "
        "à côté de app.py, puis relance.\n\nDétail : " + str(e)
    )
    st.stop()

FEATURE_ORDER = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
                  "Sleep Hours", "Sample Question Papers Practiced"]

st.title("📚 Prédire le score de performance d'un élève")
st.caption("Modèle entrainé sur le dataset Student Performance (score sur 100).")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction par fichier CSV"])

# ---------------------------------------------------------------------------
# Prédiction simple
# ---------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        col1, col2 = st.columns(2)
        with col1:
            hours = st.number_input("Heures étudiées (par jour)", min_value=0, max_value=12, value=5, step=1)
            prev_score = st.number_input("Score précédent (sur 100)", min_value=0, max_value=100, value=70, step=1)
            extra = st.selectbox("Activités extrascolaires", ["Yes", "No"])
        with col2:
            sleep = st.number_input("Heures de sommeil", min_value=0, max_value=12, value=7, step=1)
            papers = st.number_input("Nb de sujets d'entrainement pratiqués", min_value=0, max_value=20, value=5, step=1)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        extra_enc = encoder.transform([extra])[0]
        x_new = np.array([[hours, prev_score, extra_enc, sleep, papers]])
        x_new = scaler.transform(x_new)

        y_pred = model.predict(x_new)[0]
        y_pred = float(np.clip(y_pred, 0, 100))

        st.success(f"Score de performance prédit : **{y_pred:.1f} / 100**")

# ---------------------------------------------------------------------------
# Prédiction par lot (CSV)
# ---------------------------------------------------------------------------
with tab2:
    st.write("Le fichier doit contenir les colonnes suivantes, dans cet ordre : `" + ", ".join(FEATURE_ORDER) + "`.")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        try:
            df_enc = df_in.copy()
            df_enc["Extracurricular Activities"] = encoder.transform(df_enc["Extracurricular Activities"])

            x_batch = df_enc[FEATURE_ORDER].values
            x_batch = scaler.transform(x_batch)
            preds = model.predict(x_batch)
            preds = np.clip(preds, 0, 100)

            df_out = df_in.copy()
            df_out["Performance_Index_predit"] = preds
            st.dataframe(df_out)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name="predictions_student_performance.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur pendant le traitement du fichier : {e}")
