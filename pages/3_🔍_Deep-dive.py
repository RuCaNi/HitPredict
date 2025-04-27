import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("spotify_data_similarity.csv", index_col=0, skiprows=lambda i: i > 0 and i % 5000 != 0)
    return df


st.set_page_config(page_title="HitPredict 🎶", layout="wide")
st.logo("Logo.png", size="large")

st.title("🔍 Deep-dive in unsere Daten")

df = load_dataset()

st.subheader("🗃️ Unser Datenset")
st.dataframe(df)

st.markdown("---")

col1, placeholder, col2 = st.columns([3,1,3])

with col1:
    st.subheader("Wichtigkeit der Features")
    st.image("img/Feature Importance XGBoost.png")

with col2:
    st.subheader("Genauigkeit unseres Modells")
    st.image("img/XGBoost Accuracy.png")