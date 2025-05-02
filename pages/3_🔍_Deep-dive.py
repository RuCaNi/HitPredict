import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("spotify_data_similarity.csv", index_col=0, skiprows=lambda i: i > 0 and i % 5000 != 0)
    return df


st.set_page_config(page_title="HitPredict 🎶", layout="wide")
st.logo("Logo.png", size="large")

st.title("🔍 Deep-dive - Über HitPredict")

st.markdown("""
HitPredict ist eine datengetriebene Applikation, die es Künstlern und Produzenten ermöglicht, das Erfolgspotenzial eines Songs objektiv einzuschätzen – noch bevor dieser offiziell veröffentlicht wird!  
Im Zentrum des Tools stehen moderne Methoden der Audio- und Textanalyse sowie maschinelles Lernen verbunden mit APIs.

---
""")

st.subheader("📊 Analyseprozess")

st.markdown("""
Der Analyseprozess beginnt, sobald ein Nutzer eine Audiodatei – im MP3-Format – auf unserer Plattform hochlädt.  
Im ersten Schritt werden eine Vielzahl von Audioeigenschaften extrahiert:  
- Tanzbarkeit, Energie, Schlüssel, Lautstärke, Tonart
- Sprachlichkeit, Akustizität, Instrumentalität
- Wertigkeit, Tempo, Dauer, Taktart
- Wiederholung, Lesbarkeit, Stimmung, Eindeutigkeit

Die Extraktion erfolgt automatisiert über spezialisierte Audioanalyse-Frameworks wie **Essentia**.

Parallel dazu wird mit **OpenAI Whisper** eine automatische Transkription des Songtexts durchgeführt.  
Die gewonnenen Lyrics werden analysiert hinsichtlich:
- Wortvielfalt (Vocabulary Diversity)
- Lesbarkeit (Readability)
- Emotionales Sentiment (Positivity/Negativity) mittels TextBlob
- Jugendfreiheit (Profanity Analyse)
- Repetitivität
- Sprachliche Komplexität

---
""")

st.subheader("🗃️ Unser Datenset")

df = load_dataset()
st.dataframe(df)

st.divider()

st.subheader("🧠 Machine Learning Modell")

st.markdown("""
Die rund 20 extrahierten Audio- und Textfeatures werden an ein Machine-Learning-Modell übergeben:  
- **XGBoost (Extreme Gradient Boosting)** – robustes Verfahren speziell für Regressionsaufgaben
- Trainiert auf ca. **560.000 Spotify-Songs** (2000–2023)
- Zielwert: **Spotify Popularity Score** (Skala 0–100)

Starke Prädiktoren:
- Genre
- Instrumentalität
- Tanzbarkeit
- Readability
- Sentiment

Eine Visualisierung der Feature-Importance zeigt nachvollziehbar, worauf das Modell bei der Popularitätsschätzung achtet.

""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("❗ Wichtigkeit der Features")
    st.image("img/Feature Importance XGBoost.png", width=750)

with col2:
    st.subheader("🎯 Genauigkeit unseres Modells")
    st.image("img/XGBoost Accuracy.png", width=600)

st.divider()

st.subheader("📈 Ergebnisausgabe")

st.markdown("""
Nach der Analyse liefert HitPredict eine Popularitätsschätzung von **0 bis 100 Punkten**:

- **< 30 Punkte** → kein Hitpotenzial
- **30–50 Punkte** → durchschnittliches Potenzial
- **50–60+ Punkte** → Wahrscheinlichkeit für einen Hit steigt deutlich
- **60+ Punkte** → Top 3% der Songs – hohe Wahrscheinlichkeit für viralen Erfolg

Ein **Netzdiagramm** zeigt, welche Metriken verbessert werden können, um die Erfolgschancen eines Songs zu steigern.

---
""")

st.subheader("🤖 Technologien hinter HitPredict")

st.markdown("""
- **Python** für Programmierung
- **Streamlit** als Benutzeroberfläche
- **Whisper** für Transkription
- **Essentia** für Audioanalyse
- **TextBlob**, **textstat** für Textanalysen
- **scikit-learn**, **XGBoost** für Machine Learning

---
""")

st.subheader("🎯 Unser Ziel")

st.markdown("""
Mit HitPredict bieten wir Künstlerinnen und Künstlern ein Tool, das datenbasiert kreative und geschäftliche Entscheidungen unterstützt.  
Ob bei der Auswahl einer Single, der Einschätzung von Marketingbudgets oder der Priorisierung von Albumtracks –  
**HitPredict liefert eine fundierte Zweitmeinung auf Basis realer Musik- und Textdaten.**

---
""")

