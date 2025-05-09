import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("spotify_data_similarity.csv", index_col=0, skiprows=lambda i: i > 0 and i % 5000 != 0)
    return df


st.set_page_config(page_title="HitPredict 🎶", layout="wide", page_icon="favicon.png")
st.logo("Logo.png", size="large")


st.title("🔍 Deep-dive - Über HitPredict")

st.markdown("""
**HitPredict** ist eine datengetriebene Applikation, die es Künstlern und Produzenten ermöglicht, das **Erfolgspotenzial** 
eines Songs objektiv einzuschätzen – noch bevor dieser offiziell veröffentlicht wird!  
Im Zentrum des Tools stehen moderne Methoden der **Audio- und Textanalyse** sowie **maschinelles Lernen** und **APIs**.

---
""")

st.subheader("🗃️ Unser Datenset")

st.markdown("""
Unsere Vorhersage basiert auf einem [Kaggle Datensatz](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks) 
von ursprünglich über **1.160.000 Songs**, die über die [**Spotify Web API**](https://developer.spotify.com/documentation/web-api) 
abgerufen wurden.

Die Songs enthalten von Spotify berechnete Attribute, unter anderem
- **Popularity**: Beliebtheit des Songs, errechnet durch Aufrufe und Alter.
- **Year**: Jahr des Releases. Es sind ausschliesslich Songs von 2000-2023 enthalten.
- **Genre**: Genre des Songs. Alle Genres, die nicht sowohl von Essentia als auch der Spotify Web API erkannt werden, sind 'other'.
- **Danceability**: Tanzbarkeit basierend auf einer Kombination von Tempo, Rhythmus, Taktstärke und Regelmässigkeit.
- **Energy**: Mass für Intensität und Aktivität. Typischerweise fühlen sich energiegeladene Tracks schnell, laut und geräuschvoll an.
- **Key**: Tonart, in der sich der Track befindet. Verwendet Pitch Class notation.
- **Loudness**: Gesamtlautstärke eines Songs in Dezibel (dB).
- **Mode**: Modalität (Dur oder Moll) eines Songs, also die Art der Tonleiter.
- **Speechiness**: Vorhandensein von gesprochenen Wörtern in einem Track.
- **Acousticness**: Konfidenzniveau das angibt, ob der Track rein akustisch ist.
- **Instrumentalness**: Zeigt an, ob ein Song keinen Gesang enthält.
- **Valence**: Positivität des Songs. Hohe Valenz klingt glücklich und euphorisch, während niedrige Valenz eher traurig und wütend ist.
- **Tempo**: Das geschätzte Gesamttempo eines Tracks in Beats pro Minute (BPM).
- **Time signature**: Geschätzte Taktart. Gibt an, wie viele Schläge ein Takt hat.
- **Duration**: Dauer des Tracks in Millisekunden.

Um zusätzlich auch textbasierte Attribute zu verwenden, haben wir die **Lyrics** für alle 1.160.000 Songs über die 
[**LRCLIB API**](https://lrclib.net/) abgerufen. Für **560.000 Songs** war dies erfolgreich, alle anderen haben wir aussortiert.

Anhand der Lyrics konnten wir folgende **textbasierte Metriken** berechnen:
- **Repetition**: Wiederholungen basierend auf 3er N-Grams, also die Anzahl an identischen 3-Wort-Abschnitten.
- **Readability**: Lesbarkeit nach der Dale–Chall Formel. Mass für die Verständnisschwierigkeiten beim Lesen eines Textes.
- **Polarity**: Polarität der Lyrics. Beschreibt positive oder negative Stimmung.
- **Subjectivity**: Subjektivität misst den Anteil der persönlichen Meinung.
- **Explicitness**: Obszönität oder anstössige Sprache in den Lyrics.

So entstand unser finaler Datensatz für das Machine Learning *(Auszug)*:
""")

df = load_dataset()
st.dataframe(df)

st.divider()


st.subheader("🧠 Machine Learning Modell")
st.markdown("""Um Machine Learning Modelle zu trainieren, haben wir zunächst einige Ausreisser entfernt, 
wie bspw. Songs mit einer Time signature von null. Ausserdem haben wir One-Hot Encoding für die Genres angewendet, 
damit diese als numerische Werte für das Modell lesbar sind. Zudem wurden alle Werte mit dem scikit-learn Standardscaler standardisiert.

- **X**: Alle Metriken exkl. Popularity
- **Y**: Popularity Score (0–100)

Folgende Machine Learning Modelle haben wir trainiert (teils mit Hyperparameter-Tuning):
- **Lineare Regression** (mit **Rigde** und **Lasso** Regularisation)
- **Random Forest**
- **XGBoost**
- **Neuronales Netzwerk**
""")

ml_results = pd.DataFrame({
    "Lineare Regression": ["12.34", "9.73"],
    "LinReg Ridge": ["12.25", "9.69"],
    "LinReg Lasso": ["12.25", "9.69"],
    "Random Forest": ["11.44", "8.99"],
    "**XGBoost**": ["**11.35**", "**8.91**"],
    "Neural Network": ["11.42", "9.04"],
    },
index=["RMSE", "MAE"]
)
st.table(ml_results)

st.markdown("Da XGBoost die besten Resultate zeigte, haben wir das Modell für HitPredict ausgesucht.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("❗ Wichtigkeit der Features")
    st.image("img/XGBoost Feature Importance.png")

    st.markdown("""
    Anhand der **Feature-Importance** erkennt man, welche Metriken bei der Popularity-Vorhersage am wichtigsten sind.
    
    - Genre
    - Year
    - Instrumentalness
    - Danceability
    
    **Achtung**: Dies zeigt lediglich, wie stark die einzelnen Metriken berücksichtigt werden.
    Über einen positiven/negativen Einfluss kann keine Aussage gemacht werden.
    """)

with col2:
    st.subheader("🎯 Genauigkeit unseres Modells")
    st.image("img/XGBoost Accuracy.png")

    st.markdown("""
    Die **Genauigkeit** unseres Modells haben wir ebenfalls visualisiert.
    
    Hier erkennt man, dass das Modell **niedrige Werte überschätzt**, während es **hohe Werte überschätzt**.
    Das Modell bewegt sich so im Zweifel weg von den Extremen, um hohe Fehler zu vermeiden.
    
    Zudem nimmt die **Streuung** zu, je höher der tatsächliche Popularity Score ist.
    Dies liegt daran, dass gerade einmal 3% der Songs einen Score über 60 haben und es somit wenig Trainingsdaten für hohe Scores gibt.
    
    Der **strikt steigende Verlauf** lässt jedoch erkennen, dass das Modell bei der Vorhersage klar einen **Trend** identifizieren kann.
    """)

st.divider()


st.subheader("📊 Analyseprozess")

st.markdown("""
Sobald ein Nutzer eine **Audiodatei** – im MP3-Format – auf unserer Plattform hochlädt, extrahiert die Python Library 
**Essentia** alle oben genannten Metriken aus dem Audio.

[**Essentia**](https://essentia.upf.edu/essentia_python_tutorial.html) ist eine open-source C++ Library, 
die von der Universität Pompeu Fabra (UPF) für die **Audioanalyse** entwickelt wurde.
Für Essentia existieren Python-Bindings und eine Tensorflow-Erweiterung mit vortrainierten Machine Learning Modellen.
Essentia berechnet die Metriken wie folgt:
 
**Standard-Essentia**
- Duration, Tempo, Key, Mode, Loudness, Energy, Danceability und Time signature

**Tensorflow-Erweiterung**
- Genre, Acousticness, Instrumentalness, Speechiness und Valence


Anschliessend wird mit **OpenAI Whisper** – das über die [**Deepgram API**](https://developers.deepgram.com/docs/deepgram-whisper-cloud) 
läuft – eine automatische Transkription der **Lyrics** durchgeführt.

Die textbasierten Metriken werden mit den selben Methoden berechnet, wie für den Datensatz zuvor.
- [**TextBlob**](https://textblob.readthedocs.io/en/dev/): Repetition, Polarity, Subjectivity
- [**Textstat**](https://textstat.org/): Readability
- [**profanity-check**](https://github.com/vzhou842/profanity-check): Explicitness

Nun erfolgt die Vorhersage des Popularity Scores anhand des trainierten XGBoost Modells.
""")

st.divider()

st.subheader("📈 Ergebnisausgabe")

col1, col2 = st.columns([0.6,0.4])

with col1:
    st.markdown("""
    Nach der Analyse liefert HitPredict eine Popularitätsschätzung von **0 bis 100 Punkten**:
    
    - **< 30 Punkte** → kein Hitpotenzial
    - **30–50 Punkte** (Top 40%) → durchschnittliches Potenzial
    - **50–60 Punkte** (Top 9%) → Wahrscheinlichkeit für einen Hit steigt deutlich
    - **60+ Punkte** (Top 3%) → hohe Wahrscheinlichkeit für viralen Erfolg
    
    Ein **Netzdiagramm** zeigt anschliessend, welche Metriken verbessert werden können, 
    um die Erfolgschancen eines Songs zu steigern.
    """)

with col2:
    st.image("img/Popularity Score Distribution.png")

st.divider()

st.subheader("👫 Song Soulmate")

st.markdown("""
Naheliegend wäre doch auch, die berechneten Metriken mit jedem Song des Datensatzes zu **vergleichen**, 
um einen Song zu identifizieren, der dem hochgeladenen Audio am ähnlichsten ist.
– Und genauso machen wir es!

Wir berechnen dafür die **geometrische Distanz** im 44-dimensionalen Raum zwischen dem Vektor der standardisierten Metriken 
des hochgeladenen Audios und den jeweiligen Vektoren aller anderen Songs im Datensatz.
Der niedrigste Wert wird als **bestes Match** identifiziert.

Mittels einer Anfrage an die **Spotify Web API** wird die aktuelle Popularity des Matches abgerufen und zusammen 
mit einem abspielbaren Widget angezeigt.
""")

st.divider()

st.subheader("🎯 Unser Ziel")

st.markdown("""
Mit HitPredict bieten wir Künstlerinnen und Künstlern ein Tool, das datenbasiert kreative und geschäftliche Entscheidungen unterstützt.  
Ob bei der Auswahl einer Single, der Einschätzung von Marketingbudgets oder der Priorisierung von Albumtracks –  
**HitPredict liefert eine fundierte Zweitmeinung auf Basis realer Musik- und Textdaten.**

#### HitPredict: *Know your hit before it’s heard.*
""")

