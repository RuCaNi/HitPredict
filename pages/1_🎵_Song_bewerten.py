#
# HitPredict - Ein Projekt für 'Grundlagen und Methoden der Informatik für Wirtschaftswissenschaften' an der Universität St.Gallen (2025)
# Autoren: Ruben Cardell, Adam Bisharat, Helena Häußler, Colin Wirth
# ---
# HINWEIS: Dies ist das Herz des Projektes
# ---
# ACHTUNG: Installation
# 1. Es müssen alle Libraries in exakt der richtigen Version aus requirements.txt installiert werden
# 2. Die App benötigt Python 3.11
# 3. Die Essentia Modelldateien müssen im gleichen Verzeichnis wie die Landingpage.py Datei sein
# 4. Es kann Fehler beim Ausführen geben, wenn die Packages mit einer falschen Numpy Version kompiliert werden
# !!! Wir empfehlen, die App auf https://hitpredict.streamlit.app/ anzuschauen !!!
#

# Importe für Berechnungen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import essentia.standard as es
from sklearn.metrics.pairwise import euclidean_distances

# Streamlit und zugehörig
import streamlit as st
import streamlit.components.v1 as components
import tempfile

# APIs
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepgram import DeepgramClient, PrerecordedOptions

# Textanalyse
import textstat
from textblob import TextBlob
from collections import Counter
from profanity_check import predict_prob

from nltk.data import find
from nltk import download


# Das NLTK-Paket wird für die Textanalyse der Lyrics benötigt
def ensure_punkt():
    try:
        find('tokenizers/punkt_tab') # Vermeidet, bei jedem Rerun das Paket neu zu downloaden
    except LookupError:
        download('punkt_tab') # Nur Download, wenn Paket noch nicht gefunden wurde

ensure_punkt()


def load_audio(filename):
    """
    Konvertiert Audio in ein numerisches Array und extrahiert die Sample Rate.
    Grundlage für weitere Analysen.
    """
    loader = es.MonoLoader(filename=filename)  # Initialize loader

    audio = loader()
    sample_rate = 44100
    return audio, sample_rate


def load_model(audio):
    """
    Extrahiert sogenannte Embeddings, basierend auf einem vortrainierten ML Modell MusiCNN.
    Die MusiCNN Embeddings sind 200-dimensionale Vektoren für jedes Audiosegment und sollen Aspekte
    wie Genre, Stimmung, Instrumentierung und Rhythmus erfassen.
    Alle weiteren Berechnungen, die wir anhand vortrainierter ML-Algorithmen durchführen, benutzen die MusiCNN Embeddings.
    Quelle: https://essentia.upf.edu/tutorial_tensorflow_auto-tagging_classification_embeddings.html
    """
    embedding_model = es.TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")

    embeddings = embedding_model(audio)
    return embeddings


def compute_duration(audio, sample_rate):
    """
    Berechnet die Länge in Millisekunden.
    """
    duration = (len(audio) / sample_rate) * 1000

    return int(duration)


def compute_genre(embeddings):
    """
    Berechnet das Genre mit vortrainiertem ML-Modell.
    Alle Genres, die nicht sowohl von Essentia als auch der Spotify Web API erkannt werden, sind 'other'.
    """
    model = es.TensorflowPredict2D(graphFilename="msd-msd-musicnn-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
    predictions = model(embeddings)

    # Zuordnung der Wahrscheinlichkeiten von Essentia zum Namen des Genres basierend auf dem Index in der Liste
    mapping = ["rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz",
    "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica", "80s",
    "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard-rock", "ambient", "acoustic", "experimental",
    "female vocalist", "guitar", "hip-hop", "70s", "party", "country", "easy listening", "sexy", "catchy", "funk", "electro",
    "heavy-metal", "Progressive rock", "60s", "rnb", "indie-pop", "sad", "house", "happy"]

    # Es dürfen nur Genres vorhanden sein, die sowohl von Essentia extrahiert werden als auch im Dataset von Spotify  vorhanden sind
    common_genres = ['pop', 'acoustic', 'jazz', 'dance', 'electronic', 'funk', 'house', 'heavy-metal', 'indie-pop', 'folk', 'hard-rock',
                     'blues', 'hip-hop', 'rock', 'chill', 'metal', 'ambient', 'party', 'soul', 'guitar', 'punk', 'electro', 'sad', 'country']

    avg_probs = np.mean(predictions, axis=0)  # Die Wahrscheinlichkeiten werden pro Segment im Audio ermittelt, wir nehmen den Durchschnitt

    # Essentia ermittelt auch Wahrscheinlichkeiten für Genres, die nicht in dem Dataset von Spotify vorhanden sind. Diese setzen wir auf Null
    for i in range(len(mapping)):
        if mapping[i] not in common_genres:
            avg_probs[i] = 0.0

    # Manche Genres werden immer über- / untergewichtet. Manuelles Feintuning nötig
    avg_probs[0] *= 0.2  # Übergewicht bei 'rock'
    avg_probs[4] *= 0.2  # Übergewicht bei 'electronic'
    avg_probs[9] *= 0.5  # Übergewicht bei 'jazz'
    avg_probs[11] *= 0.2  # Übergewicht bei 'metal'
    avg_probs[28] *= 0.2  # Übergewicht bei 'ambient'
    avg_probs[1] *= 2  # Untergewicht bei 'pop'

    genre = mapping[np.argmax(avg_probs)]  # Das Genre der höchsten Wahrscheinlichkeit wird dem Namen zugeordnet

    if genre not in common_genres:
      genre = "other"  # Falls das Genre nicht sowohl Spotify als auch Essentia bekannt ist, wird es als 'other' gesetzt

    return genre


def compute_rhythm_features(audio):
    """
    Extrahiert das Tempo und die Beats aus dem Audiosignal.
    Grundlage für die Berechnung der Taktart.
    """
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

    tempo, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    return tempo, beats


def compute_key_and_mode(audio):
    """
    Ermittelt die Tonart (Key) und den Modus (Dur/Moll).
    Verwendet Pitch Class Notation.
    """
    key_extractor = es.KeyExtractor()
    key, scale, _ = key_extractor(audio)

    mapping = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
        "A#": 10, "Bb": 10, "B": 11}
    key = mapping.get(key, -1)  # Tonartbezeichnung in Integer (gemäss Datensatz) umwandeln

    if scale == 'major':
        mode = 1  # Dur
    else:
        mode = 0  # Moll

    return key, mode


def compute_loudness(audio):
    """
    Berechnet die Lautheit (Loudness) des Audios gemäss EBU R128 Empfehlung.
    Gesamtlautstärke des Songs in Dezibel (dB).
    """
    if len(audio.shape) == 1:
        stereo_audio = np.column_stack((audio, audio))  # Mono in Stereo duplizieren

    loudness_extractor = es.LoudnessEBUR128()
    loudness = loudness_extractor(stereo_audio)[2]  # Lautheit in dB
    return loudness


def compute_energy(audio):
    """
    Berechnet die normalisierte Energie (RMS) über die Lautstärke des Audiosignals.
    Energiegeladene Tracks fühlen sich schnell, laut und geräuschvoll an.
    """
    rms_val = es.RMS()(audio)

    return np.clip(rms_val / 0.5, 0, 1)  # Auf Bereich [0,1] normieren


def compute_danceability(audio, sample_rate):
    """
    Ermittelt die Tanzbarkeit basierend auf Essentia's Algorithmus.
    Typischerweise eine Kombination von Tempo, Rhythmus, Taktstärke und Regelmässigkeit.
    """
    danceability = es.Danceability(sampleRate=sample_rate)(audio)[0] / 3  # Werte von 0-3 auf [0,1] skalieren

    return np.clip(danceability, 0, 1)


def compute_acousticness(embeddings):
    """
    Berechnet die Acousticness anhand eines vortrainierten Modells.
    Konfidenzniveau das angibt, ob der Track rein akustisch ist.
    """
    model = es.TensorflowPredict2D(graphFilename="mood_acoustic-msd-musicnn-1.pb", output="model/Softmax")
    predictions = model(embeddings)

    acousticness = np.mean(predictions[:, 0])  # Durchschnitt der Wahrscheinlichkeiten
    return acousticness


def compute_instrumental_and_speech(embeddings):
    """
    Ermittelt Instrumentalness und Speechiness per vortrainiertem Modell.
    Instrumentalness: Zeigt an, ob ein Song keinen Gesang enthält.
    Speechiness: Vorhandensein von gesprochenen Wörtern in einem Track.
    """
    model = es.TensorflowPredict2D(graphFilename="voice_instrumental-msd-musicnn-1.pb", output="model/Softmax")
    predictions = model(embeddings)

    instrumental = np.mean(predictions[:, 0])  # Durchschnitt der Wahrscheinlichkeiten für Instrumentalness
    speech = np.mean(predictions[:, 1])  # Durchschnitt der Wahrscheinlichkeiten für Speechiness
    return instrumental, speech


def compute_valence(embeddings):
    """
    Berechnet den Valence-Wert (Positivität des Songs).
    Hohe Valenz klingt glücklich und euphorisch, während niedrige Valenz eher traurig und wütend ist.
    """
    model = es.TensorflowPredict2D(graphFilename="deam-msd-musicnn-2.pb", output="model/Identity")
    predictions = model(embeddings)

    valence = (np.mean(predictions[:, 0]) - 1) / 8  # Zuerst von 1-9 auf 0-8 verschieben, dann auf [0,1] skalieren
    return valence


def compute_time_signature(audio, beats, sample_rate):
    """
    Ermittelt die Taktart basierend auf Beats und Loudness.
    Gibt an, wie viele Schläge ein Takt hat (z.B. 4/4).
    """
    beats_loudness = es.BeatsLoudness(beats=beats, sampleRate=sample_rate)
    loudness, loudness_band_ratio = beats_loudness(audio)
    beatogram_algo = es.Beatogram()
    beatogram = beatogram_algo(loudness, loudness_band_ratio)
    meter = es.Meter()
    time_signature = meter(beatogram)

    if time_signature % 2 == 0:
        time_signature = 4.0  # Alle geraden Taktzahlen auf 4/4 setzen, gemäss Spotify Datensatz
    return time_signature


def transcribe_lyrics(filename):
    """
    Transkribiert Songtexte via Deepgram-API und OpenAI Whisper.
    """
    try:
        deepgram = DeepgramClient(st.secrets["DG_TOKEN"])  # API-Key aus .streamlit/secrets.toml

        with open(filename, 'rb') as buffer_data:  # API-Aufruf nach Deepgram-Dokumentation
            payload = {'buffer': buffer_data}
            options = PrerecordedOptions(smart_format=True, model="whisper")

            response = deepgram.listen.rest.v('1').transcribe_file(payload, options, timeout=600)

        transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]

    except:  # Error durch fehlerhafte API-Anfrage oder keine Lyrics erkannt
        st.error("Die Lyrics konnten nicht transkribiert werden. Der Song wird als Instrumental ausgewertet.")
        transcription = "False"

    return transcription


def ngram_repetition(text):
    """
    Berechnet die Repetition anhand von 3er N-Grammen im Text.
    Verhältnis der Anzahl an identischen Trigrammen zur Gesamtzahl von Trigrammen.
    """
    blob = TextBlob(text)
    ngrams = blob.ngrams(3)  # Trigramme erzeugen

    if not ngrams:
        return 0.0

    ngram_counts = Counter(map(tuple, ngrams))  # Anzahl der Trigramme
    most_common_count = max(ngram_counts.values())  # Häufigstes identisches Trigramm
    total_ngrams = sum(ngram_counts.values())  # Gesamtzahl aller erzeugten Trigramme
    return most_common_count / total_ngrams  # Verhältnis des häufigsten Trigramms zur Gesamtzahl der Trigramme


def readability_check(text):
    """
    Ermittelt die Lesbarkeit des Textes via Dale-Chall-Formel.
    Mass für die Verständnisschwierigkeiten beim Lesen eines Textes.
    """
    readability = textstat.dale_chall_readability_score(text)

    if readability > 30:
        readability = 30.0  # Obergrenze setzen
    return readability


def analyze_sentiment(text):
    """
    Bestimmt Polarität und Subjektivität des Textes.
    Polarität: Beschreibt positive oder negative Stimmung.
    Subjektivität: Anteil der persönlichen Meinung.
    """
    blob = TextBlob(text)

    return blob.sentiment.polarity, blob.sentiment.subjectivity  # Polarität und Subjektivität


def explicitness_check(text):
    """
    Berechnet die Wahrscheinlichkeit für explizite Inhalte im Text.
    Vorhandensein von Obszönität oder anstössiger Sprache.
    """
    explicitness = predict_prob([text])

    return explicitness[0]


@st.cache_data(show_spinner=False)
def extract_features(filename):
    """
    Extrahiert alle Audio- und Textmerkmale und gibt ein DataFrame zurück.
    """
    st.write("*Probiere doch unser [Musik-Quiz](Musik-Quiz), während du wartest?*")

    with st.status("Analysiere Audio...", expanded=True) as status:
        st.write("Lade Audio...")
        audio, sample_rate = load_audio(filename)
        st.write("Bereite Machine Learning vor...")
        embeddings = load_model(audio)
        st.write("Berechne Audio-Metriken...")
        duration_ms = compute_duration(audio, sample_rate)
        genre = compute_genre(embeddings)
        tempo, beats = compute_rhythm_features(audio)
        key, mode = compute_key_and_mode(audio)
        loudness = compute_loudness(audio)
        energy = compute_energy(audio)
        danceability = compute_danceability(audio, sample_rate)
        acousticness = compute_acousticness(embeddings)
        instrumentalness, speechiness = compute_instrumental_and_speech(embeddings)
        valence = compute_valence(embeddings)
        time_signature = compute_time_signature(audio, beats, sample_rate)
        st.write("Transkribiere Songtexte... (via API, ∼1 Minute)")
        lyrics = transcribe_lyrics(filename)
        st.write("Berechne textbasierte Metriken...")
        repetition = ngram_repetition(lyrics)
        readability = readability_check(lyrics)
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(lyrics)
        explicitness = explicitness_check(lyrics)

        df = pd.DataFrame([{
            'year': 2023,
            'danceability': danceability,
            'energy': energy,
            'key': key,
            'loudness': loudness,
            'mode': mode,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'valence': valence,
            'tempo': tempo,
            'duration_ms': duration_ms,
            'time_signature': time_signature,
            'repetition': repetition,
            'readability': readability,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity,
            'explicitness': explicitness,
            'genre_acoustic': False,
            'genre_ambient': False,
            'genre_blues': False,
            'genre_chill': False,
            'genre_country': False,
            'genre_dance': False,
            'genre_electro': False,
            'genre_electronic': False,
            'genre_folk': False,
            'genre_funk': False,
            'genre_guitar': False,
            'genre_hard-rock': False,
            'genre_heavy-metal': False,
            'genre_hip-hop': False,
            'genre_house': False,
            'genre_indie-pop': False,
            'genre_jazz': False,
            'genre_metal': False,
            'genre_other': False,
            'genre_party': False,
            'genre_pop': False,
            'genre_punk': False,
            'genre_rock': False,
            'genre_sad': False,
            'genre_soul': False
            }])

        df[f"genre_{genre}"] = True  # Das erkannte Genre wird One-Hot-encoded auf True gesetzt

        status.update(label="Analyse erfolgreich!", state="complete", expanded=False)

    return df


@st.cache_data(show_spinner=False)
def predict_popularity(df):
    """
    Vorhersage der Popularität mit dem XGBoost-Modell, das anhand des Spotify Datensatzes trainiert wurde.
    """
    with open("xgboost_v5.pkl", "rb") as file:
        model = pickle.load(file)
    with open("X_scaler.pkl", "rb") as file:
        X_scaler = pickle.load(file)
    with open("y_scaler.pkl", "rb") as file:
        y_scaler = pickle.load(file)

    df = X_scaler.transform(df)  # Features skalieren

    y_pred = model.predict(df)  # Popularity vorhersagen
    y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Rückskalierung der Popularity
    return y_pred_rescaled


@st.cache_data(show_spinner=False)
def get_track_info(track_id):
    """
    Initialisiert Spotify-Client und ruft Informationen zu einem Track ab.
    """
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=st.secrets["SP_CLIENT_ID"],  # Spotify Client-ID aus .streamlit/secrets.toml
        client_secret=st.secrets["SP_CLIENT_SECRET"]  # Spotify Client-Secret aus .streamlit/secrets.toml
    ))

    track = sp.track(track_id)  # API-Aufruf

    # Rückgabe als Python Dictionary
    return {'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            'album_release': track['album']['release_date'],
            'album_cover_url': track['album']['images'][0]['url'],
            'popularity': track['popularity']
            }


@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Lädt den Datensatz für die Similarity-Analyse (Song-Soulmate).
    """
    df = pd.read_csv("spotify_data_similarity.csv", index_col=0)
    return df


@st.cache_data(show_spinner=False)
def get_soulmate(X_pred, y_pred):
    """
    Bestimmt den ähnlichsten Song anhand der geometrischen Distanz aller Metriken.
    """
    df = load_dataset()
    data = pd.get_dummies(df, columns=['genre'])  # Extrahierte Metriken sind auch One-Hot-encoded

    # Für gleiche Gewichtung bei der geometrischen Distanz sollten alle Daten skaliert sein
    with open("X_scaler.pkl", "rb") as f:
        X_scaler = pickle.load(f)
    with open("y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # Spalten für (unterschiedliche) Skalierung vorbereiten
    X_cols = [col for col in data.columns if col not in ['track_id', 'popularity']]
    y_cols = ['popularity']

    # Datensatz skalieren
    X_scaled = X_scaler.transform(data[X_cols])
    y_scaled = y_scaler.transform(data[y_cols])

    data_scaled = np.hstack([y_scaled, X_scaled])

    # Extrahierte Metriken skalieren
    X_pred_scaled = X_scaler.transform(pd.DataFrame(X_pred, columns=X_cols))
    y_pred_scaled = y_scaler.transform(pd.DataFrame(y_pred, columns=y_cols))

    pred_scaled = np.hstack([y_pred_scaled, X_pred_scaled])

    # Geometrische Distanz (44-dimensionaler Raum) der extrahierten Metriken zu jedem Song im Datensatz
    distances = euclidean_distances(pred_scaled, data_scaled)
    closest_idx = distances.argmin()  # Geringste Distanz = stärkste Ähnlichkeit der Metriken

    soulmate_id = data.iloc[closest_idx]['track_id']  # Spotify Track-ID des Songs zurückgeben
    return soulmate_id


# --- STREAMLIT WEBAPP ---
# Konfiguration
st.set_page_config(page_title="HitPredict 🎶", layout="wide", page_icon="favicon.png")
st.logo("Logo.png", size="large")

st.title("🎤 Lade deinen Song hoch")

# Audio hochladen
uploaded_file = st.file_uploader("🎶 Audio hochladen (MP3)", type=["mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    # Streamlit gibt ein BytesIO Objekt zurück, Essentia benötigt einen Dateipfad
    # Daher speichern wir das BytesIO Objekt als temporäre Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        filename = temp_file.name

    # Die extrahierten Werte werden im Session_State gespeichert und nach Seitenwechsel direkt wieder angezeigt
    # Sind bereits extrahierte Metriken im Session_State, werden sie überschrieben
    st.session_state.df = extract_features(filename)


if "df" in st.session_state:  # Ist noch kein Audio im Session_Sate, bleibt der Rest der Seite leer
    df = st.session_state.df  # Verlinkung für einfachere Referenz im Code

    st.divider()
    st.header("📈 Metriken der Songanalyse")

    col1, col2 = st.columns(2)  # Spalten

    with col2:
        # Ermitteltes Genre aus One-Hot-encoding identifizieren
        current_genre = "other"
        for column in df.columns:
            if column.startswith("genre_"):  # Nur Spalten die mit genre_ beginnen
                if df.at[0, column] == True:  # Genre das als True markiert wurde
                    current_genre = column.replace("genre_", "")
                    break

        st.metric(label="Genre", value=current_genre.capitalize())  # Genre anzeigen

        mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                   6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
        key = mapping.get(float(df["key"].iloc[0]), "C")  # Tonart aus Pitch-Class Notation wieder in Buchstaben zuordnen

        if int(df["mode"].iloc[0]) == 1:  # Namen anzeigen
            mode = "Dur"
        else:
            mode = "Moll"

        st.metric(label="Tonart", value=f"{key}-{mode}")  # Tonart und Modus anzeigen
        st.metric(label="Taktart", value=f'{int(df["time_signature"].iloc[0])}/4')  # Taktart anzeigen

        st.write("")

        # Popularität vorhersagen
        popularity = predict_popularity(df)
        popularity_score = float(popularity.item())

        st.header(f"✨ Popularity Score: {popularity_score:.1f} / 100 ✨")

        dataset = load_dataset()
        percentrank = (len(dataset[dataset['popularity'] >= popularity_score]) / len(dataset)) * 100

        # Feedback basierend auf Popularität
        if popularity_score <= 30:
            st.subheader(f"Top {percentrank:.1f}% (Kein Hit)")
            st.markdown("**Dein Song hat noch nicht das Zeug zum Hit – aber jeder Star hat mal klein angefangen! 🌱**")
            st.markdown("➔ Bleib dran und nutze das Feedback, um deinen Sound auf ein neues Level zu bringen. 🎛️")
        elif popularity_score <= 50:
            st.subheader(f"🛠️ Top {percentrank:.1f}% (Hit-Potenzial)")
            st.markdown("**Dein Song hat starke Ansätze – Feintuning an Tanzbarkeit oder Lyrics und du bist auf Kurs! 🚀**")
            st.markdown("➔ Manchmal reicht ein cleverer Refrain oder ein knackiger Beat, um Herzen zu erobern! 💓")
        elif popularity_score <= 60:
            st.subheader(f"🔥 Top {percentrank:.1f}% (Wahrscheinlicher Hit)")
            st.markdown("**Dein Song tanzt an der Schwelle zum Hit – ein Funken mehr und die Crowd wird explodieren! 🎉**")
            st.markdown("➔ Das Fundament ist stark, jetzt brauchst du nur noch den perfekten Feinschliff. 🛠️")
        else:
            st.subheader(f"🚀 Top {percentrank:.1f}% (Mega-Hit)")
            st.markdown("**Dein Track ist ein Volltreffer! Die Charts warten schon auf dich – let's go! 🔥🏆**")
            st.markdown("➔ Bleib fokussiert, bleib echt – Hits entstehen, wenn Herzblut auf Timing trifft. ❤️⏳")


    # Netzdiagramm erstellen
    with col1:
        # Neues Dataframe für Skalierung und Anpassen der Namen (Grossschreiben)
        # Die Werte skalieren auf das 75. Perzentil der Daten aus dem Datensatz, also 75% der Songs sind >= dem Wert
        # So lassen sich die Werte gut visualisieren und vergleichen
        plot_df = pd.DataFrame([{
                'Energy': np.clip(df['energy']/0.89, 0, 1),
                'Loudness': np.clip(1-0.1*(df['loudness']/-4.791), 0, 1),  # Andere Berechnung, da negativ
                'Danceability': np.clip(df['danceability'] / 0.664, 0, 1),
                'Speechiness': np.clip(df['speechiness']/0.75, 0, 1),  # Andere Berechnung, da gegensätzlich zu Instrumentalness
                'Acousticness': np.clip(df['acousticness']/0.453, 0, 1),
                'Instrumentalness': np.clip(df['instrumentalness']/0.75, 0, 1),  # Andere Berechnung, da gegensätzlich zu Speechiness
                'Valence': np.clip(df['valence']/0.656, 0, 1),
                'Tempo': np.clip(df['tempo']/141.057, 0, 1),
                'Duration': np.clip(df['duration_ms']/269747, 0, 1),
                'Repetition': np.clip(df['repetition']/0.043956, 0, 1),
                'Readability': np.clip(df['readability']/16.1, 0, 1),
                'Polarity': np.clip(df['sentiment_polarity']/0.15066, 0, 1),
                'Subjectivity': np.clip(df['sentiment_subjectivity']/0.6, 0, 1),
                'Explicitness': np.clip(df['explicitness']/0.288619, 0, 1)
        }])

        # Netzdiagramm mit Matplotlib
        # Adaptiert von https://python-graph-gallery.com/390-basic-radar-chart/
        labels = plot_df.columns
        values = plot_df.iloc[0].tolist()
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='#f8641b', linewidth=1.5)  # Primärfarbe
        ax.fill(angles, values, color='#f8641b', alpha=0.4)

        ax.set_ylim(0, 1)

        # Erweiterung: Namen sollen sich nicht mit dem Rand überschneiden
        label_distance = 1.15  # Abstandsmultiplikator
        for i in range(len(labels)):  # Jedes Label wird einzeln geplottet
            angle = angles[i]
            label = labels[i]
            ax.text(angle, label_distance, label, ha='center', va='center', fontsize=7)

        # y-Abschnitte beibehalten, aber Beschriftungen ausblenden
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([''] * len(labels))
        ax.set_yticklabels([])

        plt.tight_layout()
        st.pyplot(fig=plt, use_container_width=False)


    spotify_id = get_soulmate(df, popularity)  # Song-Soulmate identifizieren

    similar_song = get_track_info(spotify_id)  # Spotify Daten (Cover etc.) für Song-Soulmate abrufen

    st.divider()
    col1, col2 = st.columns([2, 1]) # Spaltengrösse 2/3 und 1/3
    with col1:
        st.header("👫 Song Soulmate")

        subcol1, subcol2, subcol3 = st.columns([3, 1, 1])  # Song und Metriken anzeigen
        subcol1.subheader(f'{similar_song["artist_name"]} - {similar_song["track_name"]}')
        subcol2.metric(label='Popularity', value=similar_song["popularity"])
        subcol3.metric(label='Release', value=similar_song["album_release"][:4])

        components.iframe(f"https://open.spotify.com/embed/track/{spotify_id}", height=80)  # Abspielbares Spotify-Widget

    col2.write("")
    col2.write("")
    col2.write("")
    col2.image(similar_song["album_cover_url"], width=200)  # Album-Cover