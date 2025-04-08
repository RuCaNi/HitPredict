### IMPORTANT: INSTALL NECESSARY LIBRARIES ###
### COMMAND: pip install numpy==1.26.1 essentia-tensorflow=="2.1b6.dev1110" deepgram-sdk==3.10.1 textstat textblob==0.17.1 alt-profanity-check==1.3.2 scikit-learn==1.3.2 xgboost==2.1.4

### MAKE SURE THAT MODEL AND AUDIO FILES ARE PLACED IN THE SAME FOLDER AS THE .PY ###


import numpy as np
import pandas as pd
import pickle
import essentia
import essentia.standard as es

import streamlit as st
import tempfile
from deepgram import DeepgramClient, PrerecordedOptions

import textstat
from textblob import TextBlob
from collections import Counter
from profanity_check import predict_prob

import nltk


if "nltk_download" not in st.session_state:
    nltk.download('punkt_tab')
    st.session_state.nltk_download = True

# pd.set_option('display.max_columns', None) # Optional for displaying in Colab


def load_audio(filename):
    """Load audio file and return audio array and sample rate."""
    loader = es.MonoLoader(filename=filename)  # Initialize loader

    audio = loader()  # Load audio
    sample_rate = 44100  # Set sample rate
    return audio, sample_rate


def load_model(audio):
    """Compute embeddings using a pre-trained model from audio."""
    embedding_model = es.TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")

    embeddings = embedding_model(audio)  # Get embeddings
    return embeddings


def compute_duration(audio, sample_rate):
    """Calculate duration of audio in milliseconds."""
    duration = (len(audio) / sample_rate) * 1000  # Compute duration in ms

    return int(duration)


def compute_genre(embeddings):
    """Determine genre based on pre-trained essentia model."""
    model = es.TensorflowPredict2D(graphFilename="msd-msd-musicnn-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
    predictions = model(embeddings)  # Get genre predictions

    # Map essentia's probabilities to the genre name
    mapping = ["rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz",
    "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica", "80s",
    "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard-rock", "ambient", "acoustic", "experimental",
    "female vocalist", "guitar", "hip-hop", "70s", "party", "country", "easy listening", "sexy", "catchy", "funk", "electro",
    "heavy-metal", "Progressive rock", "60s", "rnb", "indie-pop", "sad", "House", "happy"]

    avg_probs = np.mean(predictions, axis=0)  # Average predictions
    genre = mapping[np.argmax(avg_probs)]  # Select genre with highest probability

    # Only use genres that are known to both essentia and spotify
    common_genres = ['pop', 'acoustic', 'jazz', 'dance', 'electronic', 'funk', 'house', 'heavy-metal', 'indie-pop', 'folk', 'hard-rock',
                     'blues', 'hip-hop', 'rock', 'chill', 'metal', 'ambient', 'party', 'soul', 'guitar', 'punk', 'electro', 'sad', 'country']

    if genre not in common_genres:
      genre = "other"  # Set genre to 'other' if not in essentia's and spotify's data

    return genre


def compute_rhythm_features(audio):
    """Extract tempo and beats from audio."""
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    
    tempo, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    return tempo, beats


def compute_key_and_mode(audio):
    """Extract musical key and mode from audio."""
    key_extractor = es.KeyExtractor()
    key, scale, _ = key_extractor(audio)

    mapping = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
        "A#": 10, "Bb": 10, "B": 11}
    key = mapping.get(key, -1)  # Map key to integer

    if scale == 'major':
      mode = 1  # Major mode
    else:
      mode = 0  # Minor mode

    return key, mode


def compute_loudness(audio):
    """Compute loudness from audio."""
    if len(audio.shape) == 1:
        stereo_audio = np.column_stack((audio, audio))  # Convert mono to stereo

    loudness_extractor = es.LoudnessEBUR128()
    loudness = loudness_extractor(stereo_audio)[2]  # Extract loudness
    return loudness


def compute_energy(audio):
    """Compute normalized energy of audio."""
    rms_val = es.RMS()(audio)
    
    return np.clip(rms_val / 0.5, 0, 1)  # Normalize energy


def compute_danceability(audio, sample_rate):
    """Compute danceability score of audio."""
    danceability = es.Danceability(sampleRate=sample_rate)(audio)[0] / 3  # Compute danceability
    
    return np.clip(danceability, 0, 1)  # Clip value between 0 and 1


def compute_acousticness(embeddings):
    """Compute acousticness score from pre-trained essentia model."""
    model = es.TensorflowPredict2D(graphFilename="mood_acoustic-msd-musicnn-1.pb", output="model/Softmax")
    predictions = model(embeddings)
    
    acousticness = np.mean(predictions[:, 0])  # Average prediction for acousticness
    return acousticness


def compute_instrumental_and_speech(embeddings):
    """Compute instrumentalness and speechiness from pre-trained essentia model."""
    model = es.TensorflowPredict2D(graphFilename="voice_instrumental-msd-musicnn-1.pb", output="model/Softmax")
    predictions = model(embeddings)
    
    instrumental = np.mean(predictions[:, 0])  # Average instrumental score
    speech = np.mean(predictions[:, 1])  # Average speechiness score
    return instrumental, speech


def compute_valence(embeddings):
    """Compute valence score from pre-trained essentia model."""
    model = es.TensorflowPredict2D(graphFilename="deam-msd-musicnn-2.pb", output="model/Identity")
    predictions = model(embeddings)
    
    valence = (np.mean(predictions[:, 0])-1) / 8  # Normalize valence
    return valence


def compute_time_signature(audio, beats, sample_rate):
    """Compute time signature using beats and loudness features."""
    beats_loudness = es.BeatsLoudness(beats=beats, sampleRate=sample_rate)
    loudness, loudness_band_ratio = beats_loudness(audio)
    beatogram_algo = es.Beatogram()
    beatogram = beatogram_algo(loudness, loudness_band_ratio)
    meter = es.Meter()
    time_signature = meter(beatogram)
    
    if time_signature % 2 == 0:
        time_signature = 4.0  # Default to common time
    return time_signature


def transcribe_lyrics(filename):
    """Transcribe lyrics from audio using the Deepgram API with OpenAI Whisper."""
    deepgram = DeepgramClient(st.secrets["DG_TOKEN"])

    with open(filename, 'rb') as buffer_data:  # API request according to Deepgram's docs
        payload = { 'buffer': buffer_data }

        options = PrerecordedOptions(
            smart_format=True, model="whisper"
        )

        response = deepgram.listen.rest.v('1').transcribe_file(payload, options, timeout=600)

    transcription = response.to_json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcription


def ngram_repetition(text):
    """Compute n-gram repetition score (higher means more repetitive)."""
    blob = TextBlob(text)
    ngrams = blob.ngrams(3)  # Generate trigrams
    
    if not ngrams:
        return 0.0
    
    ngram_counts = Counter(map(tuple, ngrams))  # Count trigrams
    most_common_count = max(ngram_counts.values())
    total_ngrams = sum(ngram_counts.values())
    return most_common_count / total_ngrams  # Compute repetition ratio


def readability_check(text):
    """Compute readability score using Dale-Chall method."""
    readability = textstat.dale_chall_readability_score(text)
    
    if readability > 30:
      readability = 30.0  # Cap maximum readability score
    return readability


def analyze_sentiment(text):
    """Analyze sentiment of text."""
    blob = TextBlob(text)
    
    return blob.sentiment.polarity, blob.sentiment.subjectivity  # Return polarity and subjectivity


def explicitness_check(text):
    """Check explicitness probability of text."""
    explicitness = predict_prob([text])
    
    return explicitness[0]


@st.cache_data(show_spinner=False)
def extract_features(filename):
    """Extract audio and lyric features and return as a DataFrame."""
    with st.status("Analyzing Audio...", expanded=True) as status:
        st.write("Loading Audio")
        audio, sample_rate = load_audio(filename)
        st.write("Preparing for Machine Learning")
        embeddings = load_model(audio)
        st.write("Computing Audio Features")
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
        st.write("Transcribing Lyrics")
        lyrics = transcribe_lyrics(filename)
        st.write("Computing Textual Features")
        repetition = ngram_repetition(lyrics)
        readability = readability_check(lyrics)
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(lyrics)
        explicitness = explicitness_check(lyrics)

        st.write("Finalizing")
    
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

        df[f"genre_{genre}"] = True  # Mark detected genre as True

        status.update(
            label="Analysis complete!", state="complete", expanded=False
        )

    return df

@st.cache_data(show_spinner=False)
def predict_popularity(df):
    """Predict popularity using a prepared machine learning model."""
    with open("model.pkl", "rb") as file:
        linreg = pickle.load(file)
    with open("X_scaler.pkl", "rb") as file:
        X_scaler = pickle.load(file)
    with open("y_scaler.pkl", "rb") as file:
        y_scaler = pickle.load(file)

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = X_scaler.transform(df[numeric_cols])  # Scale features

    y_pred = linreg.predict(df)  # Predict popularity
    y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Rescale prediction
    return y_pred_rescaled


### STREAMLIT ###

# Seitenkonfiguration
st.set_page_config(page_title="HitPredict 🎶", layout="wide")

# Navigation
page = st.sidebar.radio("Navigation 🎛️", ["🏠 Landingpage", "🎵 Song bewerten"])

# CSS Styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

h1, h2, h3, h4, h5, body {
font-family: 'Montserrat', sans-serif;
}

.stButton button {
background-color: #4CAF50;
color: white;
border-radius: 8px;
font-weight: bold;
padding: 8px 20px;
border: none;
cursor: pointer;
}

.stButton button:hover {
background-color: #45a049;
}

.main {
background: linear-gradient(to right, #11998e, #38ef7d);
padding: 20px;
border-radius: 15px;
color: white;
}

.brand-name {
font-size: 28px;
font-weight: 700;
position: absolute;
top: 10px;
left: 20px;
color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand-name'>HitPredict 🎶</div>", unsafe_allow_html=True)

# Landingpage
if page == "🏠 Landingpage":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🌟 Entdecke dein musikalisches Potenzial!")
    st.subheader("KI-gesteuerte Analysen für deinen nächsten Hit")

    st.image("https://media.giphy.com/media/l0MYGb1LuZ3n7dRnO/giphy.gif")

    st.markdown("""
    ### Warum HitPredict?
    - 🎯 **Innovative KI-Analysen**
    - 📈 **Ausdrucksstarke Visualisierung**
    - 🌟 **Inspiration & Erfolg**
    
    ---
    
    ### Erfahrungsberichte
    > "Diese Plattform hat mein Songwriting auf ein neues Level gehoben!" – *Anna M.* 
    > "Absolut empfehlenswert für jeden Musiker!" – *Tom L.*
    
    ---
    
    ### Häufig gestellte Fragen
    **Wie genau ist die Bewertung?** 
    Modernste KI-Technologie analysiert umfassend dein Musikstück.
    
    **Ist HitPredict kostenlos nutzbar?** 
    Ja, grundlegende Analysen sind kostenfrei nutzbar.
    
    **Was passiert mit meinen Daten?** 
    Deine Privatsphäre ist sicher und alle Daten bleiben vertraulich.
    """)

    if st.button("🎵 Jetzt Song bewerten!"):
        page = "🎵 Song bewerten"

    st.markdown("</div>", unsafe_allow_html=True)


# Song bewerten Seite
elif page == "🎵 Song bewerten":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🎤 Lade deinen Song hoch")

    uploaded_file = st.file_uploader("🎶 Audio hochladen (MP3)", type=["mp3"])

    if uploaded_file:
        st.audio(uploaded_file)

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(uploaded_file.read())
            filename = temp_file.name

        df = extract_features(filename)

        st.header("📊 Metriken der Songanalyse")

        # Separate genre columns and feature columns
        genre_cols = []
        feature_cols = []

        for column in df.columns:
            if column.startswith("genre_"):
                genre_cols.append(column.replace("genre_", ""))
            elif column != "year":
                feature_cols.append(column)

        for feature in feature_cols:
            value = df.loc[0, feature]
            st.metric(label=feature.capitalize(), value=round(float(value), 2))

        current_genre = "other"
        for column in genre_cols:
            if df.at[0, f"genre_{column}"] == True:
                current_genre = column
                break

        st.metric(label="Genre", value=current_genre.capitalize())

        # Dropdown for genre
        selected_genre = st.selectbox("Change genre", genre_cols, index=genre_cols.index(current_genre), disabled=True)

        if selected_genre:
            df[f"genre_{current_genre}"] = False
            df[f"genre_{selected_genre}"] = True


        # Predict popularity
        overall_score = predict_popularity(df.copy())

        st.subheader(f"✨ Gesamtbewertung des Songs: {overall_score[0,0]:.1f} / 100 ✨")

        st.markdown("</div>", unsafe_allow_html=True)
