### IMPORTANT: INSTALL NECESSARY LIBRARIES from requirements.txt ###
### MAKE SURE THAT MODEL AND AUDIO FILES ARE PLACED IN THE SAME FOLDER AS THE .PY ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import essentia.standard as es
from sklearn.metrics.pairwise import euclidean_distances

import streamlit as st
import streamlit.components.v1 as components
import tempfile

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepgram import DeepgramClient, PrerecordedOptions

import textstat
from textblob import TextBlob
from collections import Counter
from profanity_check import predict_prob

from nltk.data import find
from nltk import download


def ensure_punkt():
    try:
        find('tokenizers/punkt_tab')
    except LookupError:
        download('punkt_tab')

ensure_punkt()


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
    "heavy-metal", "Progressive rock", "60s", "rnb", "indie-pop", "sad", "house", "happy"]

    # Only use genres that are known to both essentia and spotify
    common_genres = ['pop', 'acoustic', 'jazz', 'dance', 'electronic', 'funk', 'house', 'heavy-metal', 'indie-pop', 'folk', 'hard-rock',
                     'blues', 'hip-hop', 'rock', 'chill', 'metal', 'ambient', 'party', 'soul', 'guitar', 'punk', 'electro', 'sad', 'country']

    avg_probs = np.mean(predictions, axis=0)  # Average predictions

    for i in range(len(mapping)):
        if mapping[i] not in common_genres:
            avg_probs[i] = 0.0  # Set the probabilities for all genre's neither in spotify's nor essentia's data to zero

    avg_probs[0] *= 0.1  # Adjustment for overweight on 'rock'
    avg_probs[4] *= 0.1  # Adjustment for overweight on 'electronic'
    avg_probs[11] *= 0.1  # Adjustment for overweight on 'metal'
    avg_probs[28] *= 0.1  # Adjustment for overweight on 'ambient'
    avg_probs[1] *= 3  # Adjustment for underweight on 'pop'

    genre = mapping[np.argmax(avg_probs)]  # Select genre with the highest probability

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

    transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
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
        st.write("Transcribing Lyrics (∼1 minute)")
        lyrics = transcribe_lyrics(filename)
        st.write("Computing Textual Features")
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

        df[f"genre_{genre}"] = True  # Mark detected genre as True

        status.update(
            label="Analysis complete!", state="complete", expanded=False
        )

    return df


@st.cache_data(show_spinner=False)
def predict_popularity(df):
    """Predict popularity using a prepared machine learning model."""
    with open("xgboost_v4.pkl", "rb") as file:
        model = pickle.load(file)
    with open("X_scaler.pkl", "rb") as file:
        X_scaler = pickle.load(file)
    with open("y_scaler.pkl", "rb") as file:
        y_scaler = pickle.load(file)

    df = X_scaler.transform(df)  # Scale features

    y_pred = model.predict(df)  # Predict popularity
    y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Rescale prediction
    return y_pred_rescaled


@st.cache_data(show_spinner=False)
def get_track_info(track_id): ###
    """Initialize Spotify client, search for a track, and return its info."""
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=st.secrets["SP_CLIENT_ID"],
        client_secret=st.secrets["SP_CLIENT_SECRET"]
    ))

    track = sp.track(track_id)

    return {
        'track_name': track['name'],
        'artist_name': track['artists'][0]['name'],
        'album_release': track['album']['release_date'],
        'album_cover_url': track['album']['images'][0]['url'],
        'popularity': track['popularity']
    }


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv("spotify_data_similarity.csv", index_col=0)
    df = pd.get_dummies(df, columns=['genre'])
    return df


@st.cache_data(show_spinner=False)
def get_soulmate(X_pred, y_pred):
    data = load_dataset()

    # Load scalers
    with open("X_scaler.pkl", "rb") as f:
        X_scaler = pickle.load(f)
    with open("y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # Define columns
    X_cols = [col for col in data.columns if col not in ['track_id', 'popularity']]
    y_cols = ['popularity']

    # Scale the dataset
    X_scaled = X_scaler.transform(data[X_cols])
    y_scaled = y_scaler.transform(data[y_cols])

    data_scaled = np.hstack([y_scaled, X_scaled])

    # Scale the prediction
    X_pred_scaled = X_scaler.transform(pd.DataFrame(X_pred, columns=X_cols))
    y_pred_scaled = y_scaler.transform(pd.DataFrame(y_pred, columns=y_cols))

    pred_scaled = np.hstack([y_pred_scaled, X_pred_scaled])

    # Find the closest match
    distances = euclidean_distances(pred_scaled, data_scaled)
    closest_idx = distances.argmin()

    soulmate_id = data.iloc[closest_idx]['track_id']
    return soulmate_id



### INTERFACE ###
st.set_page_config(page_title="HitPredict 🎶", layout="wide")
st.logo("Logo.png", size="large")

st.title("🎤 Lade deinen Song hoch")

uploaded_file = st.file_uploader("🎶 Audio hochladen (MP3)", type=["mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    if "df" not in st.session_state or "last_uploaded" not in st.session_state or uploaded_file != st.session_state.last_uploaded:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(uploaded_file.read())
            filename = temp_file.name

        st.session_state.df = extract_features(filename)
        st.session_state.last_uploaded = uploaded_file

    st.divider()

    st.header("📊 Metriken der Songanalyse")

    col1, col2 = st.columns(2)
    with col2:
        # Get selected genre
        current_genre = "other"
        for column in st.session_state.df.columns:
            if column.startswith("genre_"):
                if st.session_state.df.at[0, column] == True:
                    current_genre = column.replace("genre_", "")
                    break

        st.metric(label="Genre", value=current_genre.capitalize())

        mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                   6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
        key = mapping.get(float(st.session_state.df["key"]), "C")

        st.metric(label="Key", value=key)

        if int(st.session_state.df["mode"]) == 1:
            mode = "Major"
        else:
            mode = "Minor"

        st.metric(label="Mode", value=mode)

        st.metric(label="Time signature", value=f'{int(st.session_state.df["time_signature"])}/4')

        st.write("")

        # Predict popularity
        popularity = predict_popularity(st.session_state.df)

        st.header(f"✨ Popularity Score: {popularity[0, 0]:.1f} / 100 ✨")

        if popularity <= 30:
            st.subheader("0–30 Punkte (Kein Hit)")
            st.markdown("**Dein Song hat noch nicht das Zeug zum Hit – aber jeder Star hat mal klein angefangen! 🌱**")
            st.markdown("➔ Bleib dran und nutze das Feedback, um deinen Sound auf ein neues Level zu bringen. 🎛️")
        elif popularity <= 50:
            st.subheader("🛠️ 30–50 Punkte (Hit-Potenzial)")
            st.markdown(
                "**Dein Song hat starke Ansätze – Feintuning an Tanzbarkeit oder Lyrics und du bist auf Kurs! 🚀**")
            st.markdown("➔ Manchmal reicht ein cleverer Refrain oder ein knackiger Beat, um Herzen zu erobern! 💓")
        elif popularity <= 60:
            st.subheader("🔥 50–60 Punkte (Wahrscheinlicher Hit)")
            st.markdown(
                "**Dein Song tanzt an der Schwelle zum Hit – ein Funken mehr und die Crowd wird explodieren! 🎉**")
            st.markdown("➔ Das Fundament ist stark, jetzt brauchst du nur noch den perfekten Feinschliff. 🛠️")
        else:
            st.subheader("🚀 60+ Punkte (Mega-Hit)")
            st.markdown("**Dein Track ist ein Volltreffer! Die Charts warten schon auf dich – let's go! 🔥🏆**")
            st.markdown("➔ Bleib fokussiert, bleib echt – Hits entstehen, wenn Herzblut auf Timing trifft. ❤️⏳")

    with col1:
        df = st.session_state.df

        plot_df = pd.DataFrame([{
            'Danceability': np.clip(df['danceability'] / 0.664, 0, 1),
            'Energy': np.clip(df['energy'] / 0.89, 0, 1),
            'Loudness': np.clip(1 - 0.1 * (df['loudness'] / -4.791), 0, 1),  # Extra
            'Speechiness': np.clip(df['speechiness'] / 0.75, 0, 1),  # Extra
            'Acousticness': np.clip(df['acousticness'] / 0.453, 0, 1),
            'Instrumentalness': np.clip(df['instrumentalness'] / 0.75, 0, 1),  # Extra
            'Valence': np.clip(df['valence'] / 0.656, 0, 1),
            'Tempo': np.clip(df['tempo'] / 141.057, 0, 1),
            'Duration': np.clip(df['duration_ms'] / 269747, 0, 1),
            'Repetition': np.clip(df['repetition'] / 0.043956, 0, 1),
            'Readability': np.clip(df['readability'] / 16.1, 0, 1),
            'Polarity': np.clip(df['sentiment_polarity'] / 0.15066, 0, 1),
            'Subjectivity': np.clip(df['sentiment_subjectivity'] / 0.6, 0, 1),
            'Explicitness': np.clip(df['explicitness'] / 0.288619, 0, 1)
        }])

        # Radar chart setup
        labels = plot_df.columns
        values = plot_df.iloc[0].tolist()
        values += values[:1]  # Close the loop

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='#f8641b', linewidth=2, alpha=0.8)
        ax.fill(angles, values, color='#f8641b', alpha=0.33)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticklabels([])  # Hide radial scale for clarity

        plt.tight_layout()
        st.pyplot(fig=plt, use_container_width=False)


    # Song Soulmate
    spotify_id = get_soulmate(st.session_state.df, popularity)

    similar_song = get_track_info(spotify_id)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("👫 Song Soulmate")

        subcol1, subcol2, subcol3 = st.columns([3, 1, 1])
        subcol1.subheader(f'{similar_song["artist_name"]} - {similar_song["track_name"]}')
        subcol2.metric(label='Popularity', value=similar_song["popularity"])
        subcol3.metric(label='Release', value=similar_song["album_release"][:4])

        components.iframe(f"https://open.spotify.com/embed/track/{spotify_id}", height=80)

    col2.write("")
    col2.write("")
    col2.write("")
    col2.image(similar_song["album_cover_url"], width=200)
