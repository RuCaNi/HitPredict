#
# HitPredict - Ein Projekt für 'Grundlagen und Methoden der Informatik für Wirtschaftswissenschaften' an der Universität St.Gallen (2025)
# Autoren: Ruben Cardell, Adam Bisharat, Helena Häußler, Colin Wirth
# ---
# HINWEIS: Das Herz des Projektes befindet sich in pages/1_🎵_Song_bewerten.py
# ---
# ACHTUNG: Installation
# 1. Es müssen alle Libraries in exakt der richtigen Version aus requirements.txt installiert werden
# 2. Die App benötigt Python 3.11
# 3. Die Essentia Modelldateien müssen im gleichen Verzeichnis wie die Landingpage.py Datei sein
# 4. Es kann Fehler beim Ausführen geben, wenn die Packages mit einer falschen Numpy Version kompiliert werden
# !!! Wir empfehlen, die App auf https://hitpredict.streamlit.app/ anzuschauen !!!
#

# Importe
import streamlit as st
import random


# Streamlit Konfiguration
st.set_page_config(page_title="HitPredict 🎶", layout="wide", page_icon="favicon.png")
st.logo("Logo.png", size="large")


# Farbverlauf als Hintergrund
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to left, #fa9237, #f5cb40);
    }
</style>
""", unsafe_allow_html=True)


# Titel
st.title("HitPredict 🌟 Entdecke dein musikalisches Potenzial!")


# Abschnitt: Business Problem
col1, col2 = st.columns([0.7,0.3])  # Spaltenbreite 70%-30%
with col1:
    st.subheader("Was wäre, wenn du schon vor dem Release wissen könntest, wie dein neuer Song ankommt?")
    st.markdown("""
    Egal ob du Artist, Manager, Label oder Verlag bist – du kennst das Problem:
    - **Welche Single soll zuerst raus? In welchen Song steckst du dein Marketingbudget?** Und wie sicher bist du wirklich, dass dein Track das Zeug zum Hit hat?
    
    #### 💸 Musikbusiness heisst Risiko. Produktion, Mix, Mastering, Promo – alles kostet Geld, Zeit und Energie.
    Und trotzdem bleibt immer dieses kleine Fragezeichen: **Wird der Song durchstarten oder einfach untergehen?**
    
    - Genau hier hilft dir **HitPredict**. Mit einem einzigen Upload bekommst du eine **klare, datenbasierte Einschätzung**, wie viel Potenzial dein Song hat. 
    - Unser **Popularity Score (0–100)** verrät dir, wie hoch die Chancen stehen, dass dein Track durch die Decke geht.
    
    #### *Know your hit before it’s heard.*
    """)

with col2:
    # Zufällige Auswahl aus handverlesenen Musik-GIFs
    st.image(random.choice(["https://media1.tenor.com/m/9SFSfC2n0lkAAAAd/head-phones-music.gif",
     "https://media1.tenor.com/m/8DDZDteRUFgAAAAd/muzeke.gif",
     "https://media1.tenor.com/m/qysazDtt3xAAAAAC/the-simpsons-dancing.gif",
     "https://media1.tenor.com/m/dza0F8SCVvUAAAAC/johncena-john.gif",
     "https://media1.tenor.com/m/jQEfimz3kbsAAAAd/putting-on-repeat-this-is-my-jam.gif",
     "https://media1.tenor.com/m/QkAMUD6roPMAAAAd/spongebob-best-song-ever.gif",
     "https://media1.tenor.com/m/xiSmAeQc50UAAAAd/dancing-john-legend.gif",
     "https://media1.tenor.com/m/u01J4W6IXO8AAAAd/push-the-button-the-voice.gif",
     "https://media1.tenor.com/m/xStyNwaJfZsAAAAd/trump-wins-trump-2024.gif"]), width=500)

st.divider()


# Abschnitt: Wie funktioniert das?
st.subheader("🚀 Wie funktioniert das?")
st.markdown("""
Wenn du deinen Song bei uns hochlädst, passiert richtig viel Magie:
- Wir analysieren dein Stück auf Herz und Nieren – Tempo, Tanzbarkeit, Stimmung, Lyrics, Emotionen – einfach alles, was einen Song einzigartig macht.
- Dann vergleichen wir dein Werk mit einer gigantischen Datenbank von über **560.000 Songs**.
- Dazu nutzen wir **Machine Learning** (klingt kompliziert, aber heisst nur: unser System \"versteht\" automatisch, was einen Hit ausmacht).
""")


# Abschnitt: Und dann?
st.subheader("🎵 Und dann?")
st.markdown("""
Unser intelligenter Algorithmus – powered by einem Profi-Tool namens **XGBoost** – sagt dir, wie hoch dein Erfolgspotenzial ist.

Du bekommst einen **Score zwischen 0 und 100** – je höher, desto näher bist du dem nächsten grossen Hit!
""")


# Abschnitt: Warum ist das cool?
st.subheader("🎯 Warum ist das cool?")
st.markdown("""
- Du bekommst ehrliches, datenbasiertes Feedback.
- Du kannst gezielt an deinem Sound arbeiten.
- Du entscheidest besser, welchen Song du zuerst raushaust.
""")

st.divider()


# Abschnitt: Erfahrungsberichte
st.markdown("### Erfahrungsberichte")
st.markdown('''> "Früher musste ich auf mein Bauchgefühl hören. Jetzt höre ich auf HitPredict. 
Der Score hat mir bei meinem letzten Album geholfen, die perfekten Singles auszuwählen. Und ja – er lag jedes Mal richtig!" – *T. Swift*''')

st.markdown('''> "Normalerweise freestyle ich alles – aber HitPredict hat sogar meinen besten Freestyle korrekt vorhergesagt. 
Score 89! Respekt, Maschine." - *Eminem*''')

st.markdown('''> "Als DJ musst du wissen, welcher Track das Publikum komplett ausrasten lässt. 
HitPredict hat mir geholfen, die absoluten Banger für meine nächste Tour zu identifizieren. Kein Blindflug mehr." – *D. Guetta*''')

st.divider()


# Slogan
st.markdown("#### HitPredict: *Know your hit before it’s heard.*")


# Button zur Song bewerten Seite
if st.button("🎵 Jetzt Song bewerten!", type="primary"):
    st.switch_page("pages/1_🎵_Song_bewerten.py")