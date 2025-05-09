import streamlit as st


st.set_page_config(page_title="HitPredict 🎶", layout="wide", page_icon="favicon.png")
st.logo("Logo.png", size="large")

st.title("🚀 Ein Musik-Quiz, um die Wartezeit zu überbrücken")
st.write("Beantworte jede Frage mit A, B, C oder D und klicke jeweils auf 'Prüfen', um die Auswertung zu sehen.")

# Richtige Antworten
richtige_antworten = {
    'frage 1': "B) Spotify",
    'frage 2': "A) Taylor Swift",
    'frage 3': "D) Espresso – Sabrina Carpenter",
    'frage 4': "B) Universal Music Group",
    'frage 5': "C) Reggaeton"
}

aktuelle_antworten = {}

# Frage 1
st.markdown("#### 1. Welcher Musik-Streamingdienst hatte Anfang 2025 die meisten zahlenden Abonnenten weltweit?")

aktuelle_antworten["frage 1"] = st.radio(
    "Antwort auswählen:",
    ["A) Amazon Music", "B) Spotify", "C) Apple Music", "D) YouTube Music"],
    index=None, label_visibility="collapsed")

if aktuelle_antworten["frage 1"]:
    if st.button("Prüfen", key="prüfen_1"):
        fakt = "Spotify hatte im ersten Quartal 2025 rund 268 Millionen Premium-Abonnenten."

        if aktuelle_antworten["frage 1"] == richtige_antworten["frage 1"]:
            st.success(f"Richtig! ✅ {fakt}")
        else:
            st.error(f"Falsch! ❌ {fakt}")


# Frage 2
st.markdown("#### 2. Wer war laut Spotify der meistgestreamte Künstler weltweit im Jahr 2024?")

aktuelle_antworten["frage 2"] = st.radio(
    "Antwort auswählen:",
    ["A) Taylor Swift", "B) Drake", "C) Bad Bunny", "D) The Weekend"],
    index=None, label_visibility="collapsed")

if aktuelle_antworten["frage 2"]:
    if st.button("Prüfen", key="prüfen_2"):
        fakt = "Mit mehr als 26.6 Milliarden Streams wurde Taylor Swift 2024 zum zweiten Mal in Folge die meistgestreamte Künstlerin."

        if aktuelle_antworten["frage 2"] == richtige_antworten["frage 2"]:
            st.success(f"Richtig! ✅ {fakt}")
        else:
            st.error(f"Falsch! ❌ {fakt}")


# Frage 3
st.markdown("#### 3. Welcher Song war laut Spotify der meistgestreamte Song weltweit im Jahr 2024?")

aktuelle_antworten["frage 3"] = st.radio(
    "Antwort auswählen:",
    ["A) Beautiful Things – Benson Boone", "B) Birds of a Feather – Billie Eilish", "C) Cruel Summer – Taylor Swift", "D) Espresso – Sabrina Carpenter"],
    index=None, label_visibility="collapsed")

if aktuelle_antworten["frage 3"]:
    if st.button("Prüfen", key="prüfen_3"):
        fakt = "Say you can't sleep, baby, I know, That's that me espresso."

        if aktuelle_antworten["frage 3"] == richtige_antworten["frage 3"]:
            st.success(f"Richtig! ✅ {fakt}")
        else:
            st.error(f"Falsch! ❌ {fakt}")


# Frage 4
st.markdown("#### 4. Welches Unternehmen ist 2024 der grösste Inhaber von Musikrechten weltweit?")

aktuelle_antworten["frage 4"] = st.radio(
    "Antwort auswählen:",
    ["A) Amazon Music", "B) Universal Music Group", "C) Warner Music Group", "D) BMG"],
    index=None, label_visibility="collapsed")

if aktuelle_antworten["frage 4"]:
    if st.button("Prüfen", key="prüfen_4"):
        fakt = "Der Katalog der Universal Music Group umfasst über 3 Millionen Aufnahmen und 4 Millionen Kompositionen."

        if aktuelle_antworten["frage 4"] == richtige_antworten["frage 4"]:
            st.success(f"Richtig! ✅ {fakt}")
        else:
            st.error(f"Falsch! ❌ {fakt}")


# Frage 5
st.markdown("#### 5. Welcher Musikstil wurde in den 2020er Jahren durch Künstler wie Bad Bunny und Karol G weltweit immer populärer?")

aktuelle_antworten["frage 5"] = st.radio(
    "Antwort auswählen:",
    ["A) Trap", "B) EDM", "C) Reggaeton", "D) Indie Rock"],
    index=None, label_visibility="collapsed")

if aktuelle_antworten["frage 5"]:
    if st.button("Prüfen", key="prüfen_5"):
        fakt = "Reggaeton gewann 2023 mit 5 der 10 meistgeschauten Musikern auf YouTube viel Reichweite."

        if aktuelle_antworten["frage 5"] == richtige_antworten["frage 5"]:
            st.success(f"Richtig! ✅ {fakt}")
        else:
            st.error(f"Falsch! ❌ {fakt}")


# Gesamtpunktzahl anzeigen
st.markdown("---")
st.subheader("📊 Punktestand")

i = 0
punkte = 0
for frage,antwort in aktuelle_antworten.items():
    if not aktuelle_antworten[frage] == None:
        i += 1
        if antwort == richtige_antworten[frage]:
            punkte += 1

st.write(f"Du hast aktuell **{punkte} von {i} Punkten** erreicht.")


if punkte == 5:  # Verschiedene Kommentare je nach erreichtem Punktestand
    st.write("Grossartig, du bist ein Musikgenie!")
elif punkte == 4:
    st.write("Gratulation, du hast fast alles gewusst!")
elif punkte == 3:
    st.write("Mehr als die Hälfte, well done!")
elif punkte == 2:
    st.write("So kanns weiter gehen!")
elif punkte == 1:
    st.write("Eins ist besser als keins!")
else:
    st.write("Du hast noch Luft nach oben!")
