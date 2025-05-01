import streamlit as st


st.set_page_config(page_title="HitPredict 🎶", layout="wide")
st.logo("Logo.png", size="large")

st.title("🚀 Ein Musik-Quiz, um die Wartezeit zu überbrücken")
st.write("Bitte beantworte die folgenden Fragen mit A, B, C oder D und klicke dann auf 'Antworten abschicken'.")

punkte = 0 # Punktestand mit 0 initialisieren

antworten = {} # Leeres Dictionary für die Fragen definieren

antworten['frage 1'] = st.radio( #st.radio zeigt Buttons für die Antwortoptionen an und die gegebene Antwort wird dann im Dictionary unter frage 1 gespeichert
    "1. Welcher Musik-Streamingdienst hatte Anfang 2024 die meisten zahlenden Abonnenten weltweit?",
    ["A) Amazon Music", "B) Spotify", "C) Apple Music", "D) YouTube Music"],
    key="frage 1") #key sorgt dafür, dass jeder Button/Frage eindeutig identifizierbar ist

antworten['frage 2'] = st.radio(
    "2. Wer war laut Spotify der meistgestreamte Künstler weltweit im Jahr 2024?",
    ["A) Taylor Swift", "B) Drake", "C) Bad Bunny", "D) The Weekend"],
    key="frage 2")

antworten['frage 3'] = st.radio(
    "3. Welcher Song war laut Spotify der meistgestreamte Song weltweit im Jahr 2024?",
    ["A) Beautiful Things – Benson Boone", "B) Birds of a Feather – Billie Eilish", "C) Cruel Summer – Taylor Swift", "D) Espresso – Sabrina Carpenter"],
    key="frage 3")

antworten['frage 4'] = st.radio(
    "4. Welches Unternehmen ist 2024 der grösste Inhaber von Musikrechten weltweit?",
    ["A) Amazon Music", "B) Universal Music Group", "C) Warner Music Group", "D) BMG"],
    key="frage 4")

antworten['frage 5'] = st.radio(
    "5. Welcher Musikstil wurde in den 2020er Jahren durch Künstler wie Bad Bunny und Karol G weltweit immer populärer?",
    ["A) Trap", "B) EDM", "C) Reggaeton", "D) Indie Rock"],
    key="frage 5")

if st.button("Antworten abschicken"):# Button zum Abschicken der Antworten
    richtige_antworten = {   # Richtige Antworten wurden in Dictionary definiert
        'frage 1': "B",
        'frage 2': "A",
        'frage 3': "D",
        'frage 4': "B",
        'frage 5': "C"}

    for key, user_antwort in antworten.items(): #itineriert über den Dictionary
        richtige_option = richtige_antworten[key] 
        if user_antwort.startswith(richtige_option):# die gegebenen Antworten werden mit den Lösungen abgeglichen (dem ersten Buchstaben davon)
            st.success(f"{key[-1]}. Richtig!") #falls richtig wird der key um eins verringert, "Richtig" angezeigt und der Punktestand erhöht
            punkte += 1
        else:
            st.error(f"{key[-1]}. Falsch. Die richtige Antwort ist {richtige_option})") #falls falsch wird der key um eins verringert

    st.subheader("Quiz beendet!")
    st.write(f"Du hast **{punkte} von 5 Punkten** erreicht.") #die erreichte Punktzahl von maximal 5 wird angezeigt
    if punkte== 5: #verschiedene Kommentare je nach erreichtem Punktestand
        st.write("Grossartig, du bist ein Musikgenie!")
    elif punkte== 4:
        st.write("Gratulation, du hast fast alles gewusst!")
    elif punkte== 3:
        st.write("Mehr als die Hälfte, well done!")
    elif punkte== 2:
        st.write("So kanns weiter gehen!")
    elif punkte== 1:
        st.write("Eins ist besser als keins!")
    else:
        st.write("Du hast noch Luft nach oben!")



