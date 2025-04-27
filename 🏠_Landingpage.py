import streamlit as st


# Seitenkonfiguration
st.set_page_config(page_title="HitPredict 🎶", layout="wide")
st.logo("Logo.png", size="large")


# CSS Styles
st.markdown("""
<style>
/* Override Streamlit's main content background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #6bed99, #4de8c1);
}

/* Buttons */
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

/* Brand name on top */
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


# Landingpage
st.title("🌟 Entdecke dein musikalisches Potenzial!")
st.subheader("KI-gesteuerte Analysen für deinen nächsten Hit")

st.image("https://media.giphy.com/media/l0MYGb1LuZ3n7dRnO/giphy.gif")

st.markdown("### Warum HitPredict?")
st.markdown("- 🎯 **Innovative KI-Analysen**")
st.markdown("- 📈 **Ausdrucksstarke Visualisierung**")
st.markdown("- 🌟 **Inspiration & Erfolg**")

st.markdown("---")

st.markdown("### Erfahrungsberichte")
st.markdown('> "Diese Plattform hat mein Songwriting auf ein neues Level gehoben!" – *T. Swift*')
st.markdown('> "Absolut empfehlenswert für jeden Musiker!" – *B. Mars*')

st.markdown("---")

st.markdown("### Häufig gestellte Fragen")
st.markdown("**Wie genau ist die Bewertung?**  \nModernste KI-Technologie analysiert umfassend dein Musikstück.")
st.markdown("**Ist HitPredict kostenlos nutzbar?**  \nJa, grundlegende Analysen sind kostenfrei nutzbar.")
st.markdown("**Was passiert mit meinen Daten?**  \nDeine Privatsphäre ist sicher und alle Daten bleiben vertraulich.")

if st.button("🎵 Jetzt Song bewerten!"):
    st.switch_page("pages/1_🎵_Song_bewerten.py")