import streamlit as st
import requests
import os

# FastAPI endpoint (change when deployed remotely)
API_URL = os.getenv(
    "API_URL",
    "http://api:8000/predict"
    if os.getenv("DOCKER_ENV")
    else "http://localhost:8000/predict",
)

st.set_page_config(page_title="Comment Moderation", page_icon="🛡️", layout="centered")

st.title("🛡️ Real-Time Comment Moderation")
st.write("Enter a comment to check for toxicity using **unitary/toxic-bert**")

# Input box
comment = st.text_area("💬 Enter your comment:", height=120)

if st.button("Moderate"):
    if comment.strip():
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json={"text": comment})

            if response.status_code == 200:
                data = response.json()
                moderation = (
                    "🚩 **Flagged**"
                    if data["moderation_label"] == 1
                    else "✅ **Clean**"
                )

                st.subheader(f"Result: {moderation}")

                st.write("### Toxicity Probabilities:")
                for label, prob in data["toxic_probs"].items():
                    st.progress(prob)
                    st.write(f"**{label}**: {prob:.2%}")
            else:
                st.error(f"API Error: {response.status_code}")
    else:
        st.warning("Please enter a comment first.")
