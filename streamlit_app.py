import streamlit as st
import joblib
import pandas as pd
import re
import string



st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)



@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fake_news_model.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    return model, tfidf

model, tfidf = load_artifacts()


MODEL_ACCURACY = 99.57   

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------- SIDEBAR -------------

with st.sidebar:
    st.title("⚙️ About the model")

    st.write(
        "This app uses a machine learning model trained on a dataset of real and fake "
        "news articles. It looks at the text pattern and predicts whether a new piece "
        "of text is closer to fake or real news in that dataset."
    )

    st.metric("Validation accuracy", f"{MODEL_ACCURACY:.2f}%")

    st.caption(
        "Note: This is a pattern detector, not a perfect truth checker. "
        "It does not know every real world fact. Use it as a support tool."
    )

    st.markdown("---")
    st.markdown("### 🧪 Try an example")

    examples = {
        "Real news style": (
            "The government announced a new economic stimulus package on Monday, "
            "aimed at supporting small businesses affected by the recent slowdown."
        ),
        "Suspicious / clickbait style": (
            "SHOCKING NEWS! Doctors HATE this new trick that cures every disease "
            "in just 24 hours. You will not believe number 7!"
        ),
        "Very short informal text": (
            "ukraine and russia had war lol"
        ),
    }

    example_choice = st.selectbox(
        "Fill the text box with a sample:",
        ["(none)"] + list(examples.keys())
    )



st.title("📰 Fake News Detection System")
st.write(
    "Paste a news headline or a short news paragraph. "
    "The model will predict whether it looks more like fake or real news "
    "based on patterns it learned from the training dataset."
)

# Text input area
default_text = ""
if example_choice in examples:
    default_text = examples[example_choice]

user_text = st.text_area(
    "News text",
    value=default_text,
    height=200,
    placeholder="Type or paste a news headline or short article here..."
)

analyze_button = st.button("Analyze text")


# ------------- PREDICTION LOGIC -------------

def predict_news(text: str):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    probs = model.predict_proba(vectorized)[0]  # [prob_fake, prob_real]
    pred = model.predict(vectorized)[0]         # 0 or 1

    fake_prob = float(probs[0] * 100)
    real_prob = float(probs[1] * 100)

    if pred == 0:
        label = "FAKE NEWS"
        label_emoji = "🔥"
        main_conf = fake_prob
    else:
        label = "REAL NEWS"
        label_emoji = "✅"
        main_conf = real_prob

    return label, label_emoji, main_conf, fake_prob, real_prob





if analyze_button:
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        label, emoji, main_conf, fake_prob, real_prob = predict_news(user_text)

        col1, col2 = st.columns([2, 1])

        # Left: main result
        with col1:
            if label == "FAKE NEWS":
                st.error(f"{emoji} Prediction: {label}")
            else:
                st.success(f"{emoji} Prediction: {label}")

            st.write(f"Model confidence: {main_conf:.2f}%")

        # Right: probabilities
        with col2:
            st.subheader("Class probabilities")
            st.write(f"Fake news probability: {fake_prob:.2f}%")
            st.write(f"Real news probability: {real_prob:.2f}%")

        st.markdown("---")

        st.caption(
            "This prediction is based only on the text you entered and the patterns "
            "the model learned from its training data. It does not guarantee that "
            "the information is factually correct. Always verify important news from "
            "trusted sources."
        )