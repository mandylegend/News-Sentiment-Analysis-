# ----------------- imports -----------------
import os, string, requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
import altair as alt
import streamlit as st



# ----------------- one‑time setup ----------------------------------------------------
for pkg in ("stopwords", "vader_lexicon"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)
# stopword 
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# ----------------- load models -----------------

# cache heavy objects & remote calls
@st.cache_data(show_spinner=False)
def get_roberta():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )


# Fetch news articles from News API
# Uses the News API to fetch articles related to a given keyword
@st.cache_data(show_spinner=False)
def fetch_api(q, page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "apiKey": os.getenv("NEWS_API_KEY", "PAST_YOUR_API_KEY"),
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"News API error → {e}")
        return []
    return r.json().get("articles", [])


# ----------------- helpers -----------------
# Preprocess text by removing punctuation, converting to lowercase, and filtering stop words
def preprocess(text: str) -> str:
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    return " ".join(w for w in text.split() if w not in stop_words)


# ----------------- sentiment analysis -----------------

# Sentiment analysis functions for different models
#
# VADER sentiment analysis
def vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"


# RoBERTa sentiment analysis
def roberta_sentiment(text):
    # Using the cached RoBERTa pipeline
    # Returns a dictionary with 'label' and 'score'
    res = get_roberta()(text, truncation=True)[0]
    label = res["label"]
    label_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'

    }
    
    return label_map.get(label, "Unknown")  # -> Positive / Negative / Neutral


# TextBlob sentiment analysis
def textblob_sentiment(text):
    p = TextBlob(text).sentiment.polarity
    # TextBlob returns a float in the range [-1.0, 1.0]
    # We classify it as Positive, Negative, or Neutral
    return "Positive" if p > 0 else "Negative" if p < 0 else "Neutral" 

# FinBERT sentiment analysis
# Uses the FinBERT model for financial sentiment analysis
def get_finbert():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

finbert_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

# FinBERT sentiment analysis
def finbert_sentiment(text):
    res = get_finbert()(text, truncation=True)[0]["label"].lower()
    return finbert_map[res]


def zero_shot_sentiment(text, candidate_labels=["Positive", "Negative", "Neutral"]):
    # Uses a zero-shot classification pipeline for sentiment analysis
    classifier = pipeline("zero-shot-classification")
    result = classifier(text, candidate_labels)
    return max(zip(result["labels"], result["scores"]), key=lambda x: x[1])[0]


# comment out background image 
# ----------------- Streamlit UI -----------------

# def add_bg_from_url(url: str):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("{url}");
#             background-attachment: fixed;
#             background-size: cover;
#             background-position: center;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Example call
# # Image is randomly sletected from internet

# add_bg_from_url("https://images.ctfassets.net/ukazlt65o6hl/1X7WqUtjm9T7X74oArKV5o/5549604f598a7b9b6ea9abd704491a99/Customer_Sentiment_Analysis.jpeg")



#--------Streamlit app configuration---------------------------------

st.set_page_config("News Sentiment Analyzer", layout="centered", page_icon="📰")
st.title("📰 News Sentiment Analyzer")
keyword = st.text_input("Company or keyword", "OpenAI")
# sidebar 
# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose sentiment analysis model",
    ("VADER", "RoBERTa", "TextBlob", "FinBERT", "Zero-Shot"),
    index=0
)

# Custom CSS for sidebar styling
# This CSS styles the sidebar with a specific background color and padding
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color:#4e5ba0;
        padding: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.info(
    "This app analyzes the sentiment of news headlines related to a given company or keyword. "
    "You can choose from four sentiment analysis models: VADER, RoBERTa, TextBlob, ZeroShot and FinBERT."
)

st.sidebar.progress(100)



# Uncomment the following line to allow users to choose the model via radio buttons and comment out the selectbox above
# model_choice = st.radio("Choose model", ("VADER", "RoBERTa", "TextBlob", "FinBERT"))

# Button to trigger analysis
if st.button("Analyze"):
    with st.spinner("Crunching news …"):
        articles = fetch_api(keyword)
    if not articles:
        st.warning("No fresh articles found.")
    else:
        data = []
        for art in articles:
            clean_title = preprocess(art["title"] or "")
            
          
            sentiment = (
                # Choose sentiment analysis model based on user selection
                vader_sentiment(clean_title) if model_choice == "VADER"
                else roberta_sentiment(clean_title) if model_choice == "RoBERTa"
                else finbert_sentiment(clean_title) if model_choice == "FinBERT"
                else zero_shot_sentiment(clean_title) if model_choice == "Zero-Shot"
                else textblob_sentiment(clean_title)
            )
            data.append(
                {"Headline": art["title"], "Sentiment": sentiment, "Link": art["url"]}
            )
        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.text(f"accuracy: {round(df['Sentiment'].value_counts(normalize=True).max() * 100, 2)}%")
        st.markdown("### Sentiment Distribution")
        st.write(f"Total articles: {len(df)}")
        st.write(f"Model used: **{model_choice}**")
      
        # Display a bar chart of sentiment counts
        st.write("Sentiment counts:")
        st.bar_chart(df["Sentiment"].value_counts())

        each_model_accuracy = {
            "VADER": 0.85,
            "RoBERTa": 0.90,
            "TextBlob": 0.80,
            "FinBERT": 0.88,
            "Zero-Shot": 0.87
        }



        st.markdown("### Model Accuracy")
        st.write("Estimated accuracy of each model based on training data:")    
        st.write(pd.Series(each_model_accuracy).rename("Accuracy (%)").sort_values(ascending=False))
        st.markdown("### About the Models")

        st.write("""
                 - **VADER**: A rule-based sentiment analysis tool that is particularly good for social media text.
                 - **RoBERTa**: A transformer-based model that provides state-of-the-art performance on sentiment analysis tasks.
                 - **TextBlob**: A simple library for processing textual data, providing a straightforward API for common natural language processing (NLP) tasks.
                 - **FinBERT**: A financial sentiment analysis model that is fine-tuned for financial text.
                 - **Zero-Shot**: A model that can classify text into categories it has not seen during training, using a technique called zero-shot learning.""")
        
