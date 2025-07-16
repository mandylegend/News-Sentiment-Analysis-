Explanation :-

## Github link :- https://github.com/mandylegend/News-Sentiment-Analysis-.git 

## Important Liabraies :- 
   import requests
   import pandas as pd
   import nltk
   import string
   from nltk.corpus import stopwords
   from nltk.sentiment.vader import SentimentIntensityAnalyzer
   from transformers import pipeline
   import streamlit as st

## * Explanation of code :-

1. Preprocessing Function(preprocess_text)
  - Punctuation Removal: Removes any punctuations marks.
  - Lowercasing: Converts the text to lowercase for consistency.
  - Stopword Removal: Removes common words using NLTK stopwords  list.

## 2. Sentiment Models
  - VADER: Uses VADER's sentiment analysis directly
  - BERT: Uses Hugging Face's transformers pipeline for sentiment  analysis
  - TextBlob: TextBlob is a Python library for processing textual data
  - Zero-Shot:A machine learning paradigm where a model can classify or recognize new, unseen objects 
  - Finbert : A pre-trained language model specifically designed for analyzing sentiment in financial text 

  What is VADER ??
   - VADER(Valence Aware Dictionary and sEntiment Reasoner) is rule based sentiment analysis tool     that is designed for analyzing social media text , newespaper text 
   
  What is BERT ??
    
    - BERT(Bidirectional Representation for Transformers ) By simultaneously examining both sides of a word's context, BERT can capture a word's whole meaning in its context. It processes input data bidirectionally, capturing both preceding and succeeding context.

  What is textbolb
    
    - TextBlob allows us to determine whether the input textual data is positive, negative, or neutral.

  What is FinBert ??
    
    - Finbert Finetuned on Financial PhraseBank; tuned for market tone (bullish / bearish / neutral)

  What is Zero-shot ??
    
    - Classify into any label set at run‑time (e.g., “bullish / bearish / uncertain”) without extra training

## 3. Streamlit Interface:
  - User can input a company name/keyword and choose between VADER or BERT for sentiment analysis.
  - Displays news headlines, sentiment classifications, and a bar chart of sentiment counts.    


## 4. How to run streamlit in Terminal:
  - In terminal type
      1. (cd development phase) and press enter
      2. Then type Streamlit run app.py

## 5. How to run the app:
  - i have shared screenshot pdf of tutorial in detail in interface_pdf_guide



