import streamlit as st
import pickle
import pandas as pd
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

BASE_DIR = Path(__file__).resolve().parent
ps = PorterStemmer()
tfidf = pickle.load(open(BASE_DIR / 'vectorizer.pkl', 'rb'))
model = pickle.load(open(BASE_DIR / 'model.pkl', 'rb'))

# Download NLTK assets if missing so prediction works on fresh environments.
for resource, path_key in (("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")):
    try:
        nltk.data.find(path_key)
    except LookupError:
        nltk.download(resource, quiet=True)

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)
st.title("SMS Spam Classifier")
input_sms=st.text_input("Enter the message")
if st.button('Predict'):
    transform_msg=transform_text(input_sms)
    transform_sms=tfidf.transform([transform_msg])
    result=model.predict(transform_sms)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
        


