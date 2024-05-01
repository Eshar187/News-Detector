import streamlit as st
import sklearn
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas
# nltk.download('all')
import re
import nltk
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords


st.markdown("<h1 style='text-align:center;'>News Detector</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center;'>Enter Text</h4>", unsafe_allow_html=True)

text = st.text_area(" ",height=200)




def refiner(txt):
    st = txt
    news = re.sub('[^a-zA-Z]', ' ',st)
    news = news.lower()
    news = news.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    news = [ps.stem(word) for word in news if not word in set(all_stopwords)]
    news = ' '.join(news)
    return [news]




if st.button("Submit",type="primary"):

    # st.write("You entered:", text)
    x = refiner(text)
    # st.write(x)
    print(x)
    with open('countVector.pkl', 'rb') as h:
        modal = pickle.load(h)
        arr = modal.transform(x).toarray()



    with open('News Detector.pkl', 'rb') as f:
        model = pickle.load(f)

        prediction = model.predict(arr)
        if prediction == 3:
            st.write("I guess this is a Business News")
        elif prediction == 1:
            st.write("I guess this is a World News")
        elif prediction == 2:
            st.write("I guess this is a Sports News")
        else:
            st.write("I guess this is a Science-Technology News")