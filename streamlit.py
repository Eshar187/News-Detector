import streamlit as st
import sklearn
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# nltk.download('all')
import re
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer

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
    st.write(x)




    with open('News Detector.pkl', 'rb') as f:
        model = pickle.load(f)

        prediction = model.predict(text)
        st.write([prediction])