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
import time


st.markdown("<h2 style='text-align:center;'><strong>News Detector</strong></h2>", unsafe_allow_html=True)

# st.markdown("<h5 style='text-align:center;'>Enter Text</h5>", unsafe_allow_html=True)



# text = st.text_area(" ",height=200)






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



messages = st.container(height=200)
if prompt := st.chat_input("Write Article",max_chars=5000):
    # ________________________________________________________________________________________________

    x = refiner(prompt)

    with open('countVector.pkl', 'rb') as h:
        modal = pickle.load(h)
        # st.write(x)
        arr = modal.transform(x).toarray()

    with open('News Detector.pkl', 'rb') as f:
        model = pickle.load(f)
        prediction = model.predict(arr)

# _________________________________________________________________________________________________

    
    messages.chat_message("user").write(f"User: {prompt}")

    with st.spinner('Wait for it...'):
        time.sleep(3)
        st.success('Done!')

    if prediction == 3:
        # st.write("👻 I guess this is a Business News")
        messages.chat_message("assistant").write("Bot: I guess this is a Business News")
    elif prediction == 1:
        # st.write("👻 I guess this is a World News")
        messages.chat_message("assistant").write("Bot: I guess this is a World News")

    elif prediction == 2:
        # st.write("👻 I guess this is a Sports News")
        messages.chat_message("assistant").write("Bot: I guess this is a Sports News")

    elif prediction == 4:
        # st.write("👻 I guess this is a Science-Technology News")
        messages.chat_message("assistant").write("Bot: I guess this is a Science-Technology News")

    else:
        # st.write("👻 News")
        messages.chat_message("assistant").write("Bot: News")

    































# if st.button("Submit",type="primary"):

#     # st.write("You entered:", text)
#     x = refiner(text)
#     # st.write(x)
#     # print(x)
#     with open('countVector.pkl', 'rb') as h:
#         modal = pickle.load(h)
#         # st.write(x)
#         arr = modal.transform(x).toarray()
        



#     with open('News Detector.pkl', 'rb') as f:
#         model = pickle.load(f)

#         prediction = model.predict(arr)
# #Loader
#         with st.spinner('Wait for it...'):
#             time.sleep(5)
#             st.success('Done!')


#         if prediction == 3:
#             st.write("👻 I guess this is a Business News")
#         elif prediction == 1:
#             st.write("👻 I guess this is a World News")
#         elif prediction == 2:
#             st.write("👻 I guess this is a Sports News")
#         elif prediction == 4:
#             st.write("👻 I guess this is a Science-Technology News")
#         else:
#             st.write("👻 News")

