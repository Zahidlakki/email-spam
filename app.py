import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load saved model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    cleaned_words = []
    for word in words:
        if word.isalnum() and word not in stopwords.words('english'):
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

# Streamlit UI
st.title("📧 Spam Message Classifier")

input_text = st.text_area("Enter the message text below:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed = transform_text(input_text)
        vector = tfidf.transform([transformed]).toarray()
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam (Ham)")
# streamlit run app.py

