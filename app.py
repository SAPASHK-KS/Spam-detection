import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("SMS Spam Detection")
st.write("Naive Bayes Classifier")

# User input
message = st.text_area("Enter SMS Message")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)

        if prediction[0] == 1:
            st.error("SPAM MESSAGE")
        else:
            st.success("NOT SPAM (HAM)")
