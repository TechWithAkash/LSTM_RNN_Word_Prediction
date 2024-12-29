import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model with error handling
try:
    model = load_model('next_word_prediction.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the tokenizer with error handling
try:
    with open("lstm_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Tokenizer file not found! Make sure 'lstm_tokenizer.pickle' is in the correct directory.")
    st.stop()

# Function to predict the next words
def predict_next_words(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure sequence length matches max_sequence_len

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    try:
        predicted = model.predict(token_list, verbose=0)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app UI
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #333333;
        font-size: 2.8rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
        padding: 0.6rem;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 1.2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .output-text {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #007BFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>✨ Next Word Prediction With LSTM RNN ✨</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
        Predict the next word in a sequence using advanced LSTM-based AI technology. 
        Simply type a phrase, and let the AI suggest what comes next!
    </p>
    """,
    unsafe_allow_html=True,
)

# Input box for text
input_text = st.text_input("🔤 Enter a sequence of words:", "To be or not to be")

# Predict button
if st.button("🚀 Predict Next Word"):
    try:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve max sequence length
        next_word = predict_next_words(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.markdown(
                f"<p class='output-text'>The next word is: <span style='color: #28a745;'>{next_word}</span></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<p class='output-text'>Could not predict the next word. Try again!</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.markdown("<p class='output-text'>Waiting for your input... 🤔</p>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 0.9rem; color: #999;">
        Built with ❤️ using LSTM and Streamlit.
    </p>
    """,
    unsafe_allow_html=True,
)
