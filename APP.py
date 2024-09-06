import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained Transformer model
@st.cache_resource
def load_transformer_model():
    return tf.keras.models.load_model('transformer_model_improved (6).h5')

# Load a pre-trained sentiment analysis model from Hugging Face
@st.cache_resource
def load_nlp_model():
    return pipeline("sentiment-analysis")

# Initialize the tokenizer (should be the same as the one used during training)
@st.cache_resource
def load_tokenizer():
    tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
    sentences = [
        "I love machine learning and deep learning models.",
        "Transformers have revolutionized natural language processing.",
        "Neural networks are an exciting area of research.",
        "Text processing can benefit from sophisticated algorithms.",
        "Language models are key to understanding context in AI."
    ]
    tokenizer.fit_on_texts(sentences)  # This should match the training tokenizer
    return tokenizer

# Function to preprocess input text for the model
def preprocess_input_text(input_text, tokenizer, max_length=50):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Function to decode tokenized sequences back to sentences
def decode_sequence(tokenized_sequence, tokenizer):
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    decoded_sentence = ' '.join([reverse_word_map.get(token, '?') for token in tokenized_sequence if token > 0])
    return decoded_sentence

# Function to decode the intention using NLP
def decode_intention_with_nlp(decoded_sentence):
    # Load the NLP model
    nlp_model = load_nlp_model()

    # Analyze the sentiment or intention behind the sentence
    result = nlp_model(decoded_sentence)[0]

    # Return the intention based on sentiment analysis
    return f"Intention: {result['label']} with a confidence of {result['score']:.2f}"

# Streamlit app
st.title("Transformer Model for Sentence Decryption and Intention Decoding")

st.write("""
This app uses a Transformer model to decrypt tokenized sentences and decode the intention behind the sentences using NLP.
""")

# Input: Encrypted sentence (input text instead of numbers)
user_input = st.text_area("Enter a sentence to encrypt and decode:", "")

if st.button("Encrypt, Decrypt, and Decode"):
    if user_input:
        try:
            # Load the tokenizer
            tokenizer = load_tokenizer()

            # Preprocess the input text to match the model's expected input format
            encrypted_sentence = preprocess_input_text(user_input, tokenizer)
            st.write(f"Tokenized and padded input: {encrypted_sentence}")

            # Load the Transformer model
            model = load_transformer_model()

            # Predict the decrypted token sequence
            decoded_sequence = model.predict(encrypted_sentence)
            decoded_sequence = np.argmax(decoded_sequence, axis=-1)  # Get token indices with highest probability
            st.write(f"Decoded token sequence: {decoded_sequence}")

            # Convert the decoded sequence back to an English sentence
            decoded_sentence_str = decode_sequence(decoded_sequence[0], tokenizer)
            st.write(f"Decoded sentence: {decoded_sentence_str}")

            # Decode the intention using NLP
            intention = decode_intention_with_nlp(decoded_sentence_str)

            # Display the results
            st.subheader("Decrypted Sentence:")
            st.write(decoded_sentence_str)
            st.subheader("Intention:")
            st.write(intention)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a sentence.")
