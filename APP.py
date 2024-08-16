import streamlit as st
import tensorflow as tf
import numpy as np
import unicodedata
from transformers import pipeline

SEQ_LENGTH = 50
FEATURE_SIZE = 64


# Load the trained Transformer model
@st.cache_resource
def load_transformer_model():
    return tf.keras.models.load_model('transformer_model.h5')


# Load a pre-trained sentiment analysis model from Hugging Face
@st.cache_resource
def load_nlp_model():
    return pipeline("sentiment-analysis")


# Function to convert model output to English sentence
def convert_to_sentence(decoded_array):
    decoded_array = decoded_array.flatten()

    # Debugging: Print raw decoded array
    st.write(f"Raw decoded array: {decoded_array}")

    # Scale the values to a suitable range (e.g., 0 to 1)
    scaled_output = (decoded_array - np.min(decoded_array)) / (np.max(decoded_array) - np.min(decoded_array))

    # Map scaled values to ASCII range (32 to 126)
    ascii_values = (scaled_output * 94 + 32).astype(int)

    # Convert to characters
    chars = [chr(val) for val in ascii_values]

    # Join characters into a string
    decoded_sentence = ' '.join(chars)

    st.write(f"Final decoded sentence: '{decoded_sentence}'")
    return decoded_sentence

# Streamlit app code continues...

# Function to preprocess input by padding and reshaping
def preprocess_input(input_list, seq_length=SEQ_LENGTH, feature_size=FEATURE_SIZE):
    encrypted_sentence = np.array([float(x.strip()) for x in input_list])
    if len(encrypted_sentence) < seq_length * feature_size:
        padded_sentence = np.pad(encrypted_sentence, (0, seq_length * feature_size - len(encrypted_sentence)),
                                 'constant')
    else:
        padded_sentence = encrypted_sentence[:seq_length * feature_size]
    padded_sentence = padded_sentence.reshape(1, seq_length, feature_size)
    return padded_sentence

def decode_intention_with_nlp(decoded_sentence):
    nlp_model = load_nlp_model()
    result = nlp_model(decoded_sentence)[0]
    return f"Intention: {result['label']} with a confidence of {result['score']:.2f}"

# Streamlit app starts here
st.title("Transformer Model for Sentence Decryption and Intention Decoding")

user_input = st.text_area("Enter the encrypted sentence (as a comma-separated list of numbers):", "")

if st.button("Decrypt and Decode"):
    if user_input:
        try:
            input_list = user_input.split(',')
            st.write(f"Raw input list: {input_list}")
            encrypted_sentence = preprocess_input(input_list)
            st.write(f"Encrypted sentence as numpy array: {encrypted_sentence}")
            model = load_transformer_model()
            decoded_sentence = model.predict(encrypted_sentence)
            st.write(f"Decoded sentence raw output: {decoded_sentence}")

            # Debugging: Check the shape of decoded sentence
            st.write(f"Shape of decoded sentence: {decoded_sentence.shape}")

            decoded_sentence_str = convert_to_sentence(decoded_sentence[0])

            # Check if the decoded sentence string is empty or has content
            st.write(f"Decoded sentence string: '{decoded_sentence_str}'")
            if not decoded_sentence_str:
                st.write("Decoded sentence is empty or contains non-printable characters.")

            intention = decode_intention_with_nlp(decoded_sentence_str)

            st.subheader("Decrypted Sentence:")
            try:
                # Critical code here
                decoded_sentence_str = convert_to_sentence(decoded_sentence[0])
                st.write(f"Decoded sentence string: '{decoded_sentence_str}'")
            except Exception as e:
                st.error(f"An error occurred during decoding: {e}")
            st.subheader("Intention:")
            st.write(intention)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter an encrypted sentence.")
