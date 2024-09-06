import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import pipeline
import string

# Load the trained Transformer model
@st.cache_resource
def load_transformer_model():
    return tf.keras.models.load_model('transformer_model_improved (6).h5')

# Load a pre-trained sentiment analysis model from Hugging Face
@st.cache_resource
def load_nlp_model():
    return pipeline("sentiment-analysis")

# Function to convert model output to English sentence
def convert_to_sentence(decoded_array):
    # Flatten the array if it's not already 1D
    decoded_array = decoded_array.flatten()

    # Log the raw output for inspection
    st.write(f"Raw decoded output: {decoded_array}")

    # Normalize output values to a known range (e.g., 32-126 for ASCII)
    normalized_output = np.interp(decoded_array, (decoded_array.min(), decoded_array.max()), (32, 126))

    # Convert to integers
    processed_output = [int(round(x)) for x in normalized_output]

    # Convert to characters
    decoded_sentence = ''.join([chr(x) for x in processed_output])

    # Log the final decoded sentence
    st.write(f"Final decoded sentence: '{decoded_sentence}'")

    return decoded_sentence

# Optionally, define a character set to filter through
def filter_to_valid_characters(decoded_sentence):
    valid_characters = string.ascii_letters + string.digits + string.punctuation + ' '
    filtered_sentence = ''.join([char if char in valid_characters else ' ' for char in decoded_sentence])
    return filtered_sentence

# Function to preprocess input by padding and reshaping
def preprocess_input(input_list, seq_length=100, feature_size=64):
    # Convert the input list to a numpy array
    encrypted_sentence = np.array([float(x.strip()) for x in input_list])

    # Check if the sequence length is less than expected
    if len(encrypted_sentence) < seq_length * feature_size:
        # Pad the array with zeros to match the required input shape
        padded_sentence = np.pad(encrypted_sentence, (0, seq_length * feature_size - len(encrypted_sentence)),
                                 'constant')
    else:
        # Truncate if it's longer than required
        padded_sentence = encrypted_sentence[:seq_length * feature_size]

    # Reshape the padded array to the required shape
    padded_sentence = padded_sentence.reshape(1, seq_length, feature_size)

    return padded_sentence

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
This app uses a Transformer model to decrypt encrypted sentences and decode the intention behind the sentences using NLP.
""")

# Input: Encrypted sentence
user_input = st.text_area("Enter the encrypted sentence (as a comma-separated list of numbers):", "")

if st.button("Decrypt and Decode"):
    if user_input:
        try:
            # Clean and convert user input to a list of strings
            input_list = user_input.split(',')
            st.write(f"Raw input list: {input_list}")

            # Preprocess the input to match the model's expected shape
            encrypted_sentence = preprocess_input(input_list)
            st.write(f"Encrypted sentence as numpy array: {encrypted_sentence}")

            # Load the model
            model = load_transformer_model()

            # Predict the decrypted sentence
            decoded_sentence = model.predict(encrypted_sentence)

            st.write(f"Decoded sentence raw output: {decoded_sentence}")

            # Convert the decoded sentence to an English sentence
            decoded_sentence_str = convert_to_sentence(decoded_sentence[0])

            # Filter the decoded sentence to valid characters
            filtered_sentence_str = filter_to_valid_characters(decoded_sentence_str)

            # Decode the intention using NLP
            intention = decode_intention_with_nlp(filtered_sentence_str)

            # Display the results
            st.subheader("Decrypted Sentence:")
            st.write(filtered_sentence_str)
            st.subheader("Intention:")
            st.write(intention)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter an encrypted sentence.")
