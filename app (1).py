import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
import numpy as np
import pickle

# Function to extract features using VGG16
def extract_features(image, model):
    image = image.resize((224, 224))  # Resize image to match VGG16 input
    image = np.array(image)  # Convert image to numpy array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = preprocess_input(image)  # Preprocess the image
    features = model.predict(image, verbose=0)  # Get features from VGG16
    return features

# Function to generate captions
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Load the VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = load_model("my_best_model50 (1).h5")

# Load the trained captioning model
model = load_model("my_best_model50 (1).h5")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 34  # Define the maximum length of a caption

# Streamlit UI
st.title('Image Caption Generator')
st.write("Upload an image to generate a caption!")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Generating caption...")
    
    # Extract features and generate caption
    photo_features = extract_features(image, vgg_model)
    caption = generate_caption(model, tokenizer, photo_features, max_length)
    
    st.write("Generated Caption: ", caption)
