import streamlit as st
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load the pre-trained VGG16 model for feature extraction
model_feature = VGG16()
model_vgg16 = Model(inputs=model_feature.inputs, outputs=model_feature.layers[-2].output)

# Define a function to extract image features
def extract_features(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess the image for the VGG16 model
    image = preprocess_input(image)
    # Extract features
    feature = model_vgg16.predict(image, verbose=0)
    return feature

# Function to map index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption for a real-world image
def generate_caption_for_real_image(image_path, model, tokenizer, max_length):
    # Extract features for the real-world image
    image_features = extract_features(image_path)

    # Generate the caption using the trained model
    in_text = 'startseq'
    for i in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict the next word
        yhat = model.predict([image_features, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append the word as input for generating the next word
        in_text += ' ' + word
        # Stop if we reach the end tag
        if word == 'endseq':
            break

    # Remove the start and end tags from the final caption
    final_caption = in_text.split()[1:-1]
    return ' '.join(final_caption)

# Streamlit application setup
st.title("Image Caption Generator")
st.write("Upload an image and generate a caption for it!")

# Upload image through streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Generating caption...")
    
    # Save the image to a temporary file for processing
    image_path = "temp_image.png"
    image.save(image_path)

    # Load the trained model and tokenizer (provide your saved model and tokenizer)
    max_length = 34  # Specify your max length used during training
    model = tf.keras.models.load_model('my_best_model50.h5')  # Replace with the path to your model
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('tokenizer.pickle').read())  # Replace with your tokenizer
    
    # Generate caption for the uploaded image
    caption = generate_caption_for_real_image(image_path, model, tokenizer, max_length)
    
    # Display the generated caption
    st.write("Generated Caption:")
    st.write(caption)

    # Clean up by removing the temporary file
    os.remove(image_path)
