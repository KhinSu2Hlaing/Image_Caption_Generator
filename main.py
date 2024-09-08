import streamlit as st
import numpy as np
import tensorflow
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for the generation process
    in_text = 'startseq'
    # iterate over the max length of the sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get the index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating the next word
        in_text += " " + word
        # stop if we reach the end tag
        if word == 'endseq':
            break

    return in_text


def main():
    st.title("Image Caption Generator")
    st.markdown("---")

    # Display file uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Read the image file and convert it to array
        image = load_img(uploaded_file, target_size=(224, 224))
        # convert image pixels to numpy array
        image_array = img_to_array(image)
        # Reshape the image data for the model
        image = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        # preprocess image for VGG
        image = preprocess_input(image)
        # extract features
        vgg_model = VGG16()
        # restructure the model to get the second-to-last layer's output (4096,)
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        feature = vgg_model.predict(image, verbose=0)
        # Remove the batch dimension, reshape feature to (4096,)
        feature = feature.reshape((4096,))
        

        # Load the captioning model
        #model = load_model("best_model50.h5")


        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('best_model50.h5')
        

        # Load the tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Set maximum caption length
        max_length = 35

        # Generate caption from the trained model
        caption = predict_caption(model, feature.reshape((4096,)), tokenizer, max_length)
       
        caption = caption.split(' ', 1)[1]
        caption = caption.rsplit(' ', 1)[0]

        # Display the generated caption
        st.subheader("Generated Caption")
        st.write(caption)


if __name__ == "__main__":
    main()
