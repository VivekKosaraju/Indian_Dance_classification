# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:26:00 2020

@author: kosaraju vivek
"""



import numpy as np
import pandas as pd
import streamlit as st
import io
import os
import tensorflow
from werkzeug.utils import secure_filename
import h5py

# Keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



@st.cache(allow_output_mutation=True)
def models():
    model = load_model('model_VGG.h5')
    model.summary()
    return model

# included to make it visible when model is reloaded
    




def model_predict(img_path, model):
    img = image.load_img(img_path,target_size=(224,224,3), grayscale=False)

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    X = np.array(img)
    dances=['bharatanatyam','kathak','kathakali','kuchipudi','manipuri',
            'mohiniyattam','sattriya']
    preds=model.predict_classes(X)
    preds=dances[preds[0]]
   # preds=preds.tolist()
    return preds


 
def main():
    st.set_option('deprecation.showfileUploaderEncoding',False)
    st.title("Image Classification")
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file:
        st.image(image_file,caption="uploaded image",width=10,use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Predicting......"):
            model=models()
            result=model_predict(image_file,model)
            st.success('The output is {}'.format(result))
    
if __name__=='__main__':
    main()
    





    
