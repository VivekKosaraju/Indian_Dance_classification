# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:26:00 2020

@author: kosaraju vivek
"""



import numpy as np
import pandas as pd
import streamlit as st

# Keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



abc=pd.read_csv('necess.csv')
b=abc['target']
b=np.array(b)
from sklearn.preprocessing import LabelEncoder
ld=LabelEncoder()
b=ld.fit_transform(b)



# Model saved with Keras model.save()
MODEL_PATH ='model_VGG.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path,target_size=(224,224,3), grayscale=False)

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    X = np.array(img)
    preds=model.predict_classes(X)
    preds=ld.inverse_transform(preds)
    preds=preds.tolist()
    return preds[0]


 
def main():
    st.set_option('deprecation.showfileUploaderEncoding',False)
    st.title("Image Classification")

    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file:
        st.image(image_file,caption="uploaded image",width=10,use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Predicting......"):
            result=model_predict(image_file,model)
            st.success('The output is {}'.format(result))
    
if __name__=='__main__':
    main()
    





    
