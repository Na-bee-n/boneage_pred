import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image,ImageOps
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import time 
import mysql.connector

import process_input_pipeline as pp
import img_extractor
from process_input_pipeline import ImageResizer, ContrastEnhancer

# added for inception concat
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import combined_model

cm = combined_model.build_regression_model()
cm.load_weights("https://drive.google.com/file/d/1MTD0i7aOb8s1tJlJMNhuUwW_Y8Q_BQvC/view?usp=sharing")
######################################################
roi=img_extractor.RoiExtractor()

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user = "root",
    password = "",
    database = "FeedbackCollection"
)
try:
    if conn.is_connected:
        #create cursor
        cursor = conn.cursor()

        #Create feedback table if not exits
        cursor.execute('''CREATE TABLE IF NOT EXISTS feedback(
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       name VARCHAR(255),
                       feedback_type VARCHAR(255),
                       feedback_message TEXT
        )''')

        conn.commit()
except mysql.connector.Error as e:
    st.error(f"Error connecting to MySQL: {e}")

    
# from sklearn.metrics import mean_absolute_error

# Hide the Streamlit style, including the "Deploy" widget
# st.set_page_config(hide_streamlit_style=True)

# Sidebar: Navigation
st.sidebar.header('Navigation')

page = st.sidebar.selectbox('Select',
                       ['Learn About Bone Age',
                        'Bone Age Predictor',
                        'Resources', 'Feedback Form'])

# # Education page
if page == 'Learn About Bone Age':
    st.title('About Bone Age')

    expander = st.expander('What is bone age?')
    expander.write("'Bone age is an interpretation of skeletal maturity, typically based on radiographs of the left hand and wrist or knee, that has provided useful information in various clinical settings for over 75 years.' (Creo AL, Schwenk WF. Bone age: a handy tool for pediatric providers. Pediatrics. 2017;140(6).)")
    '\n'
    expander = st.expander('What are current methods of determining bone age?')
    expander.write("""
                   * Tanner-White method: ...
                   * Greulich-Pyle method: ...
                   """)
    '\n'
    expander = st.expander('What are some clinical uses for bone age?')
    expander.write("""
                   * Diagnosing certain growth (endocrinologic) conditions
                   * Determining which patients would benefit from treatment
                   * Monitoring treatment
                   * Predicting adult height
                   """)
    '\n'
    expander = st.expander('What are some non-clinical uses for bone age?')
    expander.write("""
                   * Athletics
                   * Forensics
                   * Legal/policy
                   """)

#Bone age prediction
# @st.cache_data
# def get_model(model_path):
#     try:
#         model=load_model(model_path)
#         return model 
#     except Exception as e:
#         st.error(f"Error loading the model: {e}")
#         return None

#load the model
# model = get_model(r'F:\python\web_streamlit\ALL_IN_ONE\combined_weights_10_epoch_fourth_normalaize_1.h5')

# Function for processing image
def process_image(img, img_size=(224, 224)):
    try:
        # image = ImageOps.fit(img, img_size)
        # image = np.asarray(image)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img_resize = preprocess_input(img)
        # img_data = img_resize[np.newaxis,...]

        # image_pipeline = make_pipeline(
        #         ImageResizer(),
        #         ContrastEnhancer()
        # )

        # img_data = image_pipeline.make_pipeline(img)
        new_size=(299,299)
        resized_img = img.resize(new_size, Image.BICUBIC)
        img_array=np.array(resized_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        cl1 = clahe.apply(img_array)
        cl2=cv2.cvtColor(cl1,cv2.COLOR_GRAY2BGR)
        return cl2
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None 
    

# Function for returning prediction
def predict(img_data_c,img_data_m, gender):
    try:
        if gender == 'Female':
            # gender_input = np.array([0])
            gender_input=0
        elif gender == 'Male':
            gender_input = 1


        pred =cm.predict([np.array([img_data_c]),np.array([img_data_m]), np.array([gender_input])])
        # pred = model.predict(img_data)
        return pred
    except Exception as e:
        st.error(f"Error predicting bone age: {e}")
        return None

if page == 'Bone Age Predictor':
    
    st.title('Bone Age Prediction')



    # Upload image
    uploaded_file = st.file_uploader('Upload an image', type='png')
    if uploaded_file is not None:
        # Adding progress bar
        progress_bar = st.progress(0) 

        with st.spinner("Uploading..."):
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i+1)
                
        st.success("Upload Complete")

        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded image', use_column_width=True)

        '\n'
        gender = st.radio('Sex:', ('Male', 'Female'))
        '\n'
        
        if st.button('Predict bone age'):
            st.write("Predicted Bone age is : ")
            img_data = process_image(img)

            # roi=img_extractor.RoiExtractor()
            roi.process_img(img_data)

            
            boneage = predict(roi.carpal_img,roi.metacarpal_img,gender)[0][0]
            print(boneage)
            # boneage=round(boneage,1)
            st.write(boneage,'months')


if page == 'Resources':
    st.title('Resources')

if page == "Feedback Form":
    st.title("Feedback Form")

    name = st.text_input("Name: ")

    feedback_type = st.selectbox("Feedback Type: ",["General","Bug Report","Features"])

    feedback_message = st.text_area("Feedback Message:","")

    if st.button("Submit Feedback"):
        #Insert feedback into the database
        cursor.execute('''INSERT INTO feedback (name,feedback_type,feedback_message) VALUES (%s,%s,%s)''',(name,feedback_type,feedback_message))

        conn.commit()
        st.success("Feedback Submitted")


cursor.close()
conn.close()

 
