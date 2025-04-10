# Diabetic Retinopathy Detection 🩺🧠

This repository provides a deep learning-based solution for the early detection of **Diabetic Retinopathy (DR)** using retinal fundus images. 
Diabetic Retinopathy is a complication of diabetes that affects the eyes and can lead to blindness if not detected and treated in time. 
Automated systems for DR detection can support ophthalmologists in identifying the disease at an early stage, improving the chances of timely intervention.


## Dataset: DIARETDB1

The project uses the DIARETDB1 dataset, a publicly available benchmark dataset for evaluating diabetic retinopathy detection methods.
This dataset consists of high-resolution fundus images labeled by experts to indicate the presence and severity of diabetic retinopathy features 
such as microaneurysms, hemorrhages, and exudates.

- Total Images: 130 color fundus photographs
- Image size: 1500 x 1152 pixels
- Labels: Normal and Abnormal cases


## File Overview

1. train-and-save-model.py

This script handles the training, evaluation, and saving of a deep learning model for detecting diabetic retinopathy. Key functionalities include:

- Preprocessing of fundus images (resizing, normalization, augmentation)
- Building a Convolutional Neural Network (CNN) model
- Training the model on the DIARETDB1 dataset
- Saving the trained model along with the PCA and Scaler objects for later use
- Evaluation of model performance with accuracy and loss graphs

2. app.py
   This script contains of the code required to create the front-end using streamlit.
   To run the front end, execute the following command on command prompt: streamlit run app.py


## Contributors: Anushka Bhaik - Developer & Researcher
