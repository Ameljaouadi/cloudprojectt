# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from gradio.components import  Image, Label
from gradio import Interface
import cv2

import numpy as np
from keras.models import load_model
import monconfig

import sys
sys.path.append("C:/Users/ASUS/Desktop/copie0/webapp") 
import monconfig




# load the trained CNN model
cnn_model = load_model(monconfig.MODEL_LOC)





def make_prediction(test_image):
    # Convert the image array to a format compatible with OpenCV
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    
    # Resize the image to the required input size of your model (224x224)
    test_image = cv2.resize(test_image, (224, 224))
    
    # Preprocess the image (if necessary) before feeding it to the model
    test_image = test_image / 255.0  # Normalize pixel values
    
    # Expand the dimensions to match the expected input shape of the model
    test_image = np.expand_dims(test_image, axis=0)
    
    #test_image = test_image.image
    #test_image = image.load_img(test_image, target_size=(224, 224))
    #test_image = image.img_to_array(test_image) / 255.
    #test_image = np.expand_dims(test_image, axis=0)
    result = cnn_model.predict(test_image)
    return {"Normal": str(result[0][0]), "Pneumonia": str(result[0][1])}


image_input = Image()
output_label = Label()



title = "PneumoSCAN"
description = "<div style='text-align: center;'>This application uses a Convolutional Neural Network (CNN) model to predict whether " \
              "a chosen X-ray shows if the person has pneumonia disease or not.</div>"

Interface(fn=make_prediction,
             inputs=image_input,
             outputs=output_label,
             examples=[["image1_normal.jpeg"],
                       ["image2_normal.jpeg"],
                       ["image3_normal.jpeg"],
                       ["image4_normal.jpeg"],
                       ["image1_pneumonia_virus.jpeg"],
                       ["image2_pneumonia_virus.jpeg"],
                       ["image1_pneumonia_bacteria.jpeg"],
                       ["image2_pneumonia_bacteria.jpeg"]],
             title=title,
             description=description) \
    .launch()
