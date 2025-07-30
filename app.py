import streamlit as st
import tensorflow as tf #import model
from PIL import Image   #upload image 
#pip install pillow     
#import numpy as np
import numpy as np

# @--annotation (calling predefined one)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")
model=load_model()  #load the model


def preprocess_image(image):
    image=image.resize((150,150))  #model size
    img_array=np.array(image)
    #Handle grayscale(1 channel) or rgb (4 channel)
    if img_array.ndim==2:
        img_array=np.stack((img_array,)*3,axis=-1)
        #it will convert grayscale to rgb
    elif img_array.shape[-1]==4:
        img_array=img_array[:,:,:3]
        #drop unwanted 
    img_array=img_array/255.0 #convert to pixel
    img_array=np.expand_dims(img_array,axis=0)
    return img_array



st.set_page_config(page_title="pneumonia detected",layout="centered")
st.title("pneumonia detection")
st.write("upload a chest x-ray image to detect")
uploaded_file=st.file_uploader("upload a chest x-ray image",type=["jpg","png","jpeg"])
if uploaded_file:
    st.success("File uploaded success")
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded")
    with st.spinner("Analysis X-ray "):
        preprocessed=preprocess_image(image)
        prediction=model.predict(preprocessed)[0][0]
        st.success("prediction completed ")
        if prediction>0.5:
            st.error(f"pneumonia affected ( confidence:{prediction:.2f})")
        else:
            st.success(f"pneumonia normal ( confidence:{prediction:.2f})")


