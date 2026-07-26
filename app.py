import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import gdown

# Page Configuration
st.set_page_config(
  page_title="Plant Disease Detection",
  page_icon="🌿",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Custom CSS

st.markdown("""
<style>

[data-testid="stSidebar"]{
background:#1B4332;
}

[data-testid="stSidebar"] *{
color:white;
}

/* Increase radio button font size */
[data-testid="stSidebar"] div[role="radiogroup"] label p{
    font-size:18px !important;
    font-weight:300;
}


.main{
background:#F4FFF7;
}

.hero{
padding:40px;
background:linear-gradient(135deg,#2D6A4F,#40916C);
border-radius:20px;
color:white;
text-align:center;
margin-bottom:30px;
box-shadow:0px 8px 20px rgba(0,0,0,.15);
}

.hero h1{
font-size:48px;
}

.hero p{
font-size:20px;
}

.feature-card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 6px 20px rgba(0,0,0,.08);
text-align:center;
transition:.3s;
height:180px;
}

.feature-card:hover{
transform:translateY(-8px);
}

.result-card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 6px 20px rgba(0,0,0,.08);
}

.green-btn button{
background:#2D6A4F;
color:white;
border-radius:10px;
height:55px;
width:100%;
font-size:20px;
}

.stButton>button{
background:#2D6A4F;
color:white;
border:none;
border-radius:10px;
padding:12px 30px;
font-size:18px;
font-weight:bold;
}

.stButton>button:hover{
background:#1B4332;
}

</style>
""", unsafe_allow_html=True)

# Load Model

# model = tf.keras.models.load_model('/content/drive/MyDrive/Saved Model/trained_model.keras')
# 1. Download the model from your Google Drive link safely
@st.cache_resource  # This prevents the app from downloading the model every time you click a button
def load_my_model():
    # PASTE YOUR FILE ID FROM STEP 2 BETWEEN THE QUOTES BELOW:
    file_id = "1hfPCd1KlLDUz1pEhZPEl06DVHM47luSI"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "trained_model.keras"
    
    if not os.path.exists(output):
        with st.spinner("Downloading plant disease model from Google Drive... Please wait."):
            gdown.download(url, output, quiet=False)
            
    # Load your model (Change this line if you are using PyTorch instead of Keras/TensorFlow)
    return tf.keras.models.load_model(output)

# 2. Call the function to actually load it
model = load_my_model()



# Prediction Function

def model_prediction(test_image):

  image = tf.keras.preprocessing.image.load_img(
      test_image,
      target_size=(128,128)
  )

  input_arr = tf.keras.preprocessing.image.img_to_array(image)

  input_arr = np.array([input_arr])

  prediction = model.predict(input_arr)

  return prediction

# Labels

class_name = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

# Sidebar

# st.sidebar.title("Dashboard")
st.sidebar.markdown("""
<h1 style="
    font-weight:300;
    font-size:32px;
    color:white;
    margin-bottom:20px;
">
    Dashboard
</h1>
""", unsafe_allow_html=True)


page = st.sidebar.radio(
  "",
  [
      "🏠 Home",
      "🔍 Disease Detection",
      "📖 About"
  ]
)

st.sidebar.markdown("---")

# HOME PAGE

if page=="🏠 Home":

  st.markdown("""
  <div class='hero'>
  <h1>🌿 Plant Disease Detection</h1>
  </div>
  """,unsafe_allow_html=True)

  st.markdown("""
  Welcome to the Plant Disease Recognition System! 🌿🔍

  Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

  ### How It Works
  1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
  2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
  3. **Results:** View the results and recommendations for further action.

  ### Why Choose Us?
  - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
  - **User-Friendly:** Simple and intuitive interface for seamless user experience.
  - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

  ### Get Started
  Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

  ### About Us
  Learn more about the project, our team, and our goals on the **About** page.
  """)

# DISEASE DETECTION PAGE

elif page == "🔍 Disease Detection":

  st.markdown("""
  <div class='hero'>
      <h1>🔍 Disease Detection</h1>
      <p>
      Upload a clear image of any plant leaf.
      </p>
  </div>
  """, unsafe_allow_html=True)

  st.subheader("📤 Upload Leaf Image")
  uploaded_image = st.file_uploader(
          "Choose an Image",
          type=["jpg", "jpeg", "png"]
      )

  if uploaded_image is not None:
      st.success("Image uploaded successfully!")

  st.subheader("🖼 Image Preview")

      # Preview button (disabled until an image is uploaded)
  preview = st.button(
          "Preview Image",
          disabled=(uploaded_image is None),
          key="preview_btn"
      )

  if uploaded_image is None:
          st.warning("Please upload an image first.")

  elif preview:
          image = Image.open(uploaded_image)
          st.image(
              image,width=200
          )

  st.write("")
  st.write("")

  if uploaded_image is not None:

      col1,col2,col3=st.columns([1,2,1])

      with col2:

          predict = st.button(
              "🌿 Predict Disease",
              use_container_width=True
          )

      if predict:

          with st.spinner("Analyzing leaf image..."):

              prediction = model_prediction(uploaded_image)

          confidence = float(np.max(prediction))*100

          result_index = np.argmax(prediction)

          disease = class_name[result_index]

          display_name = disease.replace("___"," : ").replace("_"," ")

          is_healthy = "healthy" in disease.lower()

          st.write("")
          st.markdown("## 🌿 Prediction Result")

          st.metric(
                  "🌱 Plant & Disease",
                  display_name
              )

          if is_healthy:

                  st.success("✅ Healthy Leaf Detected")

          else:

                  st.error("⚠ Disease Detected")

          st.write("")
          st.write("---")

          st.subheader("📋 AI Analysis Summary")

          summary1,summary2,summary3 = st.columns(3)

          with summary1:
              st.info(f"**Prediction**\n\n{display_name}")

          with summary2:
              st.info(f"**Confidence**\n\n{confidence:.2f}%")

          with summary3:
              if is_healthy:
                  st.success("Plant Status\n\nHealthy 🌿")
              else:
                  st.warning("Plant Status\n\nDiseased 🍂")

          # if is_healthy:
          #     st.balloons()
          # else:
          #     st.snow()


# ABOUT PAGE

elif page == "📖 About":

  st.markdown("""
  <div class='hero'>
      <h1>📖 About This Project</h1>
      <p>AI-powered Plant Disease Detection using Deep Learning</p>
  </div>
  """, unsafe_allow_html=True)

  st.subheader("🌿 Project Overview")

  st.markdown("""
              #### About Dataset
              This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
              This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
              A new directory containing 33 test images is created later for prediction purpose.
              #### Content

              """)

  st.write("")

  col1, col2, col3, col4 = st.columns(4)

  with col1:
      st.metric("🌿 Disease Classes", "38")

  with col2:
      st.metric("📸 Training Images", "70,295")

  with col3:
      st.metric("✅ Validation Images", "17,572")

  with col4:
      st.metric("📷 Test Images", "33")

  st.write("")

  st.subheader("🧠 Technologies Used")

  tech1, tech2, tech3 = st.columns(3)

  with tech1:
      st.info("""
### Python

- TensorFlow
- NumPy
- Pillow
""")

  with tech2:
      st.info("""
### Deep Learning

- CNN
- Keras
- Image Classification
""")

  with tech3:
      st.info("""
### Web

- Streamlit
- HTML
- CSS
""")

  st.write("")



  st.write("")

  st.markdown("---")

  st.caption(
      "Developed using TensorFlow, Keras and Streamlit."
  )


st.markdown("""
<style>
.stProgress > div > div > div > div{
  background:#2D6A4F;
}

div[data-testid="metric-container"]{
  background:white;
  border-radius:15px;
  padding:15px;
  box-shadow:0px 4px 15px rgba(0,0,0,.08);
}

.stAlert{
  border-radius:12px;
}

img{
  border-radius:15px;
}
</style>
""", unsafe_allow_html=True)



