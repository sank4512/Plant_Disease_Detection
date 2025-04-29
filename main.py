import streamlit as st
import tensorflow as tf 
import numpy as np

#OUR MODEL
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_cnn_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

#SIDEBAR
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("select page", ["Home", "About", "Diseases Prediction"])

# Dark mode toggle using session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Add a button to toggle the theme
st.sidebar.button("Toggle Dark Mode", on_click=toggle_theme)

# Apply the theme
if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #f7f7f7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

#HOME PAGE
if app_mode == "Home":
    st.header("PLANT DISEASES RECOGNITION SYSTEM")
    st.image("static/plant2.jpg", use_container_width=True)
    st.markdown(""" 
# 🌿 Welcome to Our Plant Diseases Recognition System! 🌿

Plants are the foundation of life on Earth, providing food, oxygen, and countless resources essential for human survival. However, like all living organisms, plants are vulnerable to diseases that can severely impact their health, productivity, and overall ecosystem balance.

At Plant Diseases Recognition System, we aim to provide an intelligent, AI-powered solution to identify plant diseases accurately and efficiently. Our platform is designed to assist farmers, researchers, and gardening enthusiasts in diagnosing plant ailments, ensuring healthier crops, and promoting sustainable agriculture.

# 🌱 Why Choose Our System?
-   ✔ Accurate Diagnosis – Our AI model is trained on a vast dataset of plant diseases to provide precise predictions.
- ✔ Fast and Easy to Use – Upload an image, and within seconds, get a diagnosis for your plant.
- ✔ Wide Range of Plants Supported – Our system recognizes diseases in multiple plant species, including fruits, vegetables, and flowers.
- ✔ User-Friendly Interface – Designed for farmers, botanists, and plant lovers of all backgrounds.

# 🎯 Our Vision            
Our vision is to leverage artificial intelligence and machine learning to revolutionize plant disease detection. 
By providing a quick and reliable diagnosis, we aim to empower farmers and plant enthusiasts with the knowledge they need to take timely action, ultimately reducing crop losses and ensuring food security worldwide.

# 🛠️ Our Mission
- 🌍 Sustainable Farming – We strive to reduce the use of harmful pesticides by enabling early detection and treatment of plant diseases.
- 📈 Increased Crop Yield – By identifying issues before they escalate, we help farmers maximize their harvest.
- 💡 Knowledge Sharing – We are committed to educating users on plant diseases and their effective treatments.

# 🌟 Our Values
- 🔬 Innovation – We continuously enhance our system with the latest advancements in AI and deep learning.
- 👨‍🌾 Community Support – Our goal is to support farmers, gardeners, and researchers in their efforts to maintain healthy crops.
- 🌿 Sustainability – We promote eco-friendly solutions for disease prevention and plant care.

# 📞 Contact Us
#### 💬 Have questions or need assistance? We’re here to help!

- 📞 Phone: 8767375722
- 📧 Email: arpitkadam922@gmail.com

📍 Follow us on social media for the latest updates on plant disease detection and agriculture technology!

# 🌻 Happy Planting! 🌻
We believe that every plant deserves a chance to thrive. Let’s work together to protect them and create a greener, healthier world! 🌎🌱
                
""")
    st.image("static/plant_disease.jpg", use_container_width=True)
    

# ABOUT PAGE
elif app_mode == "About":
    st.header("About Us")
    st.image("static/team.png", use_container_width=True)
    st.markdown("""
    ## Our Mission and Vision

    Our team is committed to assisting individuals and organizations in the detection and management of plant diseases using cutting-edge artificial intelligence and machine learning technologies. We believe in empowering farmers, gardeners, and plant enthusiasts by providing them with a tool that enables early identification of plant ailments, which ultimately leads to healthier crops and more sustainable farming practices. 
    
    Our vision is to make plant disease detection more accessible and efficient, ensuring that crops are protected and agricultural productivity is maximized. Through our system, we aim to reduce the use of harmful pesticides by facilitating early disease diagnosis, promoting eco-friendly farming practices.

    ## Why Plant Disease Recognition is Important?

    Plants are vital to life on Earth, providing food, oxygen, and medicinal benefits. However, diseases can significantly impact plant health and crop yield. Early detection of diseases can help farmers take timely action, preventing the spread of infections and reducing the reliance on chemical treatments. This not only helps in protecting the plants but also ensures food security for a growing global population.

    By leveraging AI and machine learning, we can analyze plant images quickly and accurately, offering reliable disease predictions and potential solutions.

    ## The Dataset

    This plant disease recognition system is built on a robust dataset originally sourced from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). The dataset consists of 87,000 RGB images of plants with corresponding labels. These images represent various stages of plant health, including healthy and diseased plants. The data is categorized into 38 different plant classes, which include common fruits, vegetables, and flowers.

    The dataset features images for diseases such as apple scab, powdery mildew, early blight, bacterial spots, and more. These categories allow the system to perform multi-class classification, enabling accurate identification of a wide range of plant diseases.

    ## Technology Behind the System

    Our system utilizes convolutional neural networks (CNNs), a class of deep learning algorithms that are particularly effective for image recognition tasks. CNNs have been trained on this dataset to learn the intricate patterns and features associated with different plant diseases. 

    Key components of the technology include:
    - **Image Preprocessing**: Images are resized and normalized to make them suitable for input into the model.
    - **Convolutional Neural Networks**: The system uses deep learning to classify images based on their features.
    - **Model Evaluation**: The model has been evaluated on a separate validation dataset to ensure high accuracy and reliability.

    ## How It Works

    The user simply uploads an image of a plant leaf or plant in a particular condition, and our system processes the image to detect whether it is diseased and, if so, identify the disease. The system provides an accurate prediction in seconds, giving the user actionable insights that can be used to take preventive or corrective measures.

    ## Our Team

    This project was developed by a group of passionate individuals who are dedicated to leveraging AI to solve real-world agricultural challenges:

    - **Arpit Kadam**: A Machine Learning enthusiast specializing in computer vision and AI solutions for real-world applications.
    - **Sanket Jadhav**: A data science expert with a focus on predictive modeling and statistical analysis.
    - **Ketan Suryavanshi**: A software engineer with expertise in AI development and building scalable systems for diverse applications.

    ## Acknowledgments

    We would like to extend our gratitude to the contributors of the Kaggle dataset and all those involved in the field of agricultural AI. Their efforts have made it possible for us to build this system and help farmers and plant enthusiasts around the world.

    ## Future Plans

    We plan to continue enhancing the system by incorporating additional plant species, expanding the range of diseases detected, and integrating it with real-time data sources such as agricultural weather conditions. This will make the system more comprehensive and applicable to various types of farming practices.

    ### Get Involved
    We are always open to contributions and suggestions. If you have any feedback or would like to collaborate with us on future developments, feel free to reach out to us via the contact information below.
                
    ## Contact Us: 
    - Email: arpitkadam922@gmail.com
    - Phone: 8767375722
    - Personal Website: [Arpit Kadam](https://arpit-kadam.netlify.app/)
    - LinkedIn: [Linkedin](https://www.linkedin.com/in/arpitkadam/)
    - GitHub: https://github.com/ArpitKadam922
    - Instagram: [Arpit Kadam](https://www.instagram.com/arpit__kadam/)
    - Buy me a coffee: [Buy me a coffee](https://buymeacoffee.com/arpitkadam)
    """)

    st.markdown("This project is done by **Arpit Kadam**, **Sanket Jadhav**, and **Ketan Suryavanshi**")



# DISEASES PREDICTION PAGE
elif app_mode == "Diseases Prediction":
    st.header("🌿 Plant Diseases Prediction 🌿")

    st.markdown("""
    Upload an image of a plant's leaf or the plant itself, and our AI-powered system will predict if it is diseased and, if so, which disease it has. The system leverages deep learning to accurately identify diseases based on the images provided.
    """)

    # File uploader for image input
    test_image = st.file_uploader("📸 Upload Plant Image", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

    # Prediction button
    if st.button("🔍 Predict Disease") and test_image is not None:
        with st.spinner("🤖 Predicting..."):
            result_index = model_prediction(test_image)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
                'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            disease = class_name[result_index]

            st.markdown(f"### 🌱 Predicted Disease: {disease}")
            st.success("✅ Prediction complete!")

    # Display additional information and advice for users
    st.markdown("""
    ## How It Works
    1. **Upload**: Select an image of a plant with a potential disease.
    2. **Prediction**: Click "Predict Disease" to let the AI analyze the image.
    3. **Result**: The system will identify the disease or confirm if the plant is healthy.

    ## 🌿 Tips for Plant Care:
    - Early detection of plant diseases can prevent further damage.
    - Once the disease is identified, you can research the appropriate treatment.
    - Consider using eco-friendly solutions to protect both your plants and the environment.

    For more information or to consult experts, feel free to reach out to us via the contact section on the About page.
    """)

    st.markdown("""
    #### 📞 Need Assistance?
    If you have any questions or would like to know more about the system, don't hesitate to contact us.
    Contact Information provided at About Page
    """)

