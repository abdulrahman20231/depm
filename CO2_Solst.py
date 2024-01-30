import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="CO2 Solubility Calculator",
    page_icon="🌊",  # Add your favicon here
)

# Function to download data as csv
def download_link(object_to_download, download_filename):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{download_filename}">Download Results</a>'

# Function to predict solubility
def predict_solubility(data0):
    # ... (unchanged)

# HTML styling for better appearance
html_temp = """
    <div style="background-color: #1f68b4; padding: 1.5px">
        <h1 style="color: white; text-align: center;">UH CO2 Solubility in Brine Calculator</h1>
    </div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Header
st.header("Department of Petroleum Engineering: Interaction of Phase Behavior and Flow in Porous Media (IPBFPM) Consortium.")
st.markdown('<style>h2{color: #1f68b4;}</style>', unsafe_allow_html=True)
st.subheader("Product Description - Calculates the CO2 Solubility in Brine for Different 9 Salt Types at Specific Pressure and Temperature.")
st.markdown('<style>h2{color: #1f68b4;}</style>', unsafe_allow_html=True)
st.subheader("[Download Input Template File.](https://drive.google.com/file/d/1IrmFmwePqceAU4qlLBqsmssMtXlAe1Pm/view?usp=sharing)")

# File uploader
file = st.file_uploader("Upload the CSV file", type=['csv'], help="Upload your data for prediction")

if file is not None:
    # Load the data
    data = pd.read_csv(file)

    # Display the loaded data
    st.subheader('Loaded Data:')
    st.write(data)

    # Call the predict function
    results = predict_solubility(data)

    # Display the result table
    if st.button('Predict'):
        st.write(results)

    # Download the results as csv
    if st.button('Download Results'):
        csv_data = results.to_csv(index=False)
        tmp_download_link = download_link(csv_data, 'CO2_Solubility_Results.csv')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

# Additional content
st.subheader("[Based on the work in Ref;Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). A novel computational strategy to estimate CO2 solubility in brine solutions for CCUS applications. Applied Energy, 342, 121134.](https://www.sciencedirect.com/science/article/pii/S0306261923004981?casa_token=kPpCANAGDIUAAAAA:IGNAx8egWSeRs54UtPnUG1C9OLRKir1DOGPwYm7O2nfeWCP4wKqsCY46_sJGVrk9-YgDrclfGzB4)")

# University of Houston logo
image = Image.open('uhlogo.jpg')
st.image(image, caption='A product of the University of Houston')

# Footer
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2024 University of Houston</p>", unsafe_allow_html=True)
