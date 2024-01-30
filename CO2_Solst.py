import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np
from PIL import Image

# Function to download data as csv
def download_link(object_to_download, download_filename):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{download_filename}">Download Results</a>'

# Function to predict solubility
def predict_solubility(data0):
    # ... (rest of your code remains unchanged)

html_temp = """
<div style="background-color:tomato;padding:10px;border-radius:10px;margin-bottom:20px">
<h1 style="color:white;text-align:center;">UH CO2 Solubility in Brine Calculator</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.header("Department of Petroleum Engineering: Interaction of Phase Behavior and Flow in Porous Media ([IPBFPM](https://dindoruk.egr.uh.edu/)) Consortium.")
st.subheader("Product Description - Calculates the CO2 Solubility in Brine for Different 9 Salts types at specific Pressure and Temperature.")
st.subheader("[Download Input Template File.](https://drive.google.com/file/d/1IrmFmwePqceAU4qlLBqsmssMtXlAe1Pm/view?usp=sharing)")

file = st.file_uploader("Upload the CSV file", type=['csv'])

if file is not None:
    # Load the data
    data = pd.read_csv(file)

    # Display the loaded data
    st.subheader('Loaded Data:')
    st.dataframe(data)

    # Call the predict function
    results = predict_solubility(data)

    # Display the result table
    if st.button('Predict'):
        st.subheader('Prediction Results:')
        st.dataframe(results)

    # Download the results as csv
    if st.button('Download Results'):
        csv_data = results.to_csv(index=False)
        tmp_download_link = download_link(csv_data, 'CO2_Solubility_Results.csv')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

st.subheader("[Based on the work in Ref;Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). A novel computational strategy to estimate CO2 solubility in brine solutions for CCUS applications. Applied Energy, 342, 121134.](https://www.sciencedirect.com/science/article/pii/S0306261923004981?casa_token=kPpCANAGDIUAAAAA:IGNAx8egWSeRs54UtPnUG1C9OLRKir1DOGPwYm7O2nfeWCP4wKqsCY46_sJGVrk9-YgDrclfGzB4)")
st.image(Image.open('uhlogo.jpg'), caption='A product of University of Houston', use_column_width=True)
