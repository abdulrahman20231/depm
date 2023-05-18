import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np

import requests
# df = pd.read_csv("C:/Utilities/MMP Papers/HC inj code/MMP_HC_Data ALLwith SG_Clean.csv")
# read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
 #Function to download data as csv
def download_link(object_to_download, download_filename):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{download_filename}">Download Results</a>'

# Function to predict solubility
def predict_solubility(data0):
    data1 = data0.iloc[:, 0:24]
    P = data0['P,Psia']
    T = data0['T,F']
    pressures_converted = P / 14.504
    temp = (T - 32) * 5/9 + 273.15
    pc = 73.8
    tc = 304.25
    rP = pressures_converted / pc
    rT = temp / tc
    Inputs = pd.DataFrame({'rT': np.full_like(pressures_converted, rT), 'rP': rP})
    file_inputs = 'pure_water_solubility.pkl'
    with open(file_inputs, 'rb') as f_pure:
        model_pure = pickle.load(f_pure)
        sc1 = model_pure['scaler']
        model1 = model_pure['model']
    X_input1 = sc1.transform(Inputs)
    sol = model1.predict(X_input1)
    file_inputs1 = 'CO2_Brine_solubility.pkl'
    with open(file_inputs1, 'rb') as f_brine:
        model_brine = pickle.load(f_brine)
        sc2 = model_brine['scaler']
        model2 = model_brine['model']
    X_inputb = sc2.transform(data1)
    solb = model2.predict(X_inputb)
    results = data0.copy()
    results['Brine to Pure Water solubility Ratio'] = solb
    results['Pure Water Solubility Mole Frac'] = sol
    results['Co2 Solubility in Brine at P&T'] = sol * solb
    return results
html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">UH CO2 Solubility in Brine Calculator  </h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.header(
    "Department of Petroleum Engineering: Interaction of Phase Behavior and Flow in Porous Media ([IPBFPM](https://dindoruk.egr.uh.edu/)) Consortium.")

st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader(
    "Product Description - Calculates the CO2 Solubility in Brine for Different 9 Salts type at specific Pressure and Temperature.")
st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader(
    "[Download Input Template File.](https://drive.google.com/file/d/1IrmFmwePqceAU4qlLBqsmssMtXlAe1Pm/view?usp=sharing)")
# st.markdown("[Input Template File Link](https://drive.google.com/file/d/1HNyZjobmTEBcWfk0C2cmClQfahTONrX1/view?usp=sharing)",unsafe_allow_html=True)


file = st.file_uploader("Upload the CSV file", type=['csv'])

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

st.subheader(
    "[Based on the work in Ref;Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). A novel computational strategy to estimate CO2 solubility in brine solutions for CCUS applications. Applied Energy, 342, 121134.](https://www.sciencedirect.com/science/article/pii/S0306261923004981?casa_token=kPpCANAGDIUAAAAA:IGNAx8egWSeRs54UtPnUG1C9OLRKir1DOGPwYm7O2nfeWCP4wKqsCY46_sJGVrk9-YgDrclfGzB4)")
# st.subheader('[Ref.:Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). A novel computational strategy to estimate CO2 solubility in brine solutions for CCUS applications. Applied Energy, 342, 121134.](https://www.sciencedirect.com/science/article/pii/S0306261923004981?casa_token=kPpCANAGDIUAAAAA:IGNAx8egWSeRs54UtPnUG1C9OLRKir1DOGPwYm7O2nfeWCP4wKqsCY46_sJGVrk9-YgDrclfGzB4)')

from PIL import Image

image = Image.open('uhlogo.jpg')
st.image(image, caption='A product of University of Houston')

# print('result===',result)
