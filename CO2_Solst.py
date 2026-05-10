import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np
from PIL import Image
import os

# -----------------------------
# Page configuration and style
# -----------------------------
st.set_page_config(page_title=" CO2 Solubility Calculator", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 15pt;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Top logos
# -----------------------------
left_space, logo1, logo2, right_space = st.columns([1, 1, 1, 1])

with logo1:
    if os.path.exists("TAMU.png"):
        st.image("TAMU.png", width=200)

with logo2:
    if os.path.exists("IPBF.png"):
        st.image("IPBF.png", width=200)

# -----------------------------
# Original functions
# -----------------------------

# Function to download data as csv
def download_link(object_to_download, download_filename):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{download_filename}">Download Results</a>'

# Function to predict solubility
def predict_solubility(data0):
    P = data0['P,Psia']
    T = data0['T,F']
    pressures_converted = P / 14.504
    temp = (T - 32) * 5/9 + 273.15
    pc = 73.8
    tc = 304.25
    rP = pressures_converted / pc
    rT = temp / tc
    Inputs = pd.DataFrame({'rT': np.full_like(pressures_converted, rT), 'rP': rP})

    # Read ion fixed properties (charge and energy) from a file or any other source
    ion_properties = {
        'Na': {'charge': 1, 'energy': 365},
        'Cl': {'charge': 1, 'energy': 340},
        'HCO3': {'charge': 1, 'energy': 335},
        'Ca': {'charge': 2, 'energy': 1505},
        'CO3': {'charge': 2, 'energy': 1315},
        'SO4': {'charge': 2, 'energy': 1080},
        'Mg': {'charge': 2, 'energy': 1830},
        'K': {'charge': 1, 'energy': 295},
    }

    # Add charges and energy columns based on concentration values
    for ion, properties in ion_properties.items():
        charge_col = f'{ion}_charge'
        energy_col = f'{ion}_energy'
        concentration_col_wt = f'{ion}_concentration_wt%'
        concentration_col = f'{ion}_concentration'

        data0[charge_col] = np.where(data0[concentration_col_wt] != 0, properties['charge'], 0)
        data0[energy_col] = np.where(data0[concentration_col_wt] != 0, properties['energy'], 0)
        data0[concentration_col] = data0[concentration_col_wt]  # Copy values without '_wt%' suffix

    # Drop columns with '_wt%' suffix
    data0 = data0.drop(columns=[f'{ion}_concentration_wt%' for ion in ion_properties.keys()])

    # Continue with the rest of the code
    file_inputs = 'pure_water_solubility.pkl'
    with open(file_inputs, 'rb') as f_pure:
        model_pure = pickle.load(f_pure)
        sc1 = model_pure['scaler']
        model1 = model_pure['model']
        # Make sure the order of features in sc1 is the same as the order in the input data
        sc1_features = sc1.get_feature_names_out()
        Inputs = Inputs[sc1_features]
    X_input1 = sc1.transform(Inputs)

    file_inputs1 = 'CO2_Brine_solubility.pkl'
    with open(file_inputs1, 'rb') as f_brine:
        model_brine = pickle.load(f_brine)
        sc2 = model_brine['scaler']
        model2 = model_brine['model']
        # Make sure the order of features in sc2 is the same as the order in the input data
        sc2_features = sc2.get_feature_names_out()
        data0 = data0[sc2_features]
    X_inputb = sc2.transform(data0)

    sol = model1.predict(X_input1)
    solb = model2.predict(X_inputb)
    results = data0.copy()
    results['Brine to Pure Water solubility Ratio'] = solb
    results['Pure Water Solubility (Mole Frac)'] = sol
    results['Co2 Solubility in Brine at P&T(Mole Frac)'] = sol * solb
    return results

# -----------------------------
# Title / Header
# -----------------------------
st.title("UH CO₂ Solubility in Brine Calculator")

st.markdown(
    "Product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium"
)

st.markdown(
    "Calculates the CO₂ solubility in brine for different salt types at specific pressure and temperature."
)

st.markdown(
    "1) **Download the example input CSV template**: "
    "[Click here](https://drive.google.com/file/d/1-UV3qfaFh39qHkoAJYZBhtUrJhPZHE54/view?usp=sharing)"
)

st.markdown(
    "2) **Based on the work shown in:** "
    "[Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). "
    "A novel computational strategy to estimate CO₂ solubility in brine solutions for CCUS applications. "
    "Applied Energy, 342, 121134.](https://www.sciencedirect.com/science/article/pii/S0306261923004981?casa_token=kPpCANAGDIUAAAAA:IGNAx8egWSeRs54UtPnUG1C9OLRKir1DOGPwYm7O2nfeWCP4wKqsCY46_sJGVrk9-YgDrclfGzB4)"
)

st.divider()

# -----------------------------
# Original UI logic
# -----------------------------
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

st.divider()

