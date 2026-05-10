import pandas as pd
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# -----------------------------
# Page configuration and style
# -----------------------------
st.set_page_config(page_title="CO2 Solubility Calculator", layout="wide")

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
# Function to predict solubility
# -----------------------------
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

    # Read ion fixed properties
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
        data0[concentration_col] = data0[concentration_col_wt]

    # Drop columns with '_wt%' suffix
    data0 = data0.drop(columns=[f'{ion}_concentration_wt%' for ion in ion_properties.keys()])

    # Pure water solubility model
    file_inputs = 'pure_water_solubility.pkl'
    with open(file_inputs, 'rb') as f_pure:
        model_pure = pickle.load(f_pure)
        sc1 = model_pure['scaler']
        model1 = model_pure['model']
        sc1_features = sc1.get_feature_names_out()
        Inputs = Inputs[sc1_features]

    X_input1 = sc1.transform(Inputs)

    # CO2 brine solubility model
    file_inputs1 = 'CO2_Brine_solubility.pkl'
    with open(file_inputs1, 'rb') as f_brine:
        model_brine = pickle.load(f_brine)
        sc2 = model_brine['scaler']
        model2 = model_brine['model']
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
st.title("CO₂ Solubility in Brine Calculator")

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
# UI logic
# -----------------------------
file = st.file_uploader("Upload the CSV file", type=['csv'])

if file is not None:
    try:
        # Load the data
        data = pd.read_csv(file)

        # Display the loaded data
        st.subheader("Loaded Data:")
        st.dataframe(data, use_container_width=True)

        # Automatically run prediction
        results = predict_solubility(data)

        # Display prediction results
        st.subheader("Prediction Results:")
        st.dataframe(results, use_container_width=True)

        # Direct download button
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv_data,
            file_name="CO2_Solubility_Results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Upload a CSV file to show the prediction results.")

st.divider()
