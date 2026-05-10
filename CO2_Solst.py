import pandas as pd
import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Page configuration
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
# Title / Header
# -----------------------------
st.title("CO₂ Solubility in Brine Calculator")

st.markdown(
    "Product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium"
)

st.markdown("**Based on the work shown in:**")
st.markdown(
    "- Ratnakar, R. R., Chaubey, V., & Dindoruk, B. (2023). "
    "A novel computational strategy to estimate CO₂ solubility in brine solutions "
    "for CCUS applications. *Applied Energy, 342*, 121134."
)

st.markdown(
    "Calculates CO₂ solubility in brine for different salt compositions at specified "
    "pressure and temperature conditions using machine-learning-based models."
)

st.markdown(
    "1) **Download the example input CSV template**: "
    "[Click here](https://drive.google.com/file/d/1-UV3qfaFh39qHkoAJYZBhtUrJhPZHE54/view?usp=sharing)"
)

st.markdown(
    "2) **Reference paper**: "
    "[Click here](https://www.sciencedirect.com/science/article/pii/S0306261923004981)"
)

st.divider()

# -----------------------------
# Function to predict solubility
# -----------------------------
def predict_solubility(data0):
    P = data0["P,Psia"]
    T = data0["T,F"]

    pressures_converted = P / 14.504
    temp = (T - 32) * 5 / 9 + 273.15

    pc = 73.8
    tc = 304.25

    rP = pressures_converted / pc
    rT = temp / tc

    Inputs = pd.DataFrame(
        {
            "rT": np.full_like(pressures_converted, rT),
            "rP": rP,
        }
    )

    ion_properties = {
        "Na": {"charge": 1, "energy": 365},
        "Cl": {"charge": 1, "energy": 340},
        "HCO3": {"charge": 1, "energy": 335},
        "Ca": {"charge": 2, "energy": 1505},
        "CO3": {"charge": 2, "energy": 1315},
        "SO4": {"charge": 2, "energy": 1080},
        "Mg": {"charge": 2, "energy": 1830},
        "K": {"charge": 1, "energy": 295},
    }

    for ion, properties in ion_properties.items():
        charge_col = f"{ion}_charge"
        energy_col = f"{ion}_energy"
        concentration_col_wt = f"{ion}_concentration_wt%"
        concentration_col = f"{ion}_concentration"

        data0[charge_col] = np.where(
            data0[concentration_col_wt] != 0,
            properties["charge"],
            0,
        )

        data0[energy_col] = np.where(
            data0[concentration_col_wt] != 0,
            properties["energy"],
            0,
        )

        data0[concentration_col] = data0[concentration_col_wt]

    data0 = data0.drop(
        columns=[f"{ion}_concentration_wt%" for ion in ion_properties.keys()]
    )

    # Load pure-water solubility model
    file_inputs = "pure_water_solubility.pkl"
    with open(file_inputs, "rb") as f_pure:
        model_pure = pickle.load(f_pure)
        sc1 = model_pure["scaler"]
        model1 = model_pure["model"]

        sc1_features = sc1.get_feature_names_out()
        Inputs = Inputs[sc1_features]

    X_input1 = sc1.transform(Inputs)

    # Load brine solubility model
    file_inputs1 = "CO2_Brine_solubility.pkl"
    with open(file_inputs1, "rb") as f_brine:
        model_brine = pickle.load(f_brine)
        sc2 = model_brine["scaler"]
        model2 = model_brine["model"]

        sc2_features = sc2.get_feature_names_out()
        data0 = data0[sc2_features]

    X_inputb = sc2.transform(data0)

    sol = model1.predict(X_input1)
    solb = model2.predict(X_inputb)

    results = data0.copy()
    results["Brine to Pure Water Solubility Ratio"] = solb
    results["Pure Water Solubility (Mole Frac)"] = sol
    results["CO2 Solubility in Brine at P&T (Mole Frac)"] = sol * solb

    return results

# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("Upload the input CSV file here", type=["csv"])

if uploaded is not None:
    try:
        st.session_state["input_df"] = pd.read_csv(uploaded)
        st.success("CSV loaded. Click **Run prediction** to generate results.")
        st.dataframe(st.session_state["input_df"], use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to enable prediction.")

run_clicked = st.button(
    "Run prediction",
    type="primary",
    disabled=("input_df" not in st.session_state),
)

if run_clicked:
    try:
        result_df = predict_solubility(st.session_state["input_df"].copy())
        st.session_state["result_df"] = result_df
        st.success("Prediction complete.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

if "result_df" in st.session_state:
    st.subheader("Results")
    st.dataframe(st.session_state["result_df"], use_container_width=True)

    csv_bytes = st.session_state["result_df"].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CO₂ solubility results",
        data=csv_bytes,
        file_name=f"CO2_Solubility_Results-{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
    )
