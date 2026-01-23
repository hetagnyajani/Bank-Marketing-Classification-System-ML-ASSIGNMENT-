import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("Bank Marketing Classification System")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

# -----------------------------
# Select model
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "kNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # -----------------------------
    # Load preprocessor
    # -----------------------------
    preprocessor = joblib.load("model/preprocessor.pkl")
    X_processed = preprocessor.transform(data)

    # Handle GaussianNB dense conversion
    if model_name == "Naive_Bayes":
        X_processed = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed

    # -----------------------------
    # Load model
    # -----------------------------
    model = joblib.load(f"model/{model_name}.pkl")
    
    # -----------------------------
    # Predict probabilities
    # -----------------------------
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_processed)[:, 1]
    else:
        prob = [None] * len(data)
    
    predictions = model.predict(X_processed)

    # -----------------------------
    # Combine predictions & probability
    # -----------------------------
    results_df = pd.DataFrame({
        "Prediction": predictions,
        "Probability_Deposit_Yes": prob
    })

    # -----------------------------
    # Display table nicely using HTML
    # -----------------------------
    st.subheader("Predictions with Probabilities")

    # Center the text and use a color gradient for probability
    def color_gradient(val):
        # Green (high) → Yellow → Red (low)
        if val is None:
            return ''
        red = int((1 - val) * 255)
        green = int(val * 255)
        return f'background-color: rgb({red}, {green}, 0); color: black; text-align: center'

    styled_df = results_df.style.applymap(color_gradient, subset=['Probability_Deposit_Yes']) \
                                .set_properties(**{'text-align': 'center'}) \
                                .format({"Probability_Deposit_Yes": "{:.2f}"})

    st.dataframe(styled_df, height=400)

    st.markdown("""
    **Color Gradient Explanation:**  
    - 🟥 Red shades → Low probability  
    - 🟨 Yellow shades → Medium probability  
    - 🟩 Green shades → High probability
    """)

    # -----------------------------
    # Plotly chart with proper axes and size
    # -----------------------------
    st.subheader("Prediction Probability Distribution")

    fig = px.bar(
        results_df,
        x=results_df.index,
        y="Probability_Deposit_Yes",
        color="Probability_Deposit_Yes",
        color_continuous_scale="RdYlGn_r",
        labels={"x": "Sample Index", "Probability_Deposit_Yes": "Probability of Deposit = Yes"},
        height=400
    )

    fig.update_layout(coloraxis_colorbar=dict(title="Probability"),
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
