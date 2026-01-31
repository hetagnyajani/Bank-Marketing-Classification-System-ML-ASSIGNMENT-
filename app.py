import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("Bank Marketing Classification System")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} features")

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
    # Display table nicely
    # -----------------------------
    st.subheader("Predictions with Probabilities")

    def color_gradient(val):
        if val is None:
            return ''
        
        if val <= 0.5:
            # Red to Yellow
            red = 255
            green = int((val / 0.5) * 255)
        else:
            # Yellow to Green
            red = int((1 - (val - 0.5)/0.5) * 255)
            green = 255
        return f'background-color: rgb({red}, {green}, 0); color: black; text-align: center'

    styled_df = results_df.style.applymap(
        color_gradient,
        subset=['Probability_Deposit_Yes']
    ).set_properties(**{'text-align': 'center'}).format(
        {"Probability_Deposit_Yes": "{:.2f}"}
    )

    st.dataframe(styled_df, height=400)

    st.markdown("""
    **Color Gradient Explanation:**  
    - 🟥 Red shades → Low probability  
    - 🟨 Yellow shades → Medium probability  
    - 🟩 Green shades → High probability
    """)

    # -----------------------------
    # Probability Distribution (Improved Visualization)
    # -----------------------------
    st.subheader("Prediction Probability Distribution")

    fig = px.histogram(
        results_df,
        x="Probability_Deposit_Yes",
        nbins=30,
        title="Prediction Probability Distribution",
        labels={
            "Probability_Deposit_Yes": "Probability of Deposit = Yes"
        },
        opacity=0.8
    )

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Number of Customers",
        bargap=0.05
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to continue.")
