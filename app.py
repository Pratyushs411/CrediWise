import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Load model and preprocessing objects
# ----------------------------

model = joblib.load("model/loan_model.pkl")
imputer = joblib.load("model/imputer.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Loan Approval Predictor 🚀")

st.markdown("""
Enter the following information to check whether the loan will be **Approved** or **Rejected**.
""")

def user_input_features():
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married', ['Yes', 'No'])
    Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
    ApplicantIncome = st.number_input('Applicant Income', min_value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
    LoanAmount = st.number_input('Loan Amount (in thousands)', min_value=0)
    Loan_Amount_Term = st.number_input('Loan Term (in days)', min_value=0)
    Credit_History = st.selectbox('Credit History', [1.0, 0.0])
    Property_Area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

    return pd.DataFrame([data])

# Get input
input_df = user_input_features()

# One-hot encoding and aligning with training columns
df_encoded = pd.get_dummies(input_df)
df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

# Impute missing values
df_imputed = imputer.transform(df_encoded)

# Predict
prediction = model.predict(df_imputed)[0]
prediction_proba = model.predict_proba(df_imputed)[0]

# ----------------------------
# Show prediction result
# ----------------------------

st.subheader("Prediction Result")
if prediction == 1:
    st.success("🎉 Loan will be **Approved**!")
else:
    st.error("❌ Loan will be **Rejected**.")

# ----------------------------
# Show prediction probabilities
# ----------------------------

st.subheader("Prediction Probabilities")
st.write({
    "Rejected": f"{prediction_proba[0]*100:.2f}%",
    "Approved": f"{prediction_proba[1]*100:.2f}%"
})

# Bar chart of probabilities
st.subheader("📊 Probability Distribution")

fig, ax = plt.subplots()
labels = ['Rejected', 'Approved']
sns.barplot(x=labels, y=prediction_proba, palette="Blues_d", ax=ax)
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)
st.pyplot(fig)

# ----------------------------
# Show feature importances (if available)
# ----------------------------

if hasattr(model, "feature_importances_"):
    st.subheader("🧠 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis", ax=ax2)
    ax2.set_title("Top 10 Important Features")
    st.pyplot(fig2)

# ----------------------------
# Optional: Radar chart comparing user input
# ----------------------------

numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
user_values = input_df[numeric_features].iloc[0].values
mean_values = np.array([5000, 1500, 150, 360])  # Example means; replace with real if available

st.subheader("📡 Radar Chart: Your Input vs Average")

angles = np.linspace(0, 2 * np.pi, len(numeric_features), endpoint=False).tolist()
user_values = np.concatenate((user_values, [user_values[0]]))
mean_values = np.concatenate((mean_values, [mean_values[0]]))
angles += angles[:1]

fig3 = plt.figure(figsize=(6,6))
ax3 = plt.subplot(111, polar=True)
ax3.plot(angles, user_values, label='You', color='blue')
ax3.fill(angles, user_values, alpha=0.25, color='blue')
ax3.plot(angles, mean_values, label='Average', color='orange')
ax3.fill(angles, mean_values, alpha=0.25, color='orange')
ax3.set_thetagrids(np.degrees(angles[:-1]), numeric_features)
plt.legend(loc='upper right')
st.pyplot(fig3)
