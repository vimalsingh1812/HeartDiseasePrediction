import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the data
heart_df = pd.read_csv('heart_disease_data.csv')

# Split the data into features and target
X = heart_df.drop('target', axis=1)
Y = heart_df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit interface
st.title("Heart Disease Prediction App")

# Get user input through sidebar
st.sidebar.header("Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", (1, 0))
    cp = st.sidebar.selectbox("Chest Pain Type (0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal Pain, 3=Asymptomatic)", (0, 1, 2, 3))
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=177)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL (1=True, 0=False)", (1, 0))
    restecg = st.sidebar.selectbox("Resting ECG Results (0=Normal, 1=ST-T wave abnormality, 2=Left Ventricular Hypertrophy)", (0, 1, 2))
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=120)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", (1, 0))
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=2.5)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0=Upsloping, 1=Flat, 2=Downsloping)", (0, 1, 2))
    ca = st.sidebar.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", (1, 2, 3))

    # Store user input in a dictionary and convert to a DataFrame
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input
input_data = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_data)

# Predict the output using the trained model
prediction = model.predict(input_data)

# Display the prediction result
st.subheader('Prediction')
if prediction[0] == 0:
    st.write("The person doesn't have heart disease")
else:
    st.write("The person does have heart disease")

# Display model accuracy
st.subheader("Model Accuracy")
train_accuracy = accuracy_score(model.predict(X_train), y_train)
test_accuracy = accuracy_score(model.predict(X_test), y_test)
st.write(f"Training Accuracy: {train_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")