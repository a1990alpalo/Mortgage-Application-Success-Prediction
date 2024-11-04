from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models and scaler
logistic_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
deep_learning_model = load_model('deep_learning_model.h5')
scaler = pickle.load(open('x_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and process form data
    form_data = request.form
    applicant_income = float(form_data['ApplicantIncome'])
    co_applicant_income = float(form_data['CoapplicantIncome'])
    loan_amount = float(form_data['LoanAmount'])
    loan_amount_term = float(form_data['Loan_Amount_Term'])
    credit_history = float(form_data['Credit_History'])

    # Encode categorical variables
    gender = 1 if form_data['Gender'] == "Male" else 0
    married = 1 if form_data['Married'] == "Y" else 0
    dependents = form_data['Dependents']
    dependents_0, dependents_1, dependents_2, dependents_3 = 0, 0, 0, 0
    if dependents == "0":
        dependents_0 = 1
    elif dependents == "1":
        dependents_1 = 1
    elif dependents == "2":
        dependents_2 = 1
    else:
        dependents_3 = 1

    education = 1 if form_data['Education'] == "Graduate" else 0
    self_employed = 1 if form_data['Self_Employed'] == "Yes" else 0
    property_area = form_data['Property_Area']
    property_area_rural, property_area_semiurban, property_area_urban = 0, 0, 0
    if property_area == "Urban":
        property_area_urban = 1
    elif property_area == "Semiurban":
        property_area_semiurban = 1
    else:
        property_area_rural = 1

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [co_applicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Gender_Male": [gender],
        "Married_Yes": [married],
        "Dependents_0": [dependents_0],
        "Dependents_1": [dependents_1],
        "Dependents_2": [dependents_2],
        "Dependents_3+": [dependents_3],
        "Education_Graduate": [education],
        "Self_Employed_Yes": [self_employed],
        "Property_Area_Rural": [property_area_rural],
        "Property_Area_Semiurban": [property_area_semiurban],
        "Property_Area_Urban": [property_area_urban]
    })

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction with logistic model and get probability
    prediction_prob = logistic_model.predict_proba(scaled_data)[0][1]
    prediction_text = "accepted" if prediction_prob > 0.6 else "denied"
    formatted_probability = "{:.2f}%".format(prediction_prob * 100)

    # Render the result page with conditional approval status
    return render_template('result.html', 
                           prediction=prediction_text, 
                           probability=formatted_probability)

if __name__ == "__main__":
    app.run(debug=True)



