# import dependencies
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

# initialize the flask app
app = Flask(__name__)

# load logistic regression model, scaler, and imputer
model = pickle.load(open('log_regression.pkl', 'rb'))
x_scaler = pickle.load(open('x_scaler.pkl', 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

# define the app
@app.route('/')
def home():
    return render_template('index.html')

# to use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    # Input variables with default values if empty
    applicant_income_input = request.form.get('ApplicantIncome', '').strip()
    co_applicant_income_input = request.form.get('CoapplicantIncome', '').strip()
    loan_amount_input = request.form.get('LoanAmount', '').strip()
    loan_amount_term_input = request.form.get('Loan_Amount_Term', '').strip()
    credit_history_input = request.form.get('Credit_History', '').strip()

    # Check if any required field is empty
    if not applicant_income_input or not co_applicant_income_input or not loan_amount_input or not loan_amount_term_input or not credit_history_input:
        return render_template('index.html', error="Please fill in all required fields with valid values.")

    # Convert input values to float
    try:
        applicant_income_input = float(applicant_income_input)
        co_applicant_income_input = float(co_applicant_income_input)
        loan_amount_input = float(loan_amount_input) / 1000
        loan_amount_term_input = float(loan_amount_term_input)
        credit_history_input = float(credit_history_input)
    except ValueError:
        return render_template('index.html', error="Please enter valid numbers for income and loan details.")

    # Process other inputs
    gender_input = request.form['Gender']
    gender_male = 1 if gender_input == "Male" else 0
    gender_female = 1 - gender_male

    married_input = request.form['Married']
    married_yes = 1 if married_input == "Y" else 0
    married_no = 1 - married_yes

    dependents_input = request.form['Dependents']
    dependents_0 = 1 if dependents_input == "0" else 0
    dependents_1 = 1 if dependents_input == "1" else 0
    dependents_2 = 1 if dependents_input == "2" else 0
    dependents_3 = 1 if dependents_input == "3+" else 0

    education_input = request.form['Education']
    education_graduate = 1 if education_input == "Graduate" else 0
    education_not_graduate = 1 - education_graduate

    self_employed_input = request.form['Self_Employed']
    self_employed_yes = 1 if self_employed_input == "Yes" else 0
    self_employed_no = 1 - self_employed_yes

    property_area_input = request.form['Property_Area']
    property_area_urban = 1 if property_area_input == "Urban" else 0
    property_area_semiurban = 1 if property_area_input == "Semiurban" else 0
    property_area_rural = 1 - (property_area_urban + property_area_semiurban)

    # Prepare input data for prediction
    predictions_df = pd.DataFrame({
        "ApplicantIncome": [applicant_income_input],
        "CoapplicantIncome": [co_applicant_income_input],
        "LoanAmount": [loan_amount_input],
        "Loan_Amount_Term": [loan_amount_term_input],
        "Credit_History": [credit_history_input],
        "Gender_Female": [gender_female],
        "Gender_Male": [gender_male],
        "Married_No": [married_no],
        "Married_Yes": [married_yes],
        "Dependents_0": [dependents_0],
        "Dependents_1": [dependents_1],
        "Dependents_2": [dependents_2],
        "Dependents_3+": [dependents_3],
        "Education_Graduate": [education_graduate],
        "Education_Not Graduate": [education_not_graduate],
        "Self_Employed_No": [self_employed_no],
        "Self_Employed_Yes": [self_employed_yes],
        "Property_Area_Rural": [property_area_rural],
        "Property_Area_Semiurban": [property_area_semiurban],
        "Property_Area_Urban": [property_area_urban]
    })

    # Debugging: Print the input DataFrame
    print("Input DataFrame:")
    print(predictions_df)

    # Apply imputer to handle missing values
    predictions_df_imputed = imputer.transform(predictions_df)

    # Scale the input data
    x_test_scaled = x_scaler.transform(predictions_df_imputed)

    # Debugging: Print the scaled input data
    print("Scaled Input Data:")
    print(x_test_scaled)

    # Make prediction
    prediction = model.predict_proba(x_test_scaled)
    probability_of_approval = prediction[0][1]

    # Debugging: Print the prediction output
    print("Model Prediction (Probabilities):")
    print(prediction)

    # Format probability
    formatted_probability = "{:.2f}%".format(probability_of_approval * 100)

    # Choose the result page based on the probability
    page = "approve.html" if probability_of_approval > 0.60 else "denied.html"
    return render_template(page, prediction_text='Probability: {}'.format(formatted_probability))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)
