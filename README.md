# Mortgage-Application-Success-Prediction

This project aims to predict the success of mortgage applications using a machine learning model. The project includes a web application built with Flask that allows users to input mortgage application details and receive predictions based on the model. This README file outlines the purpose of the project, the tools and libraries used, and instructions for running the application.

Project Overview
The Mortgage Application Success Prediction project is designed to assist lenders and applicants by predicting the likelihood of a mortgage application being approved. By analyzing various factors (such as applicant income, credit history, loan amount, etc.), the model helps in evaluating applications efficiently. The project includes both a machine learning model and a web interface.

Tools and Libraries Used
1. Machine Learning Model
Scikit-learn: For training and evaluating the machine learning model, including data preprocessing, model selection, and evaluation metrics.
Pandas: To manage and process structured data. It is used for data cleaning, transformation, and exploration.
NumPy: To handle numerical operations and array manipulations, which support data processing in machine learning.
Joblib: For saving and loading the trained model, making it easy to deploy in the Flask application.
2. Web Application
Flask: A lightweight web framework used to create the web interface. Flask manages the routing and user interaction with the model.
HTML/CSS: To design the front-end of the application, allowing users to input data for predictions.
Bootstrap: A CSS framework to make the interface responsive and user-friendly.
3. Other Tools
Git: For version control, tracking changes, and collaborating on the project.
Jupyter Notebook: For initial model development, experimentation, and data analysis.
Project Structure
app.py: The main Flask application file, which handles routes, receives input data, and returns prediction results.
model.pkl: The trained machine learning model saved with Joblib. This model is loaded by the Flask app to generate predictions.
templates/: This folder contains HTML files for the web interface.
static/: This folder includes CSS and other static files for styling.
Getting Started
Prerequisites
Python 3.x
Flask: Install using pip install flask.
Scikit-learn: Install using pip install scikit-learn.
Pandas and NumPy: Install using pip install pandas numpy.
Joblib: Install using pip install joblib.
Running the Application
Clone the repository:

bash
Copy code
git clone https://github.com/a1990alpalo/Mortgage-Application-Success-Prediction.git
Navigate to the project directory:

bash
Copy code
cd Mortgage-Application-Success-Prediction
Run the Flask application:

bash
Copy code
python app.py
Open a web browser and go to http://127.0.0.1:5000 to access the application.

Usage [https://a1990alpalo.github.io/Mortgage-Application-Success-Prediction/]
Enter the required details for the mortgage application in the input fields.
Submit the form to receive a prediction on whether the mortgage application is likely to succeed.
