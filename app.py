from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET','POST'])
def index():
    companies = sorted(car.company.unique())
    car_models = sorted(car.name.unique())
    year_purchased = sorted(car.year.unique(), reverse = True)
    fuel_type = sorted(car.fuel_type.unique()) 
    company_to_models = car.groupby('company')['name'].apply(list).to_dict()

    car_models_json = json.dumps(company_to_models)

    return render_template(
        'index.html', 
        companies = companies, 
        car_models = car_models,
        car_models_json=car_models_json, 
        years = year_purchased, 
        fuel_type =fuel_type
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))
        
        with open('LinearRegressionModel.pkl', 'rb') as f:
            model = pickle.load(f)

        # Fetching user input
        user_input = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Predict the price
        predicted_price = model.predict(user_input)
        print("Predicted Price:", predicted_price[0])

        return str(np.round(predicted_price[0],2))
    
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)