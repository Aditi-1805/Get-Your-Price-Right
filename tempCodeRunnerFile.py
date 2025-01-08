from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
import pandas as pd
import pickle
import json

app = Flask(__name__)
car = pd.read_csv('Cleaned_Car_data.csv')
model = pickle.load(open('LinearRegressionModel.pkl','rb'))

@app.route('/')
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
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame(columns=['car_model', 'company', 'year', 'kms_driven', 'fuel_type'], data=[[car_model, company, year, kms_driven, fuel_type]]))
    print(prediction)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)