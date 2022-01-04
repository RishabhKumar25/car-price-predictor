from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the model
model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('cleanedcardata.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(),reverse = True)
    fuel_type = car['fuel_type'].unique()
    return render_template('predict.html',companies = companies,car_models=car_models,years=years,fuel_type=fuel_type)

@app.route('/result',methods=['POST'])
def result():
    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__ == "__main__":
    app.run(debug=True)
