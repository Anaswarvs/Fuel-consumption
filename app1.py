from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('C:/Users/HP/Desktop/task/regression/model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/HP/Desktop/task/regression/minimax.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()

    # Prepare the data for prediction
    data = {
        'ENGINE SIZE': form_data['ENGINE SIZE'],
        'CYLINDERS': form_data['CYLINDERS'],
        'FUEL CONSUMPTION': form_data['FUEL CONSUMPTION'],
        'HWY (L/100 km)': form_data['HWY (L/100 km)'],
        'COMB (L/100 km)': form_data['COMB (L/100 km)'],
        'COMB (mpg)': form_data['COMB (mpg)'],
        'Transmission_Category': form_data['Transmission_Category'],
        'Vehicle Class Category': form_data['Vehicle Class Category']
    }

    # Convert data to DataFrame with exact column order
    input_df = pd.DataFrame([data], columns=[
        'ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'HWY (L/100 km)', 
        'COMB (L/100 km)', 'COMB (mpg)', 'Transmission_Category', 
        'Vehicle Class Category'
    ])

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(scaled_input)[0]

    # Render the result on a web page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
