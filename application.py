from flask import Flask, jsonify, request, render_template, flash
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

application = Flask(__name__)
app = application
app.secret_key = 'your-secret-key-here'  # Change this in production

# Load models with error handling
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print(f"Model loaded. Number of features expected: {ridge_model.coef_.shape[0]}")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    ridge_model = None
    standard_scaler = None
except Exception as e:
    print(f"Error loading models: {e}")
    ridge_model = None
    standard_scaler = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Check if models are loaded
            if ridge_model is None or standard_scaler is None:
                flash('Model files not found. Please ensure model files are in the models/ directory.', 'error')
                return render_template('home.html')
            
            # Validate and extract form data
            # Note: The model was trained on these features in this exact order after preprocessing:
            # ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
            # BUI and DC were dropped due to high correlation
            required_fields = [
                'Temperature', 'RH', 'Ws', 'Rain', 
                'FFMC', 'DMC', 'ISI', 'Classes', 'Region'
            ]
            
            form_data = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value:
                    flash(f'{field} is required.', 'error')
                    return render_template('home.html')
                
                try:
                    # Convert to float for numerical features, handle Classes separately
                    if field != 'Classes':
                        form_data[field] = float(value)
                    else:
                        form_data[field] = 0 if 'not' in value.lower() else 1
                except ValueError:
                    flash(f'{field} must be a valid number.', 'error')
                    return render_template('home.html')
            
            # Create input array for prediction with features in the correct order
            input_data = [[
                form_data['Temperature'], form_data['RH'], form_data['Ws'], 
                form_data['Rain'], form_data['FFMC'], form_data['DMC'], 
                form_data['ISI'], form_data['Classes'], form_data['Region']
            ]]
            
            # Scale the data and make prediction
            new_data_scaled = standard_scaler.transform(input_data)
            result = ridge_model.predict(new_data_scaled)
            
            # Round result to 2 decimal places for better display
            prediction = round(result[0], 2)
            
            flash(f'Prediction successful! FWI: {prediction}', 'success')
            return render_template('home.html', results=prediction)
            
        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return render_template('home.html')
    
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
