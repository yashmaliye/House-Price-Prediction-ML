import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='../templates')

# Resolve model path for Vercel's serverless environment
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'linear.pkl')

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract features matching your linear.pkl metadata
            sqft = float(request.form.get('Square_Footage', 0))
            bedrooms = float(request.form.get('Num_Bedrooms', 0))
            bathrooms = float(request.form.get('Num_Bathrooms', 0))
            year_built = float(request.form.get('Year_Built', 0))
            lot_size = float(request.form.get('Lot_Size', 0))
            garage_size = float(request.form.get('Garage_Size', 0))
            quality = float(request.form.get('Neighborhood_Quality', 0))

            features = np.array([[sqft, bedrooms, bathrooms, year_built, lot_size, garage_size, quality]])
            
            if model:
                pred = model.predict(features)[0]
                prediction_text = f"${pred:,.2f}"
            else:
                prediction_text = "Error: Model file (linear.pkl) not loaded."
        except Exception as e:
            prediction_text = f"Error processing input: {str(e)}"

    return render_template('index.html', prediction=prediction_text)

# Vercel requires the app instance to be accessible
if __name__ == '__main__':
    app.run(debug=True)
