from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML template with embedded CSS, modern layout, and shadow effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Estate Price Predictor</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 650px;
        }
        h2 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 24px;
            font-size: 28px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .form-group {
            margin-bottom: 12px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 14px;
        }
        input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        button:active {
            transform: scale(0.98);
        }
        .result-card {
            margin-top: 24px;
            padding: 20px;
            background-color: #e0e7ff;
            border-left: 5px solid var(--primary-color);
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #312e81;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏡 House Price Predictor</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Square_Footage">Square Footage</label>
                    <input type="number" step="any" id="Square_Footage" name="Square_Footage" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bedrooms">Num Bedrooms</label>
                    <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bathrooms">Num Bathrooms</label>
                    <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" required>
                </div>
                <div class="form-group">
                    <label for="Year_Built">Year Built</label>
                    <input type="number" id="Year_Built" name="Year_Built" required>
                </div>
                <div class="form-group">
                    <label for="Lot_Size">Lot Size</label>
                    <input type="number" step="any" id="Lot_Size" name="Lot_Size" required>
                </div>
                <div class="form-group">
                    <label for="Garage_Size">Garage Size</label>
                    <input type="number" id="Garage_Size" name="Garage_Size" required>
                </div>
                <div class="form-group full-width">
                    <label for="Neighborhood_Quality">Neighborhood Quality (1-10)</label>
                    <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" required>
                </div>
            </div>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            Estimated Price: ${{ "%.2f"|format(prediction) }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the correct order required by the model
            features = np.array([[
                float(request.form['Square_Footage']),
                float(request.form['Num_Bedrooms']),
                float(request.form['Num_Bathrooms']),
                float(request.form['Year_Built']),
                float(request.form['Lot_Size']),
                float(request.form['Garage_Size']),
                float(request.form['Neighborhood_Quality'])
            ]])
            
            pred = model.predict(features)
            prediction = float(pred[0])
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
