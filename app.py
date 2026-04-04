from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the Calibrated Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"❌ Model Load Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get inputs from the No-Refresh (AJAX) form
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        amenities = 1.0 if request.form.get('amenities') else 0.0
        
        # 2. Prepare for Prediction
        features = np.array([[sqft, beds, baths, prop_type, furnish, amenities]])
        
        # 3. Calculate Price
        prediction = model.predict(features)
        val = prediction[0]
        
        # Logic to prevent negative or tiny unrealistic values
        if val < 5: val = sqft * 0.03 # Fallback to 3k per sqft minimum
        
        # 4. Format for Indian Users (Rupee Symbol + Lakh/Crore)
        if val >= 100:
            formatted_price = f"₹{round(val/100, 2)} Crore"
        else:
            formatted_price = f"₹{round(val, 2)} Lakh"

        # Return ONLY the text string so JavaScript can update the UI
        return f"Estimated Market Value: {formatted_price}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)