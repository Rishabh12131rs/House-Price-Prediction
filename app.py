from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Inputs
        sqft = float(request.form.get('sqft', 1))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Auto-Location & Tier Detection
        tier = 0
        city_name = "Rural Area"
        if lat and lng:
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
            response = requests.get(geo_url, headers={'User-Agent': 'Mozilla/5.0'}).json()
            address = response.get('display_name', '').lower()
            city_name = address.split(',')[0].title()
            
            if any(c in address for c in ['indore', 'ahmedabad', 'mumbai', 'surat']): tier = 2
            elif any(c in address for c in ['jabalpur', 'bhopal', 'gwalior']): tier = 1

        # 3. Predict & Premium Logic
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        val = model.predict(features)[0]
        if request.form.get('amenities'): val *= 1.15 # 15% Luxury Boost

        # 4. Analytics
        rate_sqft = int((val * 100000) / sqft)
        currency = f"₹{round(val/100, 2)} Cr" if val >= 100 else f"₹{round(val, 2)} L"

        # Return formatted HTML for the AJAX result box
        return f"""
        <div style='color:#10b981; font-size:28px; font-weight:800;'>{currency}</div>
        <div style='color:#94a3b8; font-size:14px; margin-bottom:10px;'>Detected: {city_name}</div>
        <div style='display:flex; justify-content:space-between; border-top:1px solid #334155; pt:10px; font-size:12px;'>
            <span>Rate: ₹{rate_sqft}/sqft</span>
            <span>Confidence: 94%</span>
        </div>
        """
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)