from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the brain
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Form Data
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Logic for Tier Detection
        tier = 1 # Default to Standard City
        city_name = "Selected Area"
        
        if lat and lng:
            try:
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                response = requests.get(geo_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3).json()
                address = response.get('display_name', '').lower()
                city_name = address.split(',')[0].title()
                
                if any(c in address for c in ['indore', 'ahmedabad', 'mumbai', 'surat']): tier = 2
                elif any(c in address for c in ['jabalpur', 'bhopal', 'gwalior']): tier = 1
                else: tier = 0
            except:
                pass # Use default tier if API fails

        # 3. ML Prediction (The order must match model.py!)
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        val = model.predict(features)[0]
        
        if request.form.get('amenities'): val *= 1.15

        # 4. Final Formatting
        currency = f"₹{round(val/100, 2)} Crore" if val >= 100 else f"₹{round(val, 2)} Lakh"
        rate = int((val * 100000) / sqft) if sqft > 0 else 0

        # Return string to AJAX
        return f"<b>{city_name}</b><br><span style='color:#10b981; font-size:24px;'>{currency}</span><br>Rate: ₹{rate}/sqft"

    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)