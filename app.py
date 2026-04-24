from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Inputs
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Advanced Tier Logic
        tier = 0 
        location_label = "Rural Area"
        city_display = "Location Pinpointed"

        if lat and lng:
            try:
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                # Nominatim requires a User-Agent to avoid 403 errors
                response = requests.get(geo_url, headers={'User-Agent': 'SmartPriceAI_v2'}).json()
                
                full_address = response.get('display_name', '').lower()
                addr_data = response.get('address', {})
                city_display = addr_data.get('city') or addr_data.get('town') or addr_data.get('village') or "Selected Area"

                # Fuzzy Keyword Search
                if any(c in full_address for c in ['indore', 'ahmedabad', 'mumbai', 'surat', 'pune']):
                    tier = 2
                    location_label = "Prime City Area"
                elif any(c in full_address for c in ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot']):
                    tier = 1
                    location_label = "Standard City Area"
            except:
                pass

        # 3. Predict
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        val = model.predict(features)[0]
        if request.form.get('amenities'): val *= 1.15

        # 4. Result
        currency = f"₹{round(val/100, 2)} Crore" if val >= 100 else f"₹{round(val, 2)} Lakh"
        rate = int((val * 100000) / sqft) if sqft > 0 else 0

        return f"""
            <div style='color: #10b981; font-size: 26px; font-weight: 800;'>{currency}</div>
            <div style='color: #94a3b8; font-size: 14px; margin-top: 5px;'>
                {city_display.title()} <b>({location_label})</b>
            </div>
            <div style='margin-top: 10px; border-top: 1px solid #334155; padding-top: 10px; font-size: 12px; display: flex; justify-content: space-between;'>
                <span>Rate: ₹{rate}/sqft</span>
                <span>Tier: {tier}/2</span>
            </div>
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)