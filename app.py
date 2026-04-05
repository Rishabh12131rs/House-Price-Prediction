from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the trained model (Ensure you ran model.py to generate this!)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model.pkl: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture basic inputs from the form
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Advanced Tier Detection (The "Price Engine")
        # 0: Rural/Village, 1: Standard (Jabalpur), 2: Prime (Indore/Ahmedabad)
        tier = 0 
        location_label = "Rural/Village"
        city_display = "Selected Area"

        if lat and lng:
            try:
                # Call Nominatim API for Reverse Geocoding
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                response = requests.get(geo_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
                
                # Get the full address string for keyword matching
                full_address = str(response.get('display_name', '')).lower()
                addr_details = response.get('address', {})
                city_display = addr_details.get('city') or addr_details.get('town') or addr_details.get('village') or "Location Detected"

                # Check for Prime Tier (Tier 2)
                if any(city in full_address for city in ['indore', 'ahmedabad', 'mumbai', 'surat', 'pune', 'delhi']):
                    tier = 2
                    location_label = "Prime City Area"
                # Check for Standard Tier (Tier 1)
                elif any(city in full_address for city in ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot']):
                    tier = 1
                    location_label = "Standard City Area"
                
            except Exception as e:
                print(f"Geo-API Error: {e}")
                # Failsafe: if API fails, keep tier at 0

        # 3. Model Prediction
        # Features MUST be in this exact order: sqft, beds, baths, tier, prop_type, furnish
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        prediction = model.predict(features)
        val = prediction[0]

        # Apply Luxury Multiplier (15% boost if amenities checked)
        if request.form.get('amenities'):
            val *= 1.15

        # 4. Result Formatting (Lakhs vs Crores)
        if val >= 100:
            formatted_price = f"₹{round(val/100, 2)} Crore"
        else:
            formatted_price = f"₹{round(val, 2)} Lakh"

        # Calculate Price per Sq Ft for the UI
        rate = int((val * 100000) / sqft) if sqft > 0 else 0

        # Return a rich HTML string for the AJAX result box
        return f"""
            <div style='color: #10b981; font-size: 26px; font-weight: 800;'>{formatted_price}</div>
            <div style='color: #94a3b8; font-size: 14px; margin-top: 5px;'>
                <i class='fas fa-map-marker-alt'></i> {city_display.title()} <b>({location_label})</b>
            </div>
            <div style='margin-top: 10px; border-top: 1px solid #334155; padding-top: 10px; font-size: 12px; display: flex; justify-content: space-between;'>
                <span>Avg Rate: ₹{rate}/sqft</span>
                <span>Tier Score: {tier}/2</span>
            </div>
        """

    except Exception as e:
        return f"<div style='color: #ef4444;'>Prediction Error: {str(e)}</div>"

if __name__ == "__main__":
    # Critical for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)