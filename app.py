from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the brain (Ensure you ran model.py to generate this!)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model.pkl. Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Form Inputs
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Strict City & Tier Detection Logic
        # Tier 0: Rural/Village, Tier 1: Standard (Jabalpur), Tier 2: Prime (Indore)
        tier = 0 
        location_label = "Rural/Village"
        city_display = "Location Pinpointed"

        if lat and lng and lat != "":
            try:
                # Reverse Geocoding via Nominatim
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                # Custom User-Agent is mandatory to prevent being blocked
                response = requests.get(geo_url, headers={'User-Agent': 'SmartPriceAI_v2_Rishabh'}, timeout=5).json()
                
                # Get the full address as a long string for robust keyword matching
                full_address = response.get('display_name', '').lower()
                
                # Extract a readable city/town name for the UI
                addr_data = response.get('address', {})
                city_display = addr_data.get('city') or addr_data.get('town') or addr_data.get('suburb') or addr_data.get('village') or "Selected Area"

                # SMART KEYWORD SEARCH
                # Finds city names even in nested addresses like "ITI Jabalpur"
                prime_cities = ['indore', 'ahmedabad', 'mumbai', 'surat', 'pune', 'delhi', 'bangalore']
                standard_cities = ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot', 'nagpur']

                if any(city in full_address for city in prime_cities):
                    tier = 2
                    location_label = "Prime City Area"
                elif any(city in full_address for city in standard_cities):
                    tier = 1
                    location_label = "Standard City Area"
                
                # Log to Render console for debugging
                print(f"DETECTED: {full_address} | ASSIGNED TIER: {tier}")

            except Exception as geo_err:
                print(f"Geo-API Timeout or Error: {geo_err}")

        # 3. ML Prediction (Expected order: sqft, beds, baths, tier, prop_type, furnish)
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        prediction = model.predict(features)
        val = prediction[0]
        
        # Apply Luxury Premium if amenities checkbox is ticked
        if request.form.get('amenities'):
            val *= 1.15

        # 4. Final Formatting (Lakhs vs Crores)
        if val >= 100:
            formatted_price = f"₹{round(val/100, 2)} Crore"
        else:
            formatted_price = f"₹{round(val, 2)} Lakh"

        # Calculate rate per sqft for the UI
        rate_sqft = int((val * 100000) / sqft) if sqft > 0 else 0

        # Return rich HTML string to the AJAX result box
        return f"""
            <div style='color: #10b981; font-size: 26px; font-weight: 800;'>{formatted_price}</div>
            <div style='color: #94a3b8; font-size: 14px; margin-top: 5px;'>
                <i class='fas fa-map-marker-alt'></i> {city_display.title()} <b>({location_label})</b>
            </div>
            <div style='margin-top: 10px; border-top: 1px solid #334155; padding-top: 10px; font-size: 12px; display: flex; justify-content: space-between;'>
                <span>Rate: ₹{rate_sqft}/sqft</span>
                <span>Tier: {tier}/2</span>
            </div>
        """

    except Exception as e:
        return f"<div style='color: #ef4444;'>Prediction Error: {str(e)}</div>"

if __name__ == "__main__":
    # Fetch port from environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)