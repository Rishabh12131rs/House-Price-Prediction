from flask import Flask, render_template, request
import pickle
import numpy as np
import requests # Need this for Reverse Geocoding
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
        # 1. Capture basic numbers
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. AUTOMATIC CITY DETECTION (Reverse Geocoding)
        tier = 0 # Default to Rural/Village
        if lat and lng:
            # Call OpenStreetMap Nominatim API
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
            headers = {'User-Agent': 'HousePriceApp/1.0'}
            response = requests.get(geo_url, headers=headers).json()
            
            address = response.get('address', {})
            city = address.get('city') or address.get('town') or address.get('village', '').lower()
            
            # Auto-assign tiers based on detected city
            if any(x in city for x in ['indore', 'ahmedabad', 'mumbai', 'delhi']):
                tier = 2 # Prime Tier
            elif any(x in city for x in ['jabalpur', 'bhopal', 'gandhinagar']):
                tier = 1 # Standard City Tier
        
        # 3. Predict using the detected Tier
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        prediction = model.predict(features)
        val = prediction[0]
        
        # Add a 10% premium for amenities
        if request.form.get('amenities'): val *= 1.1

        # 4. Currency Formatting
        res = f"₹{round(val/100, 2)} Crore" if val >= 100 else f"₹{round(val, 2)} Lakh"

        return f"Location Detected: {city.title()} | Estimated Price: {res}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)