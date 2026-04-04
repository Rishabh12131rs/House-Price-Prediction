from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get Form Data
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0))
        furnish = float(request.form.get('furnish', 0))
        lat = request.form.get('lat')
        lng = request.form.get('lng')

        # 2. Tier Detection (0: Village, 1: Standard City, 2: Prime City)
        tier = 0 
        detected_loc = "Rural/Unknown Area"

        if lat and lng:
            lat_f = float(lat)
            lng_f = float(lng)

            try:
                # API Call to get Address
                geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}"
                response = requests.get(geo_url, headers={'User-Agent': 'HousePriceApp/1.0'}, timeout=3).json()
                address = response.get('display_name', '').lower()
                detected_loc = address.split(',')[0].title()

                # Logic based on Address Keywords
                if any(city in address for city in ['indore', 'ahmedabad', 'mumbai', 'surat', 'pune']):
                    tier = 2
                elif any(city in address for city in ['jabalpur', 'bhopal', 'gwalior', 'vadodara', 'rajkot']):
                    tier = 1
                
            except:
                # FAILSAFE: If API fails, use Coordinate Ranges
                # Jabalpur approx range
                if 23.0 <= lat_f <= 23.3 and 79.8 <= lng_f <= 80.0:
                    tier = 1
                    detected_loc = "Jabalpur Region"
                # Indore approx range
                elif 22.6 <= lat_f <= 22.8 and 75.7 <= lng_f <= 76.0:
                    tier = 2
                    detected_loc = "Indore Region"

        # 3. Apply the Prediction
        # Features: sqft, beds, baths, tier, prop_type, furnish
        features = np.array([[sqft, beds, baths, tier, prop_type, furnish]])
        prediction = model.predict(features)
        val = prediction[0]

        # Apply Luxury Premium for Amenities
        if request.form.get('amenities'):
            val *= 1.15

        # 4. Format Result
        if val >= 100:
            formatted_price = f"₹{round(val/100, 2)} Crore"
        else:
            formatted_price = f"₹{round(val, 2)} Lakh"

        return f"<b>Location Detected:</b> {detected_loc}<br><b>Estimated Value:</b> {formatted_price}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)