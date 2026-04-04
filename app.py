from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model (Ensuring it handles the 6 features: sqft, beds, baths, type, furnish, amenities)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}")

@app.route('/')
def home():
    # This loads the initial website with the map
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture inputs from the AJAX request
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0)) # 0:Flat, 1:House, 2:Villa
        furnish = float(request.form.get('furnish', 0))    # 0:Unfurnished, 1:Semi, 2:Full
        
        # Checkbox logic
        amenities = 1.0 if request.form.get('amenities') else 0.0
        
        # 2. Prepare features for ML Model
        features = np.array([[sqft, beds, baths, prop_type, furnish, amenities]])
        
        # 3. Predict
        prediction = model.predict(features)
        val = prediction[0]
        
        # 4. Indian Currency Logic (Lakhs vs Crores)
        if val >= 100:
            formatted_price = f"₹{round(val/100, 2)} Crore"
        elif val > 0:
            formatted_price = f"₹{round(val, 2)} Lakh"
        else:
            formatted_price = "Evaluation Pending"

        # IMPORTANT: We return JUST the text string here. 
        # This prevents the map from refreshing!
        return f"Estimated Market Price: {formatted_price}"

    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    # Required for Render Deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)