from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the updated 6-feature model
# Make sure you ran 'python model.py' to generate the new pkl file!
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture inputs from the modern form
        sqft = float(request.form.get('sqft', 0))
        beds = float(request.form.get('beds', 0))
        baths = float(request.form.get('baths', 0))
        prop_type = float(request.form.get('prop_type', 0)) # 0:Flat, 1:House, 2:Villa
        furnish = float(request.form.get('furnish', 0))    # 0:Unfurnished, 1:Semi, 2:Full
        
        # Checkbox logic for amenities
        amenities = 1.0 if request.form.get('amenities') else 0.0
        
        # 2. Prepare the array for the Machine Learning model
        # The order MUST match your model.py features
        features = np.array([[sqft, beds, baths, prop_type, furnish, amenities]])
        
        # 3. Make the prediction
        prediction = model.predict(features)
        val = prediction[0]
        
        # 4. Indian Currency Formatting (Lakhs & Crores)
        if val >= 100:
            # If 100 Lakhs or more, show in Crores
            formatted_price = f"₹{round(val/100, 2)} Crore"
        elif val > 0:
            formatted_price = f"₹{round(val, 2)} Lakh"
        else:
            formatted_price = "Contact Agent for Price"

        return render_template('index.html', prediction_text=f'Estimated Market Price: {formatted_price}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in Calculation: {str(e)}")

if __name__ == "__main__":
    # This part is CRITICAL for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)