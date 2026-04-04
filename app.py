from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get basic numbers
    sqft = float(request.form['sqft'])
    beds = float(request.form['beds'])
    
    # 2. Get Area Type (City/Village)
    area_type = request.form['area_type']
    is_city = 1 if area_type == 'city' else 0
    
    # 3. Get Location (One-Hot Encoding logic)
    loc = request.form['location']
    loc_jabalpur, loc_gujarat, loc_other = 0, 0, 0
    if loc == 'jabalpur': loc_jabalpur = 1
    elif loc == 'gujarat': loc_gujarat = 1
    else: loc_other = 1

    # 4. Combine all into one array
    features = np.array([[sqft, beds, is_city, loc_jabalpur, loc_gujarat, loc_other]])
    
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Smart Valuation: ${output:,}')

if __name__ == "__main__":
    app.run(debug=True)