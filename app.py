from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model we created in Step 3
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the HTML form
    sqft = float(request.form['sqft'])
    rooms = float(request.form['rooms'])
    
    # Arrange them for the model
    features = np.array([[sqft, rooms]])
    
    # Make prediction
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated Price: ${output}')

if __name__ == "__main__":
    app.run(debug=True)