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
import matplotlib.pyplot as plt
import io
import base64

@app.route('/visualize')
def visualize():
    # Simple data for the graph
    sqft = [1000, 1500, 2000, 2500, 3000, 3500]
    price = [100000, 150000, 200000, 230000, 300000, 350000]
    
    plt.figure(figsize=(6,4))
    plt.scatter(sqft, price, color='blue', label='Actual Data')
    plt.plot(sqft, price, color='red', label='Regression Line')
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($)')
    plt.title('House Price Trend')
    plt.legend()
    
    # Save plot to a fake file in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('index.html', plot_url=plot_url)
if __name__ == "__main__":
    app.run(debug=True)