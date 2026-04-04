from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io, base64, os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all 5 inputs from the form
    input_data = [float(x) for x in request.form.values()]
    features = [np.array(input_data)]
    
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated Market Value: ${output:,}')

@app.route('/visualize')
def visualize():
    # Simple trend visualization
    sqft = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
    price = [120000, 180000, 210000, 290000, 340000, 320000, 450000]
    plt.figure(figsize=(6,4), facecolor='#f8f9fc')
    plt.scatter(sqft, price, color='#4e73df')
    plt.plot(sqft, price, color='#e74a3b', linewidth=2)
    plt.title('Price vs Square Footage Trend')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('index.html', plot_url=plot_url)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)