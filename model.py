import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Create a simple dataset (Square Feet, Bedrooms, Price)
data = {
    'sqft': [1000, 1500, 2000, 2500, 3000, 3500],
    'rooms': [1, 2, 3, 3, 4, 5],
    'price': [100000, 150000, 200000, 230000, 300000, 350000]
}
df = pd.DataFrame(data)

# 2. Split data into Features (X) and Target (y)
X = df[['sqft', 'rooms']]
y = df['price']

# 3. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Save the model to a file named 'model.pkl'
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Brain (Model) has been trained and saved!")