import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# 1. More realistic data (Sqft, Beds, Baths, Age, Parking)
data = {
    'sqft': [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    'beds': [1, 2, 3, 3, 4, 4, 5],
    'baths': [1, 1, 2, 2, 3, 3, 4],
    'age': [10, 5, 15, 2, 8, 20, 1],
    'parking': [0, 1, 1, 2, 2, 1, 2],
    'price': [120000, 180000, 210000, 290000, 340000, 320000, 450000]
}
df = pd.DataFrame(data)

# 2. Features (X) and Target (y)
X = df[['sqft', 'beds', 'baths', 'age', 'parking']]
y = df['price']

# 3. Train
model = LinearRegression()
model.fit(X, y)

# 4. Save
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Brain Updated with 5 Features!")