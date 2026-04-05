import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# 2026 Indian Market Dataset
# tier: 0=Rural, 1=Jabalpur/Bhopal, 2=Indore/Ahmedabad/Mumbai
data = {
    'sqft': [600, 1000, 1200, 1500, 2000, 2500, 800, 1100, 1000, 1500, 3000, 4500, 1300, 900],
    'beds': [1, 2, 2, 3, 3, 4, 2, 2, 2, 3, 4, 5, 2, 2],
    'baths': [1, 1, 2, 2, 3, 3, 1, 2, 2, 2, 4, 5, 2, 1],
    'tier': [0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 1, 0], 
    'prop_type': [0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0], 
    'furnish': [0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0],
    'price': [28, 45, 62, 88, 145, 225, 32, 52, 75, 115, 320, 650, 68, 38] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Train Advanced Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the updated brain
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Advanced Model Trained with 6 Features!")