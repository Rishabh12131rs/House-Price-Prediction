import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# Calibrated 2026 Dataset
# Tiers: 0:Rural/Village, 1:Standard City (Jabalpur), 2:Prime City (Indore/Ahmedabad)
data = {
    'sqft': [600, 1000, 1200, 1500, 2000, 2500, 800, 1100, 1000, 1500, 3000, 4500, 1300, 900],
    'beds': [1, 2, 2, 3, 3, 4, 2, 2, 2, 3, 4, 5, 2, 2],
    'baths': [1, 1, 2, 2, 3, 3, 1, 2, 2, 2, 4, 5, 2, 1],
    'tier': [0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 1, 0], 
    'prop_type': [0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0], 
    'furnish': [0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0],
    'price': [22, 45, 62, 88, 145, 210, 28, 52, 75, 118, 310, 680, 65, 34] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Random Forest handles the non-linear "price jumps" between cities better
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ AI Brain Calibrated! Model expects 6 features.")