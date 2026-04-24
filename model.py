import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# Calibrated 2026 Dataset with HUGE price differences for testing
# tier: 0 (Rural), 1 (Jabalpur/Bhopal), 2 (Indore/Ahmedabad)
data = {
    'sqft': [1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 800, 800, 3000],
    'beds': [2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 4],
    'baths': [2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 4],
    'tier': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 2], 
    'prop_type': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2], 
    'furnish': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    # Notice the massive jumps: 25L (Rural) -> 45L (Jabalpur) -> 75L (Indore)
    'price': [25, 45, 75, 40, 65, 110, 60, 95, 160, 18, 55, 350] 
}

df = pd.DataFrame(data)
X = df.drop('price', axis=1)
y = df['price']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the brain
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ AI BRAIN UPDATED: Location-based pricing is now active.")