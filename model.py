import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Features: Sqft, Beds, Baths, PropType(0-2), Furnish(0-2), Amenities(0/1)
# Data represents prices in Indian Rupees (in Lakhs)
data = {
    'sqft': [800, 1200, 2500, 1500, 3500, 1000],
    'beds': [1, 2, 3, 2, 4, 2],
    'baths': [1, 2, 3, 2, 4, 1],
    'prop_type': [0, 0, 1, 0, 2, 0], # 0:Flat, 1:House, 2:Villa
    'furnish': [0, 1, 2, 1, 2, 0],   # 0:Unfurnished, 1:Semi, 2:Full
    'amenities': [0, 1, 1, 0, 1, 0],
    'price_lakhs': [40, 65, 180, 85, 350, 55] 
}
df = pd.DataFrame(data)

X = df.drop('price_lakhs', axis=1)
y = df['price_lakhs']

model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Advanced Indian Market Model Trained!")