import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Expanded Data with Location and Area Type
data = {
    'sqft': [1000, 2000, 1500, 800, 3000, 1200],
    'beds': [1, 3, 2, 1, 4, 2],
    'is_city': [1, 1, 1, 0, 1, 0],  # 1 for City, 0 for Village
    'loc_jabalpur': [1, 0, 0, 1, 0, 0],
    'loc_gujarat': [0, 1, 1, 0, 0, 0],
    'loc_other': [0, 0, 0, 0, 1, 1],
    'price': [150000, 250000, 200000, 80000, 350000, 110000]
}
df = pd.DataFrame(data)

# 2. Features and Target
X = df.drop('price', axis=1)
y = df['price']

model = LinearRegression()
model.fit(X, y)

# 3. Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Brain Updated: Now understands City/Village and Locations!")