import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
data = pd.read_csv("yield_df.csv")

# Split the dataset into features (X) and target variable (y)
X = data[['pesticides_tonnes', 'average_rain_fall_mm_per_year', 'avg_temp']]
y = data['hg/ha_yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to evaluate
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor())
]

best_model = None
best_model_mse = float("inf")

# Train and evaluate the models
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} Mean Squared Error: {mse}")
    
    if mse < best_model_mse:
        best_model = model
        best_model_mse = mse

# Save the best model as a pickle file
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)



import pandas as pd
import pickle

# Load the saved Linear Regression model from the pickle file
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load new data for prediction
new_data = pd.DataFrame({
    'pesticides_tonnes': [150],
    'average_rain_fall_mm_per_year': [2000],
    'avg_temp': [20]
})

# Make predictions using the loaded model
predictions = model.predict(new_data)

# Print the predictions
print("Yield Predictions:")
print(predictions)
