import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Convert the data string to a DataFrame
#data = pd.read_csv(StringIO(data), sep='\t')
data = pd.read_csv("larger_dataset.csv")
# Convert categorical variables to numerical using label encoding
label_encoder = LabelEncoder()
data['Delivery Status'] = label_encoder.fit_transform(data['Delivery Status'])

# Convert other categorical variables to numerical using one-hot encoding
categorical_columns = ['Origin', 'Destination', 'Transport Mode', 'Weather Conditions', 'Season', 'Product Type']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split the data into features (X) and labels (y)
X = data_encoded.drop(columns=['Cost ($)'])
y = data_encoded['Cost ($)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the root mean squared error (RMSE) as a metric of model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Function to predict the cost of delivery for user inputs
def predict_delivery_cost(origin, destination, weather, product, quantity, season, transport_mode):
    # Create a DataFrame with the user inputs
    user_data = pd.DataFrame({
        'Origin': [origin],
        'Destination': [destination],
        'Transport Mode': [transport_mode],
        'Weather Conditions': [weather],
        'Season': [season],
        'Product Type': [product],
        'Quantity': [quantity]
    })
    
    # Encode the user data using one-hot encoding
    user_data_encoded = pd.get_dummies(user_data, columns=categorical_columns, drop_first=True)
    missing_cols = set(X.columns) - set(user_data_encoded.columns)
    
    # Add the missing columns with default value 0
    for col in missing_cols:
        user_data_encoded[col] = 0
    
    # Reorder the columns to match the original order
    user_data_encoded = user_data_encoded[X.columns]
    
    # Make the prediction using the trained model
    cost_prediction = rf_model.predict(user_data_encoded)
    return cost_prediction[0]

# Example usage:
origin_input = "City B"
destination_input = "City E"
weather_input = "Rainy"
product_input = "maize"
quantity_input = 5
season_input = "Summer"
transport_mode_input = "Train"

predicted_cost = predict_delivery_cost(origin_input, destination_input, weather_input, product_input, quantity_input, season_input, transport_mode_input)
print(f"Predicted cost of delivery: ${predicted_cost:.2f}")


# Save the trained model as a pickle file
with open('Logistic_rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)




# Load the trained model from the pickle file
with open('Logistic_rf_model.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)



# Function to predict the cost of delivery for user inputs
def predict_delivery_cost(origin, destination, weather, product, quantity, season, transport_mode):
    # Create a DataFrame with the user inputs
    user_data = pd.DataFrame({
        'Origin': [origin],
        'Destination': [destination],
        'Transport Mode': [transport_mode],
        'Weather Conditions': [weather],
        'Season': [season],
        'Product Type': [product],
        'Quantity': [quantity]
    })
    
    # Encode the user data using one-hot encoding
    user_data_encoded = pd.get_dummies(user_data, columns=categorical_columns, drop_first=True)
    missing_cols = set(X.columns) - set(user_data_encoded.columns)
    
    # Add the missing columns with default value 0
    for col in missing_cols:
        user_data_encoded[col] = 0
    
    # Reorder the columns to match the original order
    user_data_encoded = user_data_encoded[X.columns]
    
    # Make the prediction using the trained model
    cost_prediction = loaded_rf_model.predict(user_data_encoded)
    return cost_prediction[0]

# Example usage:
origin_input = "City B"
destination_input = "City E"
weather_input = "Rainy"
product_input = "rice"
quantity_input = 100
season_input = "winter"
transport_mode_input = "Air"

predicted_cost = predict_delivery_cost(origin_input, destination_input, weather_input, product_input, quantity_input, season_input, transport_mode_input)
print(f"Predicted cost of delivery: ${predicted_cost:.2f}")
