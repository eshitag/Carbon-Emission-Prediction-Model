import pandas as pd
import pickle

categorical_columns = ['Origin', 'Destination', 'Transport Mode', 'Weather Conditions', 'Season', 'Product Type']

# Function to predict the cost of delivery for user inputs
def predict_delivery_cost(data_encoded, origin, destination, weather, product, quantity, season, transport_mode):
    # Load the trained model from the pickle file
    with open('Logistic_rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    
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
    
    # Get the missing columns in user_data_encoded compared to data_encoded
    missing_cols = set(data_encoded.columns) - set(user_data_encoded.columns)
    
    # Add the missing columns with default value 0
    for col in missing_cols:
        user_data_encoded[col] = 0
    
    # Reorder the columns to match the original order in data_encoded
    user_data_encoded = user_data_encoded[data_encoded.columns]
    
    # Drop the 'Cost ($)' column as it should not be present in the user input data for predictions
    user_data_encoded = user_data_encoded.drop(columns=['Cost ($)'])
    
    # Make the prediction using the trained model
    cost_prediction = rf_model.predict(user_data_encoded)
    return cost_prediction[0]

# Example usage:
origin_input = "City A"
destination_input = "City E"
weather_input = "Rainy"
product_input = "corn"
quantity_input = 500
season_input = "Summer"
transport_mode_input = "Train"

# Load the data_encoded DataFrame from the larger_dataset.csv file
data_encoded = pd.read_csv("larger_dataset.csv")
data_encoded = pd.get_dummies(data_encoded, columns=categorical_columns, drop_first=True)

predicted_cost = predict_delivery_cost(data_encoded, origin_input, destination_input, weather_input, product_input, quantity_input, season_input, transport_mode_input)
print(f"Predicted cost of delivery: ${predicted_cost:.2f}")
