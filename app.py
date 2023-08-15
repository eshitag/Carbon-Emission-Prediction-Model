import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
categorical_columns = ['Origin', 'Destination', 'Transport Mode', 'Weather Conditions', 'Season', 'Product Type']

# Load the data_encoded DataFrame from the larger_dataset.csv file
data_encoded = pd.read_csv("larger_dataset.csv")
data_encoded = pd.get_dummies(data_encoded, columns=categorical_columns, drop_first=True)

# Load the trained model from the pickle file
#with open('Logistic_rf_model.pkl', 'rb') as file:
 #   rf_model = pickle.load(file)


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

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_food_model():
    with open('best_model.pkl', 'rb') as f:
        food_model = pickle.load(f)
    return food_model

@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/yield')
def indexyield():
    return render_template('/yield.html')

@app.route('/emission')
def indexemission():
    return render_template('/emission.html')

@app.route('/logistics')
def indexelogistics():
    return render_template('/logistics.html')


@app.route('/predictFood', methods=['GET', 'POST'])
def predictFood():
    if request.method == 'POST':
        pesticides_tonnes = request.form['pesticides_tonnes']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        avg_temp = request.form['avg_temp']
        food_model = load_food_model()

        # Load new data for prediction
        new_data = pd.DataFrame({
            'pesticides_tonnes': [pesticides_tonnes],
            'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
            'avg_temp': [avg_temp]
        })

        # Make predictions using the loaded model
        predictions = food_model.predict(new_data)

        # Print the predictions
        result = "Predicted Yield in  hg/ha_yield (The term hg/ha_yield represents the yield of a particular crop or agricultural product measured in hectograms per hectare. It is a common unit of measurement used in agriculture to quantify the productivity or output of a specific crop per unit area.) <H2>"+ str(predictions)
        return result
    

@app.route('/predictLogistics', methods=['POST'])
def predictLogistics():
    origin_input = request.form['origin']
    destination_input = request.form['destination']
    weather_input = request.form['weather']
    product_input = request.form['product']
    quantity_input = float(request.form['quantity'])
    season_input = request.form['season']
    transport_mode_input = request.form['transport_mode']

    # Call the predict_delivery_cost function
    predicted_cost = predict_delivery_cost(data_encoded, origin_input, destination_input, weather_input, product_input, quantity_input, season_input, transport_mode_input)

    return f"Predicted cost of delivery: ${predicted_cost:.2f}"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        year = request.form['year']
        country = request.form['country']
        emissions_type = request.form['emissions_type']
        # Load the machine learning model
        rf_model = load_model()
                    
        input_values = {
        "Area Code (M49)": 227,
        "Element Code": 7225,
        "Item Code": 6818,
        "Year Code": 5,
        "Year": year,
        "Source Code": 0,
        "Domain_Emissions totals": 1,
        "Area_Afghanistan": 0,
        "Area_Albania": 0,
        "Area_Algeria": 0,
        "Area_American Samoa": 0,
        "Area_Andorra": 0,
        "Area_Angola": 0,
        "Area_Anguilla": 0,
        "Area_Antigua and Barbuda": 0,
        "Area_Argentina": 0,
        "Area_Armenia": 0,
        "Area_Aruba": 0,
        "Area_Australia": 0,
        "Area_Austria": 0,
        "Area_Azerbaijan": 0,
        "Area_Bahamas": 0,
        "Area_Bahrain": 0,
        "Area_Bangladesh": 0,
        "Area_Barbados": 0,
        "Area_Belarus": 0,
        "Area_Belgium": 0,
        "Area_Belize": 0,
        "Area_Benin": 0,
        "Area_Bermuda": 0,
        "Area_Bhutan": 0,
        "Area_Bolivia (Plurinational State of)": 0,
        "Area_Bosnia and Herzegovina": 0,
        "Area_Botswana": 0,
        "Area_Brazil": 0,
        "Area_British Virgin Islands": 0,
        "Area_Brunei Darussalam": 0,
        "Area_Bulgaria": 0,
        "Area_Burkina Faso": 0,
        "Area_Burundi": 0,
        "Area_Cabo Verde": 0,
        "Area_Cambodia": 0,
        "Area_Cameroon": 0,
        "Area_Canada": 0,
        "Area_Cayman Islands": 0,
        "Area_Central African Republic": 0,
        "Area_Chad": 0,
        "Area_Channel Islands": 0,
        "Area_Chile": 0,
        "Area_China": 1,
        "Area_China, Hong Kong SAR": 0,
        "Area_China, Macao SAR": 0,
        "Area_China, Taiwan Province of": 0,
        "Area_China, mainland": 0,
        "Area_Colombia": 0,
        "Area_Comoros": 0,
        "Area_Congo": 0,
        "Area_Cook Islands": 0,
        "Area_Costa Rica": 0,
        "Area_Croatia": 0,
        "Area_Cuba": 0,
        "Area_Cyprus": 0,
        "Area_Czechia": 0,
        "Area_CÃ´te d'Ivoire": 0,
        "Area_Democratic People's Republic of Korea": 0,
        "Area_Democratic Republic of the Congo": 0,
        "Area_Denmark": 0,
        "Area_Djibouti": 0,
        "Area_Dominica": 0,
        "Area_Dominican Republic": 0,
        "Area_Ecuador": 0,
        "Area_Egypt": 0,
        "Area_El Salvador": 0,
        "Area_Equatorial Guinea": 0,
        "Area_Eritrea": 0,
        "Area_Estonia": 0,
        "Area_Eswatini": 0,
        "Area_Ethiopia": 0,
        "Area_Falkland Islands (Malvinas)": 0,
        "Area_Faroe Islands": 0,
        "Area_Fiji": 0,
        "Area_Finland": 0,
        "Area_France": 0,
        "Area_French Guiana": 0,
        "Area_French Polynesia": 0,
        "Area_Gabon": 0,
        "Area_Gambia": 0,
        "Area_Georgia": 0,
        "Area_Germany": 0,
        "Area_Ghana": 0,
        "Area_Gibraltar": 0,
        "Area_Greece": 0,
        "Area_Greenland": 0,
        "Area_Grenada": 0,
        "Area_Guadeloupe": 0,
        "Area_Guam": 0,
        "Area_Guatemala": 0,
        "Area_Guinea": 0,
        "Area_Guinea-Bissau": 0,
        "Area_Guyana": 0,
        "Area_Haiti": 0,
        "Area_Holy See": 0,
        "Area_Honduras": 0,
        "Area_Hungary": 0,
        "Area_Iceland": 0,
        "Area_India": 0,
        "Area_Indonesia": 0,
        "Area_Iran (Islamic Republic of)": 0,
        "Area_Iraq": 0,
        "Area_Ireland": 0,
        "Area_Isle of Man": 0,
        "Area_Israel": 0,
        "Area_Italy": 0,
        "Area_Jamaica": 0,
        "Area_Japan": 0,
        "Area_Jordan": 0,
        "Area_Kazakhstan": 0,
        "Area_Kenya": 0,
        "Area_Kiribati": 0,
        "Area_Kuwait": 0,
        "Area_Kyrgyzstan": 0,
        "Area_Lao People's Democratic Republic": 0,
        "Area_Latvia": 0,
        "Area_Lebanon": 0,
        "Area_Lesotho": 0,
        "Area_Liberia": 0,
        "Area_Libya": 0,
        "Area_Liechtenstein": 0,
        "Area_Lithuania": 0,
        "Area_Luxembourg": 0,
        "Area_Madagascar": 0,
        "Area_Malawi": 0,
        "Area_Malaysia": 0,
        "Area_Maldives": 0,
        "Area_Mali": 0,
        "Area_Malta": 0,
        "Area_Marshall Islands": 0,
        "Area_Martinique": 0,
        "Area_Mauritania": 0,
        "Area_Mauritius": 0,
        "Area_Mayotte": 0,
        "Area_Mexico": 0,
        "Area_Micronesia (Federated States of)": 0,
        "Area_Monaco": 0,
        "Area_Mongolia": 0,
        "Area_Montenegro": 0,
        "Area_Montserrat": 0,
        "Area_Morocco": 0,
        "Area_Mozambique": 0,
        "Area_Myanmar": 0,
        "Area_Namibia": 0,
        "Area_Nauru": 0,
        "Area_Nepal": 0,
        "Area_Netherlands (Kingdom of the)": 0,
        "Area_Netherlands Antilles (former)": 0,
        "Area_New Caledonia": 0,
        "Area_New Zealand": 0,
        "Area_Nicaragua": 0,
        "Area_Niger": 0,
        "Area_Nigeria": 0,
        "Area_Niue": 0,
        "Area_North Macedonia": 0,
        "Area_Northern Mariana Islands": 0,
        "Area_Norway": 0,
        "Area_Oman": 0,
        "Area_Pakistan": 0,
        "Area_Palau": 0,
        "Area_Palestine": 0,
        "Area_Panama": 0,
        "Area_Papua New Guinea": 0,
        "Area_Paraguay": 0,
        "Area_Peru": 0,
        "Area_Philippines": 0,
        "Area_Poland": 0,
        "Area_Portugal": 0,
        "Area_Puerto Rico": 0,
        "Area_Qatar": 0,
        "Area_Republic of Korea": 0,
        "Area_Republic of Moldova": 0,
        "Area_Romania": 0,
        "Area_Russian Federation": 0,
        "Area_Rwanda": 0,
        "Area_RÃ©union": 0,
        "Area_Saint Helena, Ascension and Tristan da Cunha": 0,
        "Area_Saint Kitts and Nevis": 0,
        "Area_Saint Lucia": 0,
        "Area_Saint Pierre and Miquelon": 0,
        "Area_Saint Vincent and the Grenadines": 0,
        "Area_Samoa": 0,
        "Area_San Marino": 0,
        "Area_Sao Tome and Principe": 0,
        "Area_Saudi Arabia": 0,
        "Area_Senegal": 0,
        "Area_Serbia": 0,
        "Area_Seychelles": 0,
        "Area_Sierra Leone": 0,
        "Area_Singapore": 0,
        "Area_Slovakia": 0,
        "Area_Slovenia": 0,
        "Area_Solomon Islands": 0,
        "Area_Somalia": 0,
        "Area_South Africa": 0,
        "Area_South Sudan": 0,
        "Area_Spain": 0,
        "Area_Sri Lanka": 0,
        "Area_Sudan": 0,
        "Area_Sudan (former)": 0,
        "Area_Suriname": 0,
        "Area_Sweden": 0,
        "Area_Switzerland": 0,
        "Area_Syrian Arab Republic": 0,
        "Area_Tajikistan": 0,
        "Area_Thailand": 0,
        "Area_Timor-Leste": 0,
        "Area_Togo": 0,
        "Area_Tokelau": 0,
        "Area_Tonga": 0,
        "Area_Trinidad and Tobago": 0,
        "Area_Tunisia": 0,
        "Area_Turkmenistan": 0,
        "Area_Turks and Caicos Islands": 0,
        "Area_Tuvalu": 0,
        "Area_TÃ¼rkiye": 0,
        "Area_Uganda": 0,
        "Area_Ukraine": 0,
        "Area_United Arab Emirates": 0,
        "Area_United Kingdom of Great Britain and Northern Ireland": 0,
        "Area_United Republic of Tanzania": 0,
        "Area_United States Virgin Islands": 0,
        "Area_United States of America": 0,
        "Area_Uruguay": 0,
        "Area_Uzbekistan": 0,
        "Area_Vanuatu": 0,
        "Area_Venezuela (Bolivarian Republic of)": 0,
        "Area_Viet Nam": 0,
        "Area_Wallis and Futuna Islands": 0,
        "Area_Yemen": 0,
        "Area_Zambia": 0,
        "Area_Zimbabwe": 0,
        "Element_Emissions (CH4)": 1,
        "Element_Emissions (CO2)": 0,
        "Element_Emissions (CO2eq) (AR5)": 0,
        "Element_Emissions (CO2eq) from CH4 (AR5)": 0,
        "Element_Emissions (CO2eq) from F-gases (AR5)": 0,
        "Element_Emissions (CO2eq) from N2O (AR5)": 0,
        "Element_Emissions (N2O)": 0,
        "Item_Agrifood Systems Waste Disposal": 0,
        "Item_Food Packaging": 0,
        "Item_Food Processing": 0,
        "Item_Food Retail": 0,
        "Item_Food Transport": 0,
        "Item_Waste": 1,
        "Source_FAO TIER 1": 1,
        "Flag_E": 1
        }


        # Perform any necessary preprocessing on the input data
        # ...
        # Create a DataFrame from the input values
        input_df = pd.DataFrame(input_values, index=[0])

        # Get the predicted value using the Random Forest Regressor model
        predicted_value = rf_model.predict(input_df)
        # Make predictions using the loaded model
        #prediction = rf_model.predict([[year, country, emissions_type]])
        result = "Predicted Emissions in kilotones: "+ str(predicted_value)
        # Render a template to display the prediction result
        #return render_template('prediction.html', year=year, country=country, emissions_type=emissions_type, prediction=prediction)
        return result
        # Redirect to the '/predict' route with the data as URL parameters
       # return redirect(url_for('predict', year=year, country=country, emissions_type=emissions_type))
    else:
        return render_template('/index.html')

if __name__ == '__main__':
    app.run()
