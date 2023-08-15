import random
import pandas as pd
import io

# Sample small dataset (5 rows) in CSV format
small_dataset = '''Date,Origin,Destination,Transport Mode,Distance (km),Weather Conditions,Season,Product Type,Quantity,Lead Time (hours),Cost ($),Demand,Supplier,Carrier,Delivery Status
2023-07-15,City A,City D,Truck,400,Sunny,Summer,rice,100,48,2000,High,Supplier X,Carrier Y,On Time
2023-07-16,City B,City E,Train,800,Rainy,Summer,maize,500,72,3500,Medium,Supplier Z,Carrier X,Delayed
2023-07-17,City C,City F,Air,1200,Cloudy,Summer,corn,300,24,1500,Low,Supplier Y,Carrier Z,On Time
2023-07-18,City D,City G,Ship,2000,Sunny,Summer,wheat,800,96,5000,High,Supplier A,Carrier B,On Time
2023-07-19,City E,City H,Truck,600,Sunny,Summer,potatos,200,36,3000,High,Supplier C,Carrier D,Delayed'''

# Convert the small dataset to a pandas DataFrame
small_df = pd.read_csv(io.StringIO(small_dataset))

# Number of rows needed in the larger dataset
target_rows = 10000

# Create a larger dataset through random sampling with replacement
larger_data = [random.choice(small_df.values.tolist()) for _ in range(target_rows)]
larger_df = pd.DataFrame(larger_data, columns=small_df.columns)

# Save the larger dataset to a CSV file
larger_df.to_csv('larger_dataset.csv', index=False)

# Print the first few rows of the larger dataset
print(larger_df.head())
