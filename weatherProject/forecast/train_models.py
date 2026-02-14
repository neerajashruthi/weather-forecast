import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Get absolute path to CSV
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # forecast/
BASE_DIR = os.path.dirname(APP_DIR)                  # weatherProject/
CSV_PATH = os.path.join(BASE_DIR, "../weather.csv")     # C:\Weather_App\weather.csv

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

data = pd.read_csv(CSV_PATH)
data = data.dropna().drop_duplicates()

le = LabelEncoder()
data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])

X = data[["MinTemp","MaxTemp","WindGustDir","WindGustSpeed","Humidity","Pressure","Temp"]]
y = data["RainTomorrow"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(x_train, y_train)
joblib.dump(rain_model, os.path.join(APP_DIR, "rain_model.pkl"))
joblib.dump(le, os.path.join(APP_DIR, "label_encoder.pkl"))

def prepare_regression_data(data, feature):
    x, y = [], []
    for i in range(len(data)-1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(x).reshape(-1,1), np.array(y)

x_temp, y_temp = prepare_regression_data(data, "Temp")
temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
temp_model.fit(x_temp, y_temp)
joblib.dump(temp_model, os.path.join(APP_DIR, "temp_model.pkl"))

x_hum, y_hum = prepare_regression_data(data, "Humidity")
hum_model = RandomForestRegressor(n_estimators=100, random_state=42)
hum_model.fit(x_hum, y_hum)
joblib.dump(hum_model, os.path.join(APP_DIR, "hum_model.pkl"))

print("All models trained and saved successfully!")
