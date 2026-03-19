from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse

import requests
import pandas as pd
import numpy as np
import os
from django.shortcuts import render
from datetime import datetime, timedelta
from django.conf import settings
import joblib
import pytz

APP_DIR = os.path.join(settings.BASE_DIR, "forecast")

rain_model = joblib.load(os.path.join(APP_DIR, "rain_model.pkl"))
temp_model = joblib.load(os.path.join(APP_DIR, "temp_model.pkl"))
hum_model = joblib.load(os.path.join(APP_DIR, "hum_model.pkl"))
le = joblib.load(os.path.join(APP_DIR, "label_encoder.pkl"))

API_KEY = "f90e26f8696a46f8bb6c699ff056ad91"
BASE_URL = "https://api.openweathermap.org/data/2.5"


def get_current_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        print("API ERROR:", data)
        return None
    return{
        "city":data['name'],
        "current_temp":round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "temp_min": round(data["main"]["temp_min"]),
        "temp_max": round(data["main"]["temp_max"]),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "wind_gust_dir": data["wind"].get("deg", 0),
        "pressure": data["main"]["pressure"],
        "WindGustSpeed": data["wind"].get("speed", 0),
        "clouds": data['clouds']['all'],
        "Visibility": data['visibility'],
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])
    x = data[["MinTemp","MaxTemp","WindGustDir","WindGustSpeed","Humidity","Pressure","Temp"]]
    y = data["RainTomorrow"]
    return x,y,le

def train_rain_model(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 42)
    model = RandomForestClassifier(n_estimators=100, random_state = 42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Mean Squared Error for Rain Model: ")
    print(mean_squared_error(y_test,y_pred))
    return model

def prepare_regression_data(data, feature):
    x, y = [],[]
    for i in range(len(data)-1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    return x,y

def train_regression_model(x,y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x,y)
    return model

def predit_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))

        predictions.append(next_value[0])
    return predictions[1:]
                                

# def weather_view():
#     city = input("Enter the City Name: ")
#     current_weather = get_current_weather(city)
#     try:
#         current_weather = get_current_weather(city)
#     except ValueError as e:
#         print(e)
#         return
#     historical_data = read_historical_data("weather.csv")
#     x,y,le = prepare_data(historical_data)
#     rain_model = train_rain_model(x,y)
#     wind_deg = current_weather["wind_gust_dir"] % 360

#     compass_points = [
#     ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
#     ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
#     ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
#     ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
#     ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
#     ("NNW", 326.25, 348.75),
#     ]

#     compass_direction = next(
#     (point for point, start, end in compass_points if start <= wind_deg < end))
#     compass_direction_encoded = (
#     le.transform([compass_direction])[0]
#     if compass_direction in le.classes_
#     else -1)

#     current_data = {
#         "MinTemp":current_weather["temp_min"],
#         "MaxTemp":current_weather["temp_max"],
#         "WindGustDir":compass_direction_encoded,
#         "WindGustSpeed": current_weather["WindGustSpeed"],
#         "Humidity": current_weather["humidity"],
#         "Pressure":current_weather["pressure"],
#         "Temp":current_weather["current_temp"],
#     }
#     current_df = pd.DataFrame([current_data])
#     rain_prediction = rain_model.predict(current_df)[0]
#     x_temp,y_temp = prepare_regression_data(historical_data,"Temp")
#     x_hum,y_hum = prepare_regression_data(historical_data,"Humidity")
#     temp_model = train_regression_model(x_temp,y_temp)
#     hum_model = train_regression_model(x_hum,y_hum)
#     future_temp = predit_future(temp_model,current_weather["temp_min"])
#     future_humidity = predit_future(hum_model,current_weather["humidity"])
#     timezone = pytz.timezone('Asia/Kolkata')
#     now = datetime.now(timezone)
#     next_hour = now + timedelta(hours=1)
#     next_hour = next_hour.replace(minute = 0, second = 0, microsecond = 0)
#     future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
#     print(f"City: {city}, {current_weather['country']}")
#     print(f"Current Temperature: {current_weather['current_temp']} ")
#     print(f"Feels Like: {current_weather['feels_like']} ")
#     print(f"Minimum Temperature: {current_weather['temp_min']}℃ ")
#     print(f"Maximum Temperature: {current_weather['temp_max']} ℃") 
#     print(f"Humidity: {current_weather['humidity']}% ") 
#     print(f"Weather Prediction: {current_weather['description']}") 
#     print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")
#     print("\nFuture Temperature Predictions: ")
#     for time, temp in zip(future_times, future_temp):
#         print(f"{time}:{round(temp,1)}℃")
#     print("\nFuture Humidity Predictions: ")
#     for time, temp in zip(future_times, future_humidity):
#         print(f"{time}:{round(temp,1)}%")

# weather_view()







def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        
        current_weather = get_current_weather(city)
        if current_weather is None:
            return render(request, "weather.html", {
                "error": "City not found"
            })
        wind_deg = current_weather["wind_gust_dir"] % 360

        compass_points = [("N", 348.75, 360),("N", 0, 11.25),("NNE", 11.25, 33.75),("NE", 33.75, 56.25),("ENE", 56.25, 78.75),("E", 78.75, 101.25),
                        ("ESE", 101.25, 123.75),("SE", 123.75, 146.25),("SSE", 146.25, 168.75),("S", 168.75, 191.25),("SSW", 191.25, 213.75),
                        ("SW", 213.75, 236.25),("WSW", 236.25, 258.75),("W", 258.75, 281.25),("WNW", 281.25, 303.75),("NW", 303.75, 326.25),
                        ("NNW", 326.25, 348.75),
                        ]

        compass_direction = next(
        point for point, start, end in compass_points
        if start <= wind_deg < end
        )

        compass_direction_encoded = (
        le.transform([compass_direction])[0]
        if compass_direction in le.classes_
        else -1)

        current_data = {
            "MinTemp":current_weather["temp_min"],
            "MaxTemp":current_weather["temp_max"],
            "WindGustDir":compass_direction_encoded,
            "WindGustSpeed": current_weather["WindGustSpeed"],
            "Humidity": current_weather["humidity"],
            "Pressure":current_weather["pressure"],
            "Temp":current_weather["current_temp"],
        }
        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]
        future_temp = predit_future(temp_model,current_weather["temp_min"])
        future_humidity = predit_future(hum_model,current_weather["humidity"])
        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute = 0, second = 0, microsecond = 0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity
        
        pressure_text = ''
        wind_text=''
        day_stats_flag ='false'

        raw_description = current_weather["description"].lower()

        if "rain" in raw_description:
            weather_class = "rain"            
        elif "cloud" in raw_description:
            weather_class = "clouds"            
        elif "clear" in raw_description:
            weather_class = "clear"            
        elif "mist" in raw_description or "haze" in raw_description:
            weather_class = "mist"            
        else:
            weather_class = "clear"

            
        
        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds' : current_weather['clouds'],
            'weather_class': weather_class,
            'real_description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            

            'time': datetime.now(),
            'date': datetime.now().strftime("%B, %d, %Y"),
            'wind': current_weather['WindGustSpeed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['Visibility'],
            'time1':time1,
            'time2':time2,
            'time3':time3,
            'time4':time4,
            'time5':time5,

            'temp1':f"{round(temp1,1)}",
            'temp2':f"{round(temp2,1)}",
            'temp3':f"{round(temp3,1)}",
            'temp4':f"{round(temp4,1)}",
            'temp5':f"{round(temp5,1)}",

            'hum1': f"{round(hum1,1)}",
            'hum2': f"{round(hum2,1)}",
            'hum3': f"{round(hum3,1)}",
            'hum4': f"{round(hum4,1)}",
            'hum5': f"{round(hum5,1)}",
            'wind_text':'kilometers per hour. pressure is ',
            'pressure_text' :'mb. visibility is ',
            'visibility_text':'. Maximum temperature is ',
            'maxtemp_text':'℃. Minimum temperature is ',
            'mintemp_text':'℃.', 'day_stats_flag':day_stats_flag
        }
        return render(request, 'weather.html', context)
        return render(request, 'weather.html', {
        'weather_class': 'clouds',
        'real_description': 'Search for a city',
        'day_stats_flag': 'false'
        })
    else :return render(request, 'weather.html', {        
        'day_stats_flag': 'false'
        })
    