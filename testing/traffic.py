import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import openrouteservice
import folium
import requests
from streamlit_folium import st_folium
from datetime import datetime

# API Keys (Replace with your actual keys)
ORS_API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"  # OpenRouteService
TOMTOM_API_KEY = "yVGxrcU8Ooo3SAmPyCLhFPLFucsbrPdo"  # Replace with your TomTom API key
WEATHER_API_KEY = "743205e155e24349b0793335253003"  # Replace with your OpenWeatherMap API key

# Initialize OpenRouteService client
client = openrouteservice.Client(key=ORS_API_KEY)

# Current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_day = now.strftime("%A")
current_hour = now.hour

# Session state initialization
if "start" not in st.session_state:
    st.session_state.start = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "routes" not in st.session_state:
    st.session_state.routes = None
if "congestion_level" not in st.session_state:
    st.session_state.congestion_level = None

# Travel mode selection
travel_mode = st.selectbox("Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"

# Map initialization
m = folium.Map(location=[17.38, 78.47], zoom_start=12)

# Add markers for start and destination
if st.session_state.start:
    folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.destination:
    folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

# Interactive map for selecting locations
clicked_location = st_folium(m, width=700, height=500)

if clicked_location and clicked_location["last_clicked"]:
    lat, lon = clicked_location["last_clicked"]["lat"], clicked_location["last_clicked"]["lng"]
    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.write("✅ Start location selected!")
    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.write("✅ Destination location selected!")

# Reset button
if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.session_state.congestion_level = None
    st.rerun()

# Load ML model and preprocessors
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/traffic_congestion_model.h5")

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_pickle("models/scaler.pkl")
le_urban_rural = load_pickle("models/le_urban_rural.pkl")
le_Road_Closure = load_pickle("models/le_Road_Closure.pkl")
le_congestion = load_pickle("models/le_congestion_level.pkl")
ohe = load_pickle("models/ohe.pkl")

# TomTom Traffic API function
def get_tomtom_traffic_data(start, destination):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{start[0]},{start[1]}",
        "destination": f"{destination[0]},{destination[1]}"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        avg_speed = data["flowSegmentData"]["currentSpeed"]  # km/h
        traffic_density = data["flowSegmentData"]["currentTravelTime"] / data["flowSegmentData"]["freeFlowTravelTime"]
        return avg_speed, traffic_density
    else:
        st.error("Failed to fetch traffic data from TomTom")
        return 30, 5  # Default values

# Weather API function (OpenWeatherMap example)
def get_weather_data(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}&aqi=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data["current"]["temp_c"]  # Extract temperature
        weather_condition = data["current"]["condition"]["text"]  # Extract weather condition
        return temperature, weather_condition
    else:
        st.error(f"Failed to fetch weather data. Error Code: {response.status_code}")
        return 25, "Sunny"  # Default values

# Preprocessing function
def preprocess_input(df_input):
    df = df_input.copy()
    df["urban_rural"] = le_urban_rural.transform(df["urban_rural"])
    df["Road_Closure"] = le_Road_Closure.transform(df["Road_Closure"])
    
    num_features = ["Temperature", "Avg_Speed", "Accidents_Reported", "Traffic_Density"]
    df[num_features] = scaler.transform(df[num_features])
    
    cat_features = ["Road_Type", "Weather_Condition", "Public_Transport", "Day_of_Week"]
    encoded_data = ohe.transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(cat_features))
    
    final_df = pd.concat([encoded_df, df.drop(columns=cat_features)], axis=1)
    return final_df

# Prediction function
def predict_traffic(input_data):
    input_array = input_data.values.reshape(1, input_data.shape[0], 28)
    prediction = model.predict(input_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return le_congestion.inverse_transform([predicted_label])[0]

# Route calculation and processing
if st.session_state.start and st.session_state.destination:
    # Fetch traffic data from TomTom
    avg_speed, traffic_density = get_tomtom_traffic_data(st.session_state.start, st.session_state.destination)
    
    # Fetch weather data for the midpoint of the route
    mid_lat = (st.session_state.start[0] + st.session_state.destination[0]) / 2
    mid_lon = (st.session_state.start[1] + st.session_state.destination[1]) / 2
    temperature, weather_condition = get_weather_data(mid_lat, mid_lon)
    
    # Fetch route from OpenRouteService
    st.session_state.routes = client.directions(
        coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
        profile=profile,
        format="geojson"
    )
    
    # Extract road type from route data (simplified assumption)
    route_data = st.session_state.routes["features"][0]["properties"]
    road_names = [step.get("name", "Street") for step in route_data["segments"][0]["steps"] if step.get("name")]
    road_type = "Highway" if "highway" in " ".join(road_names).lower() else "Street"
    
    # Prepare input data for prediction
    input_df = pd.DataFrame({
        "Road_Type": [road_type],
        "Weather_Condition": [weather_condition],
        "Public_Transport": ["Medium"],  # Assumption (could be API-driven if available)
        "Day_of_Week": [current_day],
        "Traffic_Density": [traffic_density],
        "Avg_Speed": [avg_speed],
        "Temperature": [temperature],
        "Accidents_Reported": [0],  # Placeholder (TomTom doesn’t provide this directly)
        "Road_Closure": ["No"],  # Assumption (could be API-driven)
        "hour": [current_hour],
        "is_weekend": [1 if current_day in ["Saturday", "Sunday"] else 0],
        "rush_hour": [1 if current_hour in [8, 9, 17, 18] else 0],
        "urban_rural": ["Urban"],  # Assumption (could be derived from coordinates)
        "high_risk_zone": [0]  # Assumption
    })

    # Display input data
    st.write("### Input Data from APIs:")
    st.dataframe(input_df)

    # Predict congestion
    processed_input = preprocess_input(input_df)
    congestion_level = predict_traffic(processed_input)
    st.session_state.congestion_level = congestion_level
    st.success(f"🚦 Predicted Congestion Level: **{congestion_level}**")

    # Fetch alternative routes if congestion is high or moderate
    alternative_count = 3 if congestion_level in ["High", "Moderate"] else 1
    st.session_state.routes = client.directions(
        coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
        profile=profile,
        alternative_routes={"target_count": alternative_count},
        format="geojson"
    )

# Display routes
if st.session_state.routes:
    routes = st.session_state.routes
    num_routes = len(routes["features"])
    st.write(f"### 🚦 Found {num_routes} routes for {travel_mode}")

    for i, feature in enumerate(routes["features"]):
        distance_km = round(feature["properties"]["segments"][0]["distance"] / 1000, 2)
        duration_minutes = round(feature["properties"]["segments"][0]["duration"] / 60, 2)
        
        route_color = "green" if i > 0 or st.session_state.congestion_level == "Low" else "orange" if st.session_state.congestion_level == "Moderate" else "red"
        
        m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, 
                                 (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([(lat, lon) for lon, lat in feature["geometry"]["coordinates"]], color=route_color, weight=5).add_to(m)

        st.write(f"### 🗺️ Route {i+1}: {distance_km} km, {duration_minutes} minutes")
        st.write(f"**Congestion Level:** {st.session_state.congestion_level if i == 0 else 'Low'}")
        st_folium(m, width=700, height=500)