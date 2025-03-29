import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import openrouteservice
import folium
import requests
from streamlit_folium import st_folium
import pandas as pd
import random
from datetime import datetime

API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"
client = openrouteservice.Client(key=API_KEY)

now = datetime.now()

# Extract details
current_date = now.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
current_day = now.strftime("%A")  # Full day name (e.g., Monday)
current_hour =datetime.now().hour  


# Session state initialization
if "start" not in st.session_state:
    st.session_state.start = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "routes" not in st.session_state:
    st.session_state.routes = None
if "congestion_level" not in st.session_state:
    st.session_state.congestion_level = None
if "selected_route" not in st.session_state:
    st.session_state.selected_route = 0
if "road_data" not in st.session_state:
    st.session_state.road_data = None

travel_mode = st.selectbox("Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"

# Map initialization
m = folium.Map(location=[17.38, 78.47], zoom_start=12)

# Add markers for start and destination
if st.session_state.start:
    folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.destination:
    folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

clicked_location = st_folium(m, width=700, height=500)

if clicked_location and clicked_location["last_clicked"]:
    lat, lon = clicked_location["last_clicked"]["lat"], clicked_location["last_clicked"]["lng"]
    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.write("✅ Start location selected!")
    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.write("✅ Destination location selected!")

# Reset locations
if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.route = None
    st.session_state.road_data = None 
    st.rerun()
# Variables for traffic data
total_avg = 30  # Default avg speed
total_traffic_density = 5  # Default traffic density

# Find route when both start and destination are selected
if st.session_state.start and st.session_state.destination:
    # route = get_route(st.session_state.start, st.session_state.destination)
    # # print(route)
    st.session_state.routes = client.directions(
                coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
                profile="cycling-regular",
                # alternative_routes={"target_count": st.session_state.alternative},
                format="geojson",
            )
    print(st.session_state.routes)
    route_data = st.session_state.routes
    if "features" in route_data and route_data["features"]:
        route = route_data["features"][0]["geometry"]["coordinates"]

    
    m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)
    folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(lat, lon) for lon, lat in route], color='blue', weight=5).add_to(m)
    st_folium(m, width=700, height=500)

    if "features" in route_data and route_data["features"]:
        route = route_data["features"][0]["properties"]
    road_names = []
    for step in route["segments"][0]["steps"]:
        road_name = step.get("name", "").strip()
        if road_name and road_name != "-":
            road_names.append(road_name)
    
    unique_road_names = list(set(road_names))

    if st.session_state.road_data is None:
            road_data = []
            for road in unique_road_names:
                road_data.append({
                    "road_name": road,
                    "current_speed": random.randint(20, 60),  # Random speed (20-60 km/h)
                    "traffic_density": round(random.uniform(0.1, 1.0), 2)  # Random density (0.1-1.0)
                })
            st.session_state.road_data = road_data
    df = pd.DataFrame(st.session_state.road_data)
    st.dataframe(df.head())
    if not df.empty:
        total_avg = df['current_speed'].mean()
        total_traffic_density = df['traffic_density'].sum()

    st.write(f"Average speed: {total_avg} km/h")
    st.write(f"Total traffic density: {total_traffic_density * 10}")

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

# User input fields
road_type = st.selectbox("Road Type", ["Highway", "Main Road", "Street"])
weather = st.selectbox("Weather Condition", ["Cloudy", "Foggy", "Rainy", "Sunny"])
public_transport = st.selectbox("Public Transport Level", ["Low", "Medium", "High"])
day_of_week = st.selectbox("Day of the Week", [current_day,"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
traffic_density = st.slider("Traffic Density", 0, 100, int(total_traffic_density * 10))
avg_speed = st.slider("Average Speed (km/h)", 0, 120, int(total_avg))
temperature = st.slider("Temperature (°C)", -10, 50, 25)
accidents = st.slider("Accidents Reported", 0, 20, 2)
road_closure = st.selectbox("Road Closure", ["Yes", "No"])
hour = st.slider("Hour of the Day", 0, 23, current_hour)
is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
rush_hour = 1 if hour in [8, 9, 17, 18] else 0
urban_rural = st.selectbox("Urban or Rural", ["Urban", "Rural"])
high_risk_zone = st.selectbox("High Risk Zone", ["Yes", "No"])
high_risk_zone = 1 if high_risk_zone == "Yes" else 0

input_df = pd.DataFrame({
    "Road_Type": [road_type],
    "Weather_Condition": [weather],
    "Public_Transport": [public_transport],
    "Day_of_Week": [day_of_week],
    "Traffic_Density": [traffic_density],
    "Avg_Speed": [avg_speed],
    "Temperature": [temperature],
    "Accidents_Reported": [accidents],
    "Road_Closure": [road_closure],
    "hour": [hour],
    "is_weekend": [is_weekend],
    "rush_hour": [rush_hour],
    "urban_rural": [urban_rural],
    "high_risk_zone": [high_risk_zone]
})

st.dataframe(input_df)

# Prediction function
def predict_traffic(input_data):
    input_array = input_data.values.reshape(1, input_data.shape[0], 28)
    prediction = model.predict(input_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return le_congestion.inverse_transform([predicted_label])[0]


if st.button("Predict Congestion Level"):
    processed_input = preprocess_input(input_df)
    congestion_level = predict_traffic(processed_input)
    st.session_state.congestion_level = congestion_level
    st.success(f"🚦 Predicted Congestion Level: **{congestion_level}**")
    st.session_state.alternative = 3 if congestion_level == "High" or congestion_level == "Moderate" else 1


if st.session_state.congestion_level:
    try:
        # If alternative routes are needed, include the parameter
        if st.session_state.alternative > 1:
            st.session_state.routes = client.directions(
                coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
                profile=profile,
                alternative_routes={"target_count": st.session_state.alternative},
                format="geojson",
            )
    except Exception as e:
        st.error(f"Error fetching routes: {e}")

if st.session_state.routes:
    routes = st.session_state.routes
    num_routes = len(routes["features"])
    if st.session_state.congestion_level == "Low":
        st.success("✅ This route is clear, you can go!")
        st.write(f"### 🚦 Found {num_routes} routes for {travel_mode}")
    elif st.session_state.congestion_level == "Moderate":
        st.warning("⚠️ This route has moderate congestion, consider an alternative!")
        st.write(f"### 🚦 Found {num_routes} routes for {travel_mode}")
    else:
        st.warning("⚠️ This route has High congestion, consider an alternative!")
        st.write(f"### 🚦 Found {num_routes} alternative routes for {travel_mode}")
    # Display each route in a separate map

    for i, feature in enumerate(routes["features"]):
        distance_km = round(feature["properties"]["segments"][0]["distance"] / 1000, 2)
        duration_minutes = round(feature["properties"]["segments"][0]["duration"] / 60, 2)

        # Set route color based on congestion level
        if i == 0:  # Current route
            if st.session_state.congestion_level == "Low":
                route_color = "green"
            elif st.session_state.congestion_level == "Moderate":
                route_color = "orange"
            else:
                route_color = "red"
        else:  # Alternative routes
            route_color = "green"

        # Create a new map for each route
        m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, 
                                 (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)

        # Add markers for start and destination
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

        # Add the route to the map
        folium.PolyLine([(lat, lon) for lon, lat in feature["geometry"]["coordinates"]], color=route_color, weight=5).add_to(m)

        # Display route details
        st.write(f"### 🗺️ Route {i+1}: {distance_km} km, {duration_minutes} minutes")
        st.write(f"**Congestion Level:** {st.session_state.congestion_level if i == 0 else 'Low'}")

        # Display the map
        st_folium(m, width=700, height=500)
