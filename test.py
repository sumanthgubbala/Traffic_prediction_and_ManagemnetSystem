import streamlit as st
import openrouteservice
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# OpenRouteService API Key (Replace with your actual key)
API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"

# Initialize OpenRouteService client
client = openrouteservice.Client(key=API_KEY)

# Predefined locations in Hyderabad
HYDERABAD_LOCATIONS = [
    "Kukatpally, Hyderabad, India",
    "Bachupally, Hyderabad, India",
    "Madhapur, Hyderabad, India",
    "Hitech City, Hyderabad, India",
    "Gachibowli, Hyderabad, India",
    "Secunderabad, Hyderabad, India",
    "Ameerpet, Hyderabad, India",
    "Begumpet, Hyderabad, India",
    "Charminar, Hyderabad, India",
    "Mehdipatnam, Hyderabad, India",
]

# Function to get latitude & longitude from OpenRouteService
def get_location_coords(location_name):
    try:
        response = client.pelias_search(text=location_name + ", Hyderabad, India")
        if response and "features" in response and len(response["features"]) > 0:
            lon, lat = response["features"][0]["geometry"]["coordinates"]
            return lat, lon  # Return correct latitude & longitude
        else:
            st.error("⚠️ Location not found. Try selecting a nearby landmark.")
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None



# Streamlit UI
st.title("🚗 Hyderabad Alternative Route Finder")
st.write("Select start and destination locations within Hyderabad.")

# Selection Mode
selection_mode = st.radio("🔍 How would you like to select locations?", ["Dropdown", "Map"])



# Initialize session state for locations and routes
# Session state initialization
if "start" not in st.session_state:
    st.session_state.start = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "routes" not in st.session_state:
    st.session_state.routes = None
if "congestion_level" not in st.session_state:
    st.session_state.congestion_level = None


# Select locations using dropdown
if selection_mode == "Dropdown":
    # Select Start Location
    start_location = st.selectbox("🔵 Select Start Location (Hyderabad only)", HYDERABAD_LOCATIONS)
    if start_location:
        st.session_state.start = get_location_coords(start_location)
        st.write(f"✅ Start location set: {start_location}")

    # Select Destination Location
    destination_location = st.selectbox("🔴 Select Destination Location (Hyderabad only)", HYDERABAD_LOCATIONS)
    if destination_location:
        st.session_state.destination = get_location_coords(destination_location)
        st.write(f"✅ Destination location set: {destination_location}")

# Select locations using map
else:
    # Map View
    st.write("🗺️ Select locations on the map below")

    m = folium.Map(location=[17.385, 78.4867], zoom_start=12)

    # Add existing markers if available
    if st.session_state.start:
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
    if st.session_state.destination:
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

    # Display the interactive map
    map_data = st_folium(m, width=700, height=500)


    # Check if the user clicked on the map
    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        
        if not st.session_state.start:
            st.session_state.start = (lat, lon)
            st.success("✅ Start location selected!")
        elif not st.session_state.destination:
            st.session_state.destination = (lat, lon)
            st.success("✅ Destination location selected!")

# Travel mode selection
travel_mode = st.selectbox("🚦 Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"

# Button to Reset Locations
if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.rerun()

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

# User input fields
road_type = st.selectbox("Road Type", ["Highway", "Main Road", "Street"])
weather = st.selectbox("Weather Condition", ["Cloudy", "Foggy", "Rainy", "Sunny"])
public_transport = st.selectbox("Public Transport Level", ["Low", "Medium", "High"])
day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
traffic_density = st.slider("Traffic Density", 0, 100, 50)
avg_speed = st.slider("Average Speed (km/h)", 0, 120, 30)
temperature = st.slider("Temperature (°C)", -10, 50, 25)
accidents = st.slider("Accidents Reported", 0, 20, 2)
road_closure = st.selectbox("Road Closure", ["Yes", "No"])
hour = st.slider("Hour of the Day", 0, 23, 14)
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


if st.button("Predict Congestion Level"):
    processed_input = preprocess_input(input_df)
    congestion_level = predict_traffic(processed_input)
    st.session_state.congestion_level = congestion_level
    st.session_state.alternative = 2 if congestion_level == "High" or congestion_level == "Moderate" else 1


if st.session_state.congestion_level:
    try:
        # If alternative routes are needed, include the parameter
        if st.session_state.alternative > 1:
            st.session_state.routes = client.directions(
            coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
            profile=profile,
            alternative_routes={"target_count": st.session_state.alternative},
            format="geojson",
            radiuses=[5000, 5000],  # Increased search radius to 1000m
        )
        else:
            # Normal routing without alternatives
            st.session_state.routes = client.directions(
                coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
                profile=profile,
                format="geojson",
                radiuses=[1000, 1000]
            )
    except Exception as e:
        st.error(f"Error fetching routes: {e}")


# # Find alternative routes
# if st.session_state.start and st.session_state.destination:
#     try:
#         st.session_state.routes = client.directions(
#             coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
#             profile=profile,
#             alternative_routes={"target_count": 2},
#             format="geojson",
#             radiuses=[3000, 3000],  # Increased search radius to 1000m
#     )
#     except Exception as e:
#         st.error(f"Error fetching routes: {e}")


# Display Routes
if st.session_state.routes:
    routes = st.session_state.routes
    st.success(f"🚦 Predicted Congestion Level: **{st.session_state.congestion_level}**")
    num_routes = len(routes["features"])
    if st.session_state.congestion_level == "Low":
        st.success("✅ This route is clear, you can go!")
        st.write(f"### 🚦 Found {num_routes}  routes for {travel_mode}")
    elif st.session_state.congestion_level == "Moderate":
        st.warning("⚠️ This route has moderate congestion, consider an alternative!")
        st.write(f"### 🚦 Found {num_routes}  routes for {travel_mode}")
    else :
        st.warning("⚠️ This route has High congestion, consider an alternative!")
        st.write(f"### 🚦 Found {num_routes} alternative routes for {travel_mode}")

    colors = ""
    if st.session_state.congestion_level == "High":
        colors = "red"
    elif st.session_state.congestion_level == "Moderate":
        colors = "orange"
    else :
        colors = "green"

    # Show each route
    for i, feature in enumerate(routes["features"]):
        distance_meters = feature["properties"]["segments"][0]["distance"]
        duration_seconds = feature["properties"]["segments"][0]["duration"]

        distance_km = round(distance_meters / 1000, 2)
        duration_minutes = round(duration_seconds / 60, 2)

        st.write(f"### 🗺️ Route {i+1}")
        st.write(f"🛣️ **Distance:** {distance_km} km")
        st.write(f"⏱️ **Estimated Time:** {duration_minutes} minutes")

        # Create Map
        center_lat = (st.session_state.start[0] + st.session_state.destination[0]) / 2
        center_lon = (st.session_state.start[1] + st.session_state.destination[1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Add Markers
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

        # Add Route
        coords = [(lat, lon) for lon, lat in feature["geometry"]["coordinates"]]
        folium.PolyLine(coords, color=colors, weight=5, opacity=0.7).add_to(m)

        # Display Map
        st_folium(m, width=700, height=500)
