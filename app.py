import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from datetime import datetime, timedelta
import plotly.express as px
import openrouteservice
import folium
from streamlit_folium import st_folium
import requests
import pickle


# 🚀 OpenRouteService API Key
API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"
client = openrouteservice.Client(key=API_KEY)

# Hyderabad Bounds
HYDERABAD_BOUNDS = [[17.2, 78.2], [17.6, 78.6]]

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Define AttentionLayer and load model (unchanged)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

class_weights = load_pickle("Model/class_weights_30min_with_features.pkl")
class_weights_tensor = tf.constant([class_weights[0], class_weights[1], class_weights[2]], dtype=tf.float32)

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    weights = tf.gather(class_weights_tensor, y_true)
    weighted_loss = loss * weights
    return weighted_loss

@st.cache_resource
def load_trained_model():
    return load_model('Model/lstm_traffic_model_30min_with_features.h5', custom_objects={
        'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy,
        'AttentionLayer': AttentionLayer
    })


def get_weather():
    # Get weather data from API
    weather_api_key = '966913ecfe3446eaa89104955252502'
    location = 'Hyderabad'
    weather_url = f'http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={location}'
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()
    weather_condition = weather_data['current']['condition']['text']
    weather = ['Sunny', 'Rainy', 'Foggy']
    if weather_condition not in weather :
        return 'Sunny'
    return weather_condition



model = load_trained_model()
st.write("Model loaded successfully")

# Load scalers (unchanged)
scaler_traffic_density = joblib.load('Model/scaler_traffic_density_30min.pkl')
scaler_avg_speed = joblib.load('Model/scaler_avg_speed_30min.pkl')
scaler_hour_of_day = joblib.load('Model/scaler_hour_of_day_30min.pkl')
scaler_day_of_year = joblib.load('Model/scaler_day_of_year_30min.pkl')

features = ['traffic_density', 'avg_speed', 'is_rush_hour', 'is_weekend', 'accident_flag',
            'hour_of_day', 'day_of_year',
            'weather_Foggy', 'weather_Rainy', 'weather_Sunny',
            'weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
            'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday',
            'month_April', 'month_August', 'month_December', 'month_February', 'month_January',
            'month_July', 'month_June', 'month_March', 'month_May', 'month_November',
            'month_October', 'month_September']


if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []

# Preprocess and get_real_time_data (unchanged)
def preprocess_timestep(data_row):
    df = pd.DataFrame([data_row])
    df[['traffic_density']] = scaler_traffic_density.transform(df[['traffic_density']])
    df[['avg_speed']] = scaler_avg_speed.transform(df[['avg_speed']])
    df[['hour_of_day']] = scaler_hour_of_day.transform(df[['hour_of_day']])
    df[['day_of_year']] = scaler_day_of_year.transform(df[['day_of_year']])

    weather_encoded = pd.DataFrame(0, index=df.index, columns=['weather_Foggy', 'weather_Rainy', 'weather_Sunny'])
    weather_encoded['weather_' + df['weather'].iloc[0]] = 1

    weekday_encoded = pd.DataFrame(0, index=df.index, columns=['weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday'])
    weekday_encoded['weekday_' + df['weekday'].iloc[0]] = 1

    month_encoded = pd.DataFrame(0, index=df.index, columns=['month_April', 'month_August', 'month_December', 'month_February', 'month_January', 'month_July', 'month_June', 'month_March', 'month_May', 'month_November', 'month_October', 'month_September'])
    month_encoded['month_' + df['month'].iloc[0]] = 1

    df = pd.concat([df, weather_encoded, weekday_encoded, month_encoded], axis=1)
    df = df[features]
    return df.to_numpy().astype(np.float32)

def get_real_time_data(timestamp):
    data_row = {
        'timestamp': timestamp,
        'traffic_density': np.random.uniform(0.1, 1.0),
        'avg_speed': np.random.uniform(5, 60),
        'is_rush_hour': 1 if (8 <= timestamp.hour <= 11 or 17 <= timestamp.hour <= 20) else 0,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'accident_flag': 1 if np.random.random() < 0.1 else 0,
        'weather': get_weather(),#np.random.choice(['Sunny', 'Rainy', 'Foggy']),
        'weekday': timestamp.strftime('%A'),
        'month': timestamp.strftime('%B'),
        'hour_of_day': timestamp.hour,
        'day_of_year': timestamp.timetuple().tm_yday
    }
    return data_row
# Function to search locations in Hyderabad
def search_location(query):
    try:
        result = client.pelias_autocomplete(
            text=query,
            boundary_rect=HYDERABAD_BOUNDS[0] + HYDERABAD_BOUNDS[1],  # [min_lat, min_lon, max_lat, max_lon]
            layers=['locality', 'neighbourhood', 'venue'],
            focus_point=[17.38, 78.47]  # Center of Hyderabad
        )
        return result['features'] if result['features'] else []
    except Exception as e:
        st.error(f"⚠️ Error searching location: {e}")
        return []



# Streamlit UI
st.title("Traffic Congestion Prediction")
st.write("Automatically collects 6 timesteps and predicts congestion for the next 2 hours.")

# 🎯 Initialize Session State Variables
if "start" not in st.session_state:
    st.session_state.start = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "routes" not in st.session_state:
    st.session_state.routes = None
if "alternative_routes" not in st.session_state:
    st.session_state.alternative_route = None
if "congestion_levels" not in st.session_state:
    st.session_state.congestion_levels = None
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []
if "selected_route" not in st.session_state:
    st.session_state.selected_route = 0
if "road_data" not in st.session_state:
    st.session_state.road_data = None

if "input_data_df" not in st.session_state:
    st.session_state.input_data_df = pd.DataFrame()

# Initialize session state for graph data
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# 🎨 Congestion Color Mapping (Global Scope)
congestion_map = {"Low": "green", "Moderate": "orange", "High": "red"}
# 🎯 Travel Mode Selection
travel_mode = st.selectbox("🚗 Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"


# 📍 Map Initialization (Restricted to Hyderabad)
m = folium.Map(
    location=[17.38, 78.47], 
    zoom_start=12,
    min_zoom=11,  # Prevent zooming out too far
    max_bounds=HYDERABAD_BOUNDS,  # Restrict panning
    max_bounds_violation='bounce'  # Bounce back if out of bounds
)

icon_type = "car" if travel_mode == "Car 🚗" else "bicycle"

# 📍 Add Markers for Start & Destination
if st.session_state.start:
    folium.Marker(st.session_state.start, popup="Start",icon=folium.Icon(color="blue", icon=icon_type, prefix="fa")).add_to(m)
if st.session_state.destination:
    folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red", icon=icon_type, prefix="fa")).add_to(m)


# 🖱️ Capture Clicked Location
clicked_location = st_folium(m, width=700, height=500)

if clicked_location and clicked_location.get("last_clicked"):
    lat, lon = clicked_location["last_clicked"]["lat"], clicked_location["last_clicked"]["lng"]
    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.write("✅ Start location selected!")
    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.write("✅ Destination location selected!")

if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.session_state.road_data = None 
    st.session_state.alternative_route = None
    st.rerun()
# 🛣️ Route Calculation
if st.session_state.start and st.session_state.destination and st.session_state.routes is None:
    try:
        st.session_state.routes = client.directions(
            coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
            profile=profile,
            format="geojson",
        )
    except Exception as e:
        st.error(f"⚠️ Failed to retrieve routes: {e}")

# Initial Map (Before Prediction)
route_data = st.session_state.routes
if route_data and "features" in route_data and st.session_state.prediction_df is None:
    route = route_data["features"][0]["geometry"]["coordinates"]
    m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, 
                            (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)
    #icon_type = "car" if travel_mode == "Car 🚗" else "bicycle"
    folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue", icon=icon_type, prefix="fa")).add_to(m)
    folium.Marker(st.session_state.destination, popup="Destination",  icon=folium.Icon(color="red", icon=icon_type, prefix="fa")).add_to(m)
    folium.PolyLine([(lat, lon) for lon, lat in route], color='blue', weight=5).add_to(m)
    st_folium(m, width=700, height=500)
if st.session_state.start and st.session_state.destination:
    if st.button("Collect 6 Timesteps and Predict"):
        st.session_state.data_buffer = []
        st.session_state.timestamps = []
        start_time = datetime.now() - timedelta(minutes=150)
        
        with st.spinner("Collecting 6 realistic timesteps..."):
            collected_data = [] 
            for i in range(6):
                timestamp = start_time + timedelta(minutes=30 * i)
                data_row = get_real_time_data(timestamp)
                processed_row = preprocess_timestep(data_row)
                st.session_state.data_buffer.append(processed_row)
                st.session_state.timestamps.append(timestamp)
                data_row["timestamp"] = timestamp
                collected_data.append(data_row)
                #st.write(f"Collected timestep {i+1} at {timestamp}:")
                #st.write(data_row)
        
        st.session_state.input_data_df = pd.DataFrame(collected_data)
        

        # Predict with probabilities
        X_new = np.array(st.session_state.data_buffer).reshape(1, 6, 29)
        y_new_pred = model.predict(X_new)  # Raw probabilities
        y_new_pred_classes = np.argmax(y_new_pred, axis=-1)

        class_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        predicted_labels = [class_labels[pred] for pred in y_new_pred_classes[0]]
        congestion_levels = predicted_labels
        st.session_state.congestion_levels = congestion_levels
        

        future_timestamps = pd.date_range(start=st.session_state.timestamps[-1] + timedelta(minutes=30), periods=4, freq='30min')
        prediction_df = pd.DataFrame({
            'Timestamp': future_timestamps,
            'Predicted_Congestion_Level': y_new_pred_classes[0],
            'Predicted_Congestion_Label': predicted_labels,
            'Prob_Low': y_new_pred[0, :, 0],
            'Prob_Moderate': y_new_pred[0, :, 1],
            'Prob_High': y_new_pred[0, :, 2]
        })
        st.session_state.graph_data = prediction_df.copy()
        
# Display graph only if there is valid data
if st.session_state.graph_data is not None and not st.session_state.graph_data.empty:
    # st.dataframe(r)
    # Display collected input data
    st.subheader("Collected Input Data")
    st.dataframe(st.session_state.input_data_df)
    st.title("Traffic Density Visualization")
    st.write("### Traffic Density Over Time")
    fig = px.line(st.session_state.input_data_df, x='timestamp', y='traffic_density')
    st.plotly_chart(fig)

    st.write("### 🚗 Average Speed Over Time")
    fig_speed = px.line(st.session_state.input_data_df, x="timestamp", y="avg_speed", markers=True, title="Average Speed Trend")
    st.plotly_chart(fig_speed)


    st.title("Predicted Congestion Levels Over Time")
    st.write("\nPrediction Summary (with Probabilities):")
    st.dataframe(st.session_state.graph_data)
    fig = px.line(
        st.session_state.graph_data, 
        x='Timestamp', 
        y='Predicted_Congestion_Level', 
        title='Predicted Congestion Levels Over Time',
        labels={'Predicted_Congestion_Level': 'Congestion Level'},
        text=st.session_state.graph_data['Predicted_Congestion_Label']
    )
    fig.update_traces(mode='lines+markers+text', textposition='top center')
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=['Low', 'Moderate', 'High'])
    st.plotly_chart(fig)
   


if st.session_state.data_buffer:
    # st.write(f"Current buffer size: {len(st.session_state.data_buffer)} timesteps")
    buffer_df = pd.DataFrame({
        'Timestamp': st.session_state.timestamps,
        'Data': [f"Timestep {i+1}" for i in range(len(st.session_state.timestamps))]
    })
    # st.dataframe(buffer_df)

if st.session_state.congestion_levels is not None :
    max_congestion = max(map(lambda x: {"Low": 0, "Moderate": 1, "High": 2}[x], st.session_state.congestion_levels))
    if max_congestion is not None:
        if max_congestion == 0:
            st.success("✅ This route is clear for the next 2 hours!")
        elif max_congestion == 1:
            st.warning("⚠️ Moderate congestion expected in the next 2 hours!")
        
        else:
            st.warning("⚠️ High congestion expected in the next 2 hours, consider an alternative!")
else:
    max_congestion =0

if st.session_state.congestion_levels:
    route_data = st.session_state.routes
    alt_route_color =congestion_map[["Low", "Moderate", "High"][max_congestion]]
    if route_data and "features" in route_data and st.session_state.prediction_df is None:
        route = route_data["features"][0]["geometry"]["coordinates"]
        distance_km = round(route_data["features"][0]["properties"]["segments"][0]["distance"] / 1000,2)
        duration_minutes = round(route_data["features"][0]["properties"]["segments"][0]["distance"] / 60, 2)
        m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, 
                                (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue", icon=icon_type, prefix="fa")).add_to(m)
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red", icon=icon_type, prefix="fa")).add_to(m)
        popup_text_inital = f"Route <br>Distance: {distance_km} km<br>Time: {duration_minutes} min"
        folium.PolyLine([(lat, lon) for lon, lat in route], color=alt_route_color, weight=5, popup=folium.Popup(popup_text_inital, max_width=200)).add_to(m)
        st_folium(m, width=700, height=500)


if st.session_state.congestion_levels and max_congestion > 0:
    st.subheader("Alternative Routes:")
    alternative_count = 2 if max_congestion > 1 else 2
    #st.write(alternative_count)
    st.write(f"Generating alternative routes...")
    st.session_state.alternative = alternative_count
    profile1 = "cycling-regular" if travel_mode == "Car 🚗" else "driving-car"
    try:
        if st.session_state.alternative > 1:
            st.session_state.alternative_route = client.directions(
                coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
                profile=profile1 ,
                alternative_routes={"target_count": alternative_count},
                format="geojson",
            )
        routes = st.session_state.alternative_route
        #st.write(f"### 🚦 Found {len(routes['features'])} routes for {travel_mode}")
        st.subheader(f"**Single Map** ")
        durations = []
        routes_alternative = []
        for i, feature in enumerate(routes["features"]):
            route = feature["geometry"]["coordinates"]
            distance_km = round(feature["properties"]["segments"][0]["distance"] / 1000, 2)
            duration_minutes = round(feature["properties"]["segments"][0]["duration"] / 60, 2)
            durations.append(duration_minutes)
            routes_alternative.append((route, distance_km, duration_minutes))

        route_intal = route_data["features"][0]["geometry"]["coordinates"]
        level = "High " if max_congestion == 2 else "Moderate"
        popup__text_inital = f" predicted congestion : {level}"
        folium.PolyLine([(lat, lon) for lon, lat in route_intal], color='red', weight=5, popup=folium.Popup(popup__text_inital, max_width=200)).add_to(m)
        # Add PolyLine for each route
        sorted_routes = [x for _, x in sorted(zip(durations, routes_alternative))]
        colors = ["green", "blue", "purple"]
        for i, (route, distance_km, duration_minutes) in enumerate(sorted_routes):
            popup_text = f"Route {i+1}<br>Distance: {distance_km} km<br>Time: {duration_minutes} min"
            color = colors[i % len(colors)]
            folium.PolyLine([(lat, lon) for lon, lat in route], color=color, weight=5,popup=folium.Popup(popup_text, max_width=200)).add_to(m)
        # Render the single map
        st_folium(m, width=700, height=500)

        for i, feature in enumerate(routes["features"]):
            distance_km = round(feature["properties"]["segments"][0]["distance"] / 1000, 2)
            duration_minutes = round(feature["properties"]["segments"][0]["duration"] / 60, 2)
            color = colors[i % len(colors)]
            #alt_route_color = congestion_map["Low"] if i > 0 else congestion_map[["Low", "Moderate", "High"][max_congestion]]
            m = folium.Map(location=[(st.session_state.start[0] + st.session_state.destination[0]) / 2, 
                                    (st.session_state.start[1] + st.session_state.destination[1]) / 2], zoom_start=13)
            folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue", icon=icon_type, prefix="fa")).add_to(m)
            folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red", icon=icon_type, prefix="fa")).add_to(m)
            folium.PolyLine([(lat, lon) for lon, lat in feature["geometry"]["coordinates"]], 
                            color=color, weight=5).add_to(m)
            
            st.write(f"### 🗺️ Route {i+1}: {distance_km} km")
            st.write(f"**Expected Time** :  {round(duration_minutes)} minutes")
            st_folium(m, width=700, height=500)
        #st.write(f"**Congestion Level:** ")
        
    except Exception as e:

        #st.error(f"⚠️ Failed to retrieve alternative routes: {e}")
        print("Please try again later")

if st.button("Clear Buffer"):
    st.session_state.data_buffer = []
    st.session_state.timestamps = []
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.session_state.road_data = None 
    st.session_state.alternative_route = None
    st.session_state.graph_data = None
    st.session_state.congestion_levels = None
    st.session_state.input_data_df= None
    st.rerun()
    st.write("Buffer cleared.")