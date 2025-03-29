import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from datetime import datetime, timedelta
import plotly.express as px

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

class_weights = joblib.load('class_weights_30min_with_features.pkl')
class_weights_tensor = tf.constant([class_weights[0], class_weights[1], class_weights[2]], dtype=tf.float32)

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    weights = tf.gather(class_weights_tensor, y_true)
    return weighted_loss

@st.cache_resource
def load_trained_model():
    return load_model('lstm_traffic_model_30min_with_features.h5', custom_objects={
        'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy,
        'AttentionLayer': AttentionLayer
    })

model = load_trained_model()
st.write("Model loaded successfully")

# Load scalers (unchanged)
scaler_traffic_density = joblib.load('scaler_traffic_density_30min.pkl')
scaler_avg_speed = joblib.load('scaler_avg_speed_30min.pkl')
scaler_hour_of_day = joblib.load('scaler_hour_of_day_30min.pkl')
scaler_day_of_year = joblib.load('scaler_day_of_year_30min.pkl')

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
        'weather': np.random.choice(['Sunny', 'Rainy', 'Foggy']),
        'weekday': timestamp.strftime('%A'),
        'month': timestamp.strftime('%B'),
        'hour_of_day': timestamp.hour,
        'day_of_year': timestamp.timetuple().tm_yday
    }
    return data_row

# Streamlit UI
st.title("Real-Time Traffic Congestion Prediction")
st.write("Automatically collects 6 timesteps and predicts congestion for the next 2-3 hours.")

if st.button("Collect 6 Timesteps and Predict"):
    st.session_state.data_buffer = []
    st.session_state.timestamps = []
    start_time = datetime.now() - timedelta(minutes=150)
    
    with st.spinner("Collecting 6 realistic timesteps..."):
        for i in range(6):
            timestamp = start_time + timedelta(minutes=30 * i)
            data_row = get_real_time_data(timestamp)
            processed_row = preprocess_timestep(data_row)
            st.session_state.data_buffer.append(processed_row)
            st.session_state.timestamps.append(timestamp)
            st.write(f"Collected timestep {i+1} at {timestamp}:")
            st.write(data_row)

    # Predict with probabilities
    X_new = np.array(st.session_state.data_buffer).reshape(1, 6, 29)
    y_new_pred = model.predict(X_new)  # Raw probabilities
    y_new_pred_classes = np.argmax(y_new_pred, axis=-1)

    class_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
    predicted_labels = [class_labels[pred] for pred in y_new_pred_classes[0]]

    future_timestamps = pd.date_range(start=st.session_state.timestamps[-1] + timedelta(minutes=30), periods=4, freq='30min')
    prediction_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Predicted_Congestion_Level': y_new_pred_classes[0],
        'Predicted_Congestion_Label': predicted_labels,
        'Prob_Low': y_new_pred[0, :, 0],
        'Prob_Moderate': y_new_pred[0, :, 1],
        'Prob_High': y_new_pred[0, :, 2]
    })

    # Display prediction table with probabilities
    st.write("\nPrediction Summary (with Probabilities):")
    st.dataframe(prediction_df)

    # Plot graph
    fig = px.line(prediction_df, x='Timestamp', y='Predicted_Congestion_Level', 
                  title='Predicted Congestion Levels Over Time',
                  labels={'Predicted_Congestion_Level': 'Congestion Level'},
                  text=prediction_df['Predicted_Congestion_Label'])
    fig.update_traces(mode='lines+markers+text', textposition='top center')
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=['Low', 'Moderate', 'High'])
    st.plotly_chart(fig)

if st.session_state.data_buffer:
    st.write(f"Current buffer size: {len(st.session_state.data_buffer)} timesteps")
    buffer_df = pd.DataFrame({
        'Timestamp': st.session_state.timestamps,
        'Data': [f"Timestep {i+1}" for i in range(len(st.session_state.timestamps))]
    })
    st.dataframe(buffer_df)

if st.button("Clear Buffer"):
    st.session_state.data_buffer = []
    st.session_state.timestamps = []
    st.write("Buffer cleared.")