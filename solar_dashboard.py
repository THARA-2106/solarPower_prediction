# solar_dashboard.py
import pandas as pd
import numpy as np
import io
import base64
import requests
import os
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px

# ---------------- API CONFIG ----------------
API_KEY = os.environ.get('API_KEY', '2f2c69d11da6fbec145966359891dd84')
LAT = 11.0168     # Change to your plant location
LON = 76.9558

# ---------------- Model Configuration ----------------
MODEL_PATH = 'solar_model.joblib'
model = None

# ---------------- Dash App Init ----------------
app = Dash(__name__)
server = app.server  # For Gunicorn
app.title = "Solar Prediction Dashboard"

# ---------------- Helper Functions ----------------
def load_or_train_model():
    """Load a trained model or train a new one if none exists."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load(MODEL_PATH)
            print("Loaded pre-trained model")
        else:
            print("Training new model...")
            df = load_and_prepare('Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv')
            model, r2, rmse = train_model(df)
            dump(model, MODEL_PATH)
            print(f"Trained new model - R²: {r2:.3f}, RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"Error loading/training model: {e}")
        # Fallback to a simple model
        model = LinearRegression()
        model.fit([[0, 0, 0, 0, 0]], [0])

def load_and_prepare(gen_file, weather_file):
    """Load and prepare the dataset."""
    try:
        gen = pd.read_csv(gen_file)
        weather = pd.read_csv(weather_file)

        # Fix date parsing
        gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], format='%d-%m-%Y %H:%M', errors='coerce')
        weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], format='%d-%m-%Y %H:%M', errors='coerce')

        gen.dropna(subset=['DATE_TIME'], inplace=True)
        weather.dropna(subset=['DATE_TIME'], inplace=True)

        df = pd.merge(gen, weather, on='DATE_TIME', how='inner')
        df = df[['DATE_TIME', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

        df.rename(columns={
            'DC_POWER': 'solar_output',
            'AMBIENT_TEMPERATURE': 'temperature',
            'MODULE_TEMPERATURE': 'module_temp',
            'IRRADIATION': 'irradiance'
        }, inplace=True)

        df = df[df['solar_output'] > 0]
        df['hour'] = df['DATE_TIME'].dt.hour
        df['month'] = df['DATE_TIME'].dt.month
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Error in load_and_prepare: {e}")
        # Return a minimal valid dataframe
        return pd.DataFrame({
            'DATE_TIME': [pd.Timestamp.now()],
            'solar_output': [0],
            'temperature': [25],
            'module_temp': [30],
            'irradiance': [500],
            'hour': [12],
            'month': [pd.Timestamp.now().month]
        })

def train_model(df):
    """Train the prediction model."""
    try:
        X = df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
        y = df['solar_output']
        
        # Use a smaller model for production
        model = RandomForestRegressor(
            n_estimators=50,  # Reduced from 200
            max_depth=8,      # Reduced from 12
            n_jobs=1,         # Use only 1 core
            random_state=42
        )
        
        # Use a subset of data for training if dataset is large
        if len(X) > 10000:
            X = X.sample(10000, random_state=42)
            y = y.loc[X.index]
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, r2, rmse
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        # Return a simple model if training fails
        model = LinearRegression()
        model.fit([[0, 0, 0, 0, 0]], [0])
        return model, 0, 0

def read_csv(contents):
    """Read CSV file from upload."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

# ---------------- Fetch Weather Forecast ----------------
def fetch_weather_forecast():
    """Fetch weather forecast from OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10).json()

        future_data = []
        for entry in response.get('list', [])[:24]:  # Next 24 hours
            dt = pd.to_datetime(entry['dt'], unit='s')
            temp = entry['main']['temp']
            cloud = entry['clouds']['all']
            irradiance = max(0, 1000 * (1 - cloud / 100))  # Approximate irradiance

            future_data.append({
                'DATE_TIME': dt,
                'temperature': temp,
                'module_temp': temp + 5,  # Simple approximation
                'irradiance': irradiance,
                'hour': dt.hour,
                'month': dt.month
            })

        return future_data

    except Exception as e:
        print(f"Error fetching weather: {e}")
        return []

# ---------------- UI Layout ----------------
app.layout = html.Div([
    html.H1("🌞 Solar Power Prediction Dashboard", style={'textAlign': 'center'}),
    html.Div([
        dcc.Upload(
            id='upload-gen',
            children=html.Div(['Drag and Drop or ', html.A('Select Generation Data')]),
            style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px', 'display': 'inline-block'}
        ),
        dcc.Upload(
            id='upload-weather',
            children=html.Div(['Drag and Drop or ', html.A('Select Weather Data')]),
            style={'width': '45%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px', 'display': 'inline-block'}
        ),
    ]),
    html.Div(id='output-data-upload'),
    html.Div([
        html.Button('Train Model', id='train-button', n_clicks=0, 
                   style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
        html.Div(id='model-metrics')
    ]),
    html.Hr(),
    html.Div([
        html.Button('Get 24-Hour Forecast', id='forecast-button', n_clicks=0,
                   style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
        dcc.Loading(
            id="loading-forecast",
            type="default",
            children=html.Div(id='forecast-output')
        )
    ]),
    dcc.Graph(id='forecast-plot'),
    html.Div(id='forecast-table')
])

# ---------------- Callbacks ----------------
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('train-button', 'disabled')],
    [Input('upload-gen', 'contents'),
     Input('upload-weather', 'contents')],
    [State('upload-gen', 'filename'),
     State('upload-weather', 'filename')]
)
def update_dashboard(gen_content, weather_content, gen_name, weather_name):
    if gen_content is None or weather_content is None:
        return "Please upload both generation and weather data files.", True

    try:
        gen_df = read_csv(gen_content)
        weather_df = read_csv(weather_content)
        
        if gen_df is None or weather_df is None:
            return "Error reading one or both files. Please check the file formats.", True

        # Save files for later use
        gen_df.to_csv(gen_name, index=False)
        weather_df.to_csv(weather_name, index=False)

        return f"Successfully uploaded {gen_name} and {weather_name}. Click 'Train Model' to proceed.", False

    except Exception as e:
        return f"Error processing files: {str(e)}", True

@app.callback(
    Output('model-metrics', 'children'),
    [Input('train-button', 'n_clicks')]
)
def train_model_callback(n_clicks):
    if n_clicks == 0:
        return ""
    
    try:
        global model
        df = load_and_prepare('Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv')
        model, r2, rmse = train_model(df)
        dump(model, MODEL_PATH)
        
        return html.Div([
            html.P(f"Model trained successfully!"),
            html.P(f"R² Score: {r2:.3f}"),
            html.P(f"RMSE: {rmse:.2f}")
        ])
    except Exception as e:
        return html.Div(f"Error training model: {str(e)}", style={'color': 'red'})

@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('forecast-table', 'children')],
    [Input('forecast-button', 'n_clicks')]
)
def forecast_solar(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return {}, ""
    
    try:
        future_data = fetch_weather_forecast()
        if not future_data:
            return {}, "Failed to fetch weather data"
            
        forecast_df = pd.DataFrame(future_data)
        
        if model is None:
            load_or_train_model()
            
        forecast_df['Predicted_Output'] = model.predict(
            forecast_df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
        )
        
        fig = px.line(
            forecast_df,
            x='DATE_TIME',
            y='Predicted_Output',
            title='24-Hour Solar Power Forecast',
            labels={'Predicted_Output': 'Predicted Output (kW)', 'DATE_TIME': 'Date/Time'}
        )
        
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in forecast_df.columns],
            data=forecast_df.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
        
        return fig, html.Div([html.H3("Forecast Data"), table])
        
    except Exception as e:
        print(f"Error in forecast_solar: {e}")
        return {}, html.Div(f"Error generating forecast: {str(e)}", style={'color': 'red'})

# Load the model when the app starts
load_or_train_model()

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)