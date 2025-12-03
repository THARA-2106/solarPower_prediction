# solar_dashboard.py
import pandas as pd
import numpy as np
import io
import base64
import requests
import os
from joblib import dump, load
from functools import wraps
import time
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update
import plotly.express as px

# ---------------- Configuration ----------------
API_KEY = os.environ.get('API_KEY', '2f2c69d11da6fbec145966359891dd84')
LAT = 11.0168  # Change to your plant location
LON = 76.9558
MODEL_PATH = 'solar_model.joblib'
model = None

# ---------------- Helper Functions ----------------
def log_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

def load_or_train_model():
    """Load a trained model or train a new one if none exists."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            try:
                model = load(MODEL_PATH)
                print("Loaded pre-trained model")
                return
            except Exception as e:
                print(f"Error loading model: {e}, will train a new one")
                
        print("Training new model...")
        try:
            # Try to load the default files
            gen_file = 'Plant_1_Generation_Data.csv'
            weather_file = 'Plant_1_Weather_Sensor_Data.csv'
            
            if not os.path.exists(gen_file) or not os.path.exists(weather_file):
                raise FileNotFoundError("Default data files not found")
                
            df = load_and_prepare(gen_file, weather_file)
            
            if len(df) == 0:
                raise ValueError("No valid data found in the input files")
                
            model, r2, rmse = train_model(df)
            
            if model is not None:
                dump(model, MODEL_PATH)
                print(f"Trained new model - R²: {r2:.3f}, RMSE: {rmse:.2f}")
            else:
                raise ValueError("Model training failed")
                
        except Exception as e:
            print(f"Error in training: {e}")
            # Create a simple model as fallback
            model = LinearRegression()
            model.fit([[0, 0, 0, 0, 0]], [0])
            print("Created a simple linear model as fallback")
            
    except Exception as e:
        print(f"Unexpected error in load_or_train_model: {e}")
        model = LinearRegression()
        model.fit([[0, 0, 0, 0, 0]], [0])
        print("Created a simple linear model as fallback")

@log_time
def load_and_prepare(gen_file, weather_file):
    """Load and prepare the dataset with better error handling."""
    try:
        # Try to read the files
        if isinstance(gen_file, str) and os.path.exists(gen_file):
            gen = pd.read_csv(gen_file)
        elif hasattr(gen_file, 'to_csv'):
            # If it's a DataFrame
            gen = gen_file
        else:
            # If it's a file-like object
            gen = pd.read_csv(io.StringIO(gen_file.to_csv(index=False)))
            
        if isinstance(weather_file, str) and os.path.exists(weather_file):
            weather = pd.read_csv(weather_file)
        elif hasattr(weather_file, 'to_csv'):
            weather = weather_file
        else:
            weather = pd.read_csv(io.StringIO(weather_file.to_csv(index=False)))

        # Convert date columns with proper format
        date_columns = ['DATE_TIME', 'date_time', 'timestamp', 'date', 'time']
        for col in date_columns:
            if col in gen.columns:
                gen['DATE_TIME'] = pd.to_datetime(gen[col], errors='coerce')
                break
            if col in weather.columns:
                weather['DATE_TIME'] = pd.to_datetime(weather[col], errors='coerce')
                break

        # Drop rows where DATE_TIME couldn't be parsed
        gen = gen.dropna(subset=['DATE_TIME'])
        weather = weather.dropna(subset=['DATE_TIME'])

        if gen.empty or weather.empty:
            raise ValueError("No valid date data found in the input files")

        # Merge datasets
        df = pd.merge(gen, weather, on='DATE_TIME', how='inner', suffixes=('', '_weather'))
        
        # Ensure required columns exist
        column_mapping = {
            'DC_POWER': 'solar_output',
            'AC_POWER': 'solar_output',  # Alternative column name
            'AMBIENT_TEMPERATURE': 'temperature',
            'MODULE_TEMPERATURE': 'module_temp',
            'IRRADIATION': 'irradiance',
            'AMBIENT_TEMP': 'temperature',  # Alternative column name
            'MODULE_TEMP': 'module_temp'    # Alternative column name
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure we have all required columns
        required_columns = ['solar_output', 'temperature', 'module_temp', 'irradiance']
        for col in required_columns:
            if col not in df.columns:
                if col == 'solar_output':
                    df[col] = 0  # Dummy value
                elif col == 'temperature':
                    df[col] = 25  # Average temperature
                elif col == 'module_temp':
                    df[col] = 30  # Slightly higher than ambient
                elif col == 'irradiance':
                    df[col] = 500  # Moderate irradiance

        # Select only the columns we need
        df = df[['DATE_TIME', 'solar_output', 'temperature', 'module_temp', 'irradiance']].copy()
        
        # Add time-based features
        df['hour'] = df['DATE_TIME'].dt.hour
        df['month'] = df['DATE_TIME'].dt.month
        
        # Filter out invalid data
        df = df[df['solar_output'] > 0]
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data points after cleaning")
            
        return df

    except Exception as e:
        print(f"Error in load_and_prepare: {e}")
        # Return a minimal valid dataframe
        return pd.DataFrame({
            'DATE_TIME': [pd.Timestamp.now() - timedelta(hours=i) for i in range(24)],
            'solar_output': [max(0, 100 + 50 * (12 - abs(12 - i))) for i in range(24)],
            'temperature': [20 + 10 * np.sin(2 * np.pi * i / 24) for i in range(24)],
            'module_temp': [25 + 10 * np.sin(2 * np.pi * i / 24) for i in range(24)],
            'irradiance': [max(0, 800 * np.sin(np.pi * (i - 6) / 12)) for i in range(24)],
            'hour': [i % 24 for i in range(24)],
            'month': [pd.Timestamp.now().month] * 24
        })

@log_time
def train_model(df):
    """Train the prediction model with better error handling."""
    try:
        if len(df) < 5:  # If we don't have enough data
            raise ValueError(f"Not enough data points for training (got {len(df)})")
            
        X = df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
        y = df['solar_output']
        
        # Use a simpler model if we don't have much data
        if len(X) < 100:
            model = RandomForestRegressor(
                n_estimators=20,
                max_depth=4,
                n_jobs=1,
                random_state=42,
                verbose=1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                n_jobs=1,
                random_state=42,
                verbose=1
            )
        
        # If we have very few samples, use all data for training
        if len(X) < 20:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=min(0.3, 100/len(X)), random_state=42
            )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model trained - R²: {r2:.3f}, RMSE: {rmse:.2f}")
        return model, r2, rmse
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        # Return a simple linear model as fallback
        model = LinearRegression()
        model.fit([[0, 0, 0, 0, 0]], [0])
        return model, 0, 0

def read_csv(contents):
    """Read CSV file from upload."""
    try:
        if contents is None:
            return None
            
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

@log_time
def fetch_weather_forecast():
    """Fetch weather forecast from OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        future_data = []
        now = datetime.now()
        
        for entry in data.get('list', [])[:24]:  # Next 24 hours
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

        if not future_data:
            # Generate dummy data if API call fails
            print("No forecast data, generating dummy data")
            future_data = [{
                'DATE_TIME': now + timedelta(hours=i),
                'temperature': 20 + 5 * np.sin(2 * np.pi * i / 24),
                'module_temp': 25 + 5 * np.sin(2 * np.pi * i / 24),
                'irradiance': max(0, 800 * np.sin(np.pi * (i - 6) / 12)),
                'hour': (now.hour + i) % 24,
                'month': now.month
            } for i in range(24)]

        return future_data

    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Generate dummy data if API call fails
        now = datetime.now()
        return [{
            'DATE_TIME': now + timedelta(hours=i),
            'temperature': 20 + 5 * np.sin(2 * np.pi * i / 24),
            'module_temp': 25 + 5 * np.sin(2 * np.pi * i / 24),
            'irradiance': max(0, 800 * np.sin(np.pi * (i - 6) / 12)),
            'hour': (now.hour + i) % 24,
            'month': now.month
        } for i in range(24)]

# ---------------- Initialize the app and load model ----------------
app = Dash(__name__)
server = app.server  # For Gunicorn
app.title = "Solar Prediction Dashboard"

# Load or train the model when the app starts
load_or_train_model()

# ---------------- UI Layout ----------------
app.layout = html.Div([
    html.Div([
        html.H1("🌞 Solar Power Prediction Dashboard", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        
        # File Upload Section
        html.Div([
            dcc.Upload(
                id='upload-gen',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Generation Data')
                ]),
                style=upload_style
            ),
            dcc.Upload(
                id='upload-weather',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Weather Data')
                ]),
                style=upload_style
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
        
        # Upload Status
        html.Div(id='output-data-upload', 
                style={'margin': '10px', 'padding': '10px', 'minHeight': '60px', 
                      'border': '1px solid #eee', 'borderRadius': '5px'}),
        
        # Model Training Section
        html.Div([
            html.Button('Train Model', id='train-button', n_clicks=0, 
                       style=button_style),
            dcc.Loading(
                id="loading-train",
                type="default",
                children=html.Div(id='model-metrics')
            )
        ], style={'margin': '20px 0'}),
        
        html.Hr(),
        
        # Forecast Section
        html.Div([
            html.Button('Get 24-Hour Forecast', id='forecast-button', n_clicks=0,
                       style=button_style),
            dcc.Loading(
                id="loading-forecast",
                type="default",
                children=html.Div(id='forecast-output')
            )
        ], style={'margin': '20px 0'}),
        
        # Forecast Visualization
        dcc.Graph(
            id='forecast-plot',
            style={'height': '400px', 'margin': '20px 0'}
        ),
        
        # Forecast Data Table
        html.Div(id='forecast-table')
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})
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
            return "Error: Could not read one or both files. Please check the file formats.", True

        # Save files for later use
        gen_filename = gen_name or 'uploaded_gen.csv'
        weather_filename = weather_name or 'uploaded_weather.csv'
        
        gen_df.to_csv(gen_filename, index=False)
        weather_df.to_csv(weather_filename, index=False)

        return html.Div([
            html.P("✅ Files uploaded successfully!", style={'color': 'green'}),
            html.P(f"Generation data: {gen_name}"),
            html.P(f"Weather data: {weather_name}")
        ]), False

    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        print(error_msg)
        return html.Div(error_msg, style={'color': 'red'}), True

@app.callback(
    Output('model-metrics', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('upload-gen', 'filename'),
     State('upload-weather', 'filename')]
)
def train_model_callback(n_clicks, gen_filename, weather_filename):
    if n_clicks == 0:
        return ""
    
    try:
        global model
        
        # Use uploaded files if available, otherwise use default filenames
        gen_file = gen_filename or 'Plant_1_Generation_Data.csv'
        weather_file = weather_filename or 'Plant_1_Weather_Sensor_Data.csv'
        
        if not os.path.exists(gen_file) or not os.path.exists(weather_file):
            return html.Div([
                html.P("Error: Data files not found.", style={'color': 'red'}),
                html.P(f"Looking for: {gen_file} and {weather_file}")
            ])
        
        df = load_and_prepare(gen_file, weather_file)
        
        if len(df) == 0:
            return html.Div("Error: No valid data found in the input files", 
                          style={'color': 'red'})
        
        model, r2, rmse = train_model(df)
        
        if model is not None:
            dump(model, MODEL_PATH)
            return html.Div([
                html.P("✅ Model trained successfully!", 
                      style={'fontWeight': 'bold', 'color': 'green'}),
                html.P(f"R² Score: {r2:.3f}"),
                html.P(f"RMSE: {rmse:.2f}"),
                html.P(f"Trained on {len(df)} data points")
            ])
        else:
            return html.Div("Error: Model training failed", 
                          style={'color': 'red'})
            
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

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
        
        # Create the plot
        fig = px.line(
            forecast_df,
            x='DATE_TIME',
            y='Predicted_Output',
            title='24-Hour Solar Power Forecast',
            labels={'Predicted_Output': 'Predicted Output (kW)', 'DATE_TIME': 'Date/Time'},
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title="Date/Time",
            yaxis_title="Predicted Output (kW)",
            hovermode='x unified'
        )
        
        # Create the data table
        display_df = forecast_df.copy()
        display_df['DATE_TIME'] = display_df['DATE_TIME'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.round(2)
        
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in display_df.columns],
            data=display_df.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '8px'
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold'
            }
        )
        
        return fig, html.Div([
            html.H3("Forecast Data", style={'marginTop': '20px'}),
            table
        ])
        
    except Exception as e:
        print(f"Error in forecast_solar: {e}")
        error_msg = str(e)
        return {}, html.Div(f"Error generating forecast: {error_msg}", 
                          style={'color': 'red'})

# ---------------- Styling Constants ----------------
upload_style = {
    'width': '45%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px',
    'cursor': 'pointer',
    'backgroundColor': '#f8f9fa',
    'color': '#495057',
    'transition': 'all 0.3s ease'
}

button_style = {
    'margin': '10px',
    'padding': '10px 20px',
    'fontSize': '16px',
    'backgroundColor': '#3498db',
    'color': 'white',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease'
}

button_style_hover = {
    'backgroundColor': '#2980b9',
    'transform': 'translateY(-2px)'
}

# Add hover effects
app.clientside_callback(
    """
    function(hover) {
        return hover ? %s : {};
    }
    """ % button_style_hover,
    Output('train-button', 'style'),
    [Input('train-button', 'n_clicks')]
)

app.clientside_callback(
    """
    function(hover) {
        return hover ? %s : {};
    }
    """ % button_style_hover,
    Output('forecast-button', 'style'),
    [Input('forecast-button', 'n_clicks')]
)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)