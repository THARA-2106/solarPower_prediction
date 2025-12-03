# solar_dashboard_forecasting.py
# Solar Power Prediction + Next 24 Hours Forecasting

import pandas as pd
import numpy as np
import io, base64, requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px


# ---------------- API CONFIG ----------------
API_KEY = "2f2c69d11da6fbec145966359891dd84"  # <--- Replace this!
LAT = 11.0168     # Change to your plant location
LON = 76.9558


# ---------------- Dash App Init ----------------
app = Dash(__name__)
server = app.server  # For Gunicorn
app.title = "Solar Prediction Dashboard"


# ---------------- Helper Functions ----------------
def load_and_prepare(gen_file, weather_file):
    gen = pd.read_csv(gen_file)
    weather = pd.read_csv(weather_file)

    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], dayfirst=True, errors='coerce')
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=True, errors='coerce')

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


def train_model(df):
    global model
    X = df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
    y = df['solar_output']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, rmse


def read_csv(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))


# ---------------- Fetch Weather Forecast ----------------
def fetch_weather_forecast():
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    future_data = []

    for entry in response['list'][:24]:  # Next 24 hours (3-hour interval)
        dt = pd.to_datetime(entry['dt'], unit='s')
        temp = entry['main']['temp']
        cloud = entry['clouds']['all']

        # Approximation of irradiance based on cloud cover
        irradiance = max(0, 1000 * (1 - cloud / 100))

        future_data.append({
            'DATE_TIME': dt,
            'temperature': temp,
            'module_temp': temp + 5,
            'irradiance': irradiance,
            'hour': dt.hour,
            'month': dt.month
        })

    return pd.DataFrame(future_data)


# ---------------- UI Layout ----------------
app.layout = html.Div([
    html.H1("🌞 Solar Power Prediction Dashboard", style={'textAlign': 'center'}),

    html.Div([
        dcc.Upload(id='gen-upload',
                   children=html.Div(['Drag & Drop or ', html.A('Select Generation CSV')]),
                   style={'width': '45%', 'display': 'inline-block', 'border': '1px dashed gray', 'padding': '10px'}),

        dcc.Upload(id='weather-upload',
                   children=html.Div(['Drag & Drop or ', html.A('Select Weather CSV')]),
                   style={'width': '45%', 'display': 'inline-block', 'border': '1px dashed gray', 'padding': '10px',
                          'marginLeft': '5%'})
    ], style={'marginBottom': '20px'}),

    html.Button("Train & Predict", id='train-btn', n_clicks=0, style={'marginBottom': '20px'}),
    html.Div(id='metrics-output', style={'fontSize': '18px', 'fontWeight': 'bold'}),

    dcc.Graph(id='prediction-graph'),
    html.Div(id='table-output', style={'marginBottom': '40px'}),

    html.Hr(),

    html.Button("Forecast Next 24 Hours", id='forecast-btn', n_clicks=0, style={'marginBottom': '20px'}),
    dcc.Graph(id='forecast-graph'),
    html.Div(id='forecast-table')
])


# ---------------- Model Training Callback ----------------
@app.callback(
    [Output('metrics-output', 'children'),
     Output('prediction-graph', 'figure'),
     Output('table-output', 'children')],
    [Input('train-btn', 'n_clicks')],
    [State('gen-upload', 'contents'),
     State('weather-upload', 'contents'),
     State('gen-upload', 'filename'),
     State('weather-upload', 'filename')]
)
def update_dashboard(n_clicks, gen_content, weather_content, gen_name, weather_name):
    if n_clicks == 0 or not gen_content or not weather_content:
        return "Upload both files & click Train", {}, None

    gen_df = read_csv(gen_content)
    weather_df = read_csv(weather_content)

    df = load_and_prepare(io.StringIO(gen_df.to_csv(index=False)),
                          io.StringIO(weather_df.to_csv(index=False)))

    global model
    model, r2, rmse = train_model(df)

    df['Predicted_Output'] = model.predict(df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']])

    fig = px.line(df.head(200), x='DATE_TIME', y=['solar_output', 'Predicted_Output'],
                  title="Actual vs Predicted Solar Output")

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.head(20).to_dict('records')
    )

    return f"✅ Model trained! R²={r2:.3f}, RMSE={rmse:.2f}", fig, table


# ---------------- Future Forecast Callback ----------------
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-table', 'children')],
    Input('forecast-btn', 'n_clicks')
)
def forecast_solar(n_clicks):
    if n_clicks == 0:
        return {}, None

    forecast_df = fetch_weather_forecast()
    forecast_df['Predicted_Output'] = model.predict(
        forecast_df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
    )

    fig = px.line(forecast_df, x='DATE_TIME', y='Predicted_Output',
                  title="🌅 Solar Forecast - Next 24 Hours", markers=True)

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in forecast_df.columns],
        data=forecast_df.to_dict('records')
    )

    return fig, table


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
