from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def prepare_monthly_data(df, date_col='date', price_col='numeric_price', freq='M', min_points=3):
    """
    Prepare monthly aggregated data for forecasting.
    """
    df = df.dropna(subset=[date_col, price_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    monthly_df = (
        df.groupby(pd.Grouper(key=date_col, freq=freq))[price_col]
          .agg(['mean', 'count'])
          .reset_index()
          .rename(columns={'mean': 'y', 'count': 'n'})
    )

    monthly_df = monthly_df[monthly_df['n'] >= min_points]
    monthly_df = monthly_df[['date', 'y']].rename(columns={'date': 'ds'})
    
    return monthly_df


def forecast_prices(df, date_col='date', price_col='numeric_price', periods=3, freq='M', smooth=False):
    """
    Forecast using Prophet (trend-only).
    """
    monthly_df = prepare_monthly_data(df, date_col, price_col, freq)

    if monthly_df.empty or len(monthly_df) < 3:
        raise ValueError("Not enough valid monthly data to forecast.")

    if smooth:
        monthly_df['y'] = monthly_df['y'].rolling(window=3, min_periods=1).mean()

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(monthly_df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    merged = monthly_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds', how='outer'
    ).sort_values('ds')

    return merged, model


def forecast_linear_regression(df, date_col='date', price_col='numeric_price', periods=3, freq='M'):
    """
    Forecast using Linear Regression on time index.
    """
    monthly_df = prepare_monthly_data(df, date_col, price_col, freq)

    if monthly_df.empty or len(monthly_df) < 3:
        raise ValueError("Not enough valid monthly data to forecast.")

    monthly_df = monthly_df.sort_values('ds').reset_index(drop=True)
    monthly_df['time_idx'] = np.arange(len(monthly_df))

    X = monthly_df[['time_idx']].values
    y = monthly_df['y'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict on historical + future
    last_date = monthly_df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    
    future_df = pd.DataFrame({
        'ds': future_dates,
        'time_idx': np.arange(len(monthly_df), len(monthly_df) + periods)
    })

    all_df = pd.concat([monthly_df[['ds', 'time_idx', 'y']], future_df], ignore_index=True)
    all_df['yhat'] = model.predict(all_df[['time_idx']].values)

    # Simple confidence interval (±1 std of residuals)
    residuals = y - model.predict(X)
    std_err = residuals.std()
    all_df['yhat_lower'] = all_df['yhat'] - 1.96 * std_err
    all_df['yhat_upper'] = all_df['yhat'] + 1.96 * std_err

    return all_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']], model


def forecast_exponential_smoothing(df, date_col='date', price_col='numeric_price', periods=3, freq='M'):
    """
    Forecast using Holt-Winters Exponential Smoothing.
    """
    monthly_df = prepare_monthly_data(df, date_col, price_col, freq)

    if monthly_df.empty or len(monthly_df) < 4:
        raise ValueError("Not enough valid monthly data for Exponential Smoothing (need at least 4).")

    monthly_df = monthly_df.sort_values('ds').reset_index(drop=True)
    y = monthly_df['y'].values

    # Use additive trend, no seasonality (too few data points typically)
    model = ExponentialSmoothing(
        y,
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    ).fit()

    # Forecast
    forecast_values = model.forecast(periods)

    # Build result DataFrame
    last_date = monthly_df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    future_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values
    })

    historical_df = monthly_df.copy()
    historical_df['yhat'] = model.fittedvalues

    all_df = pd.concat([historical_df, future_df], ignore_index=True)

    # Simple confidence interval based on residuals
    residuals = y - model.fittedvalues
    std_err = residuals.std()
    all_df['yhat_lower'] = all_df['yhat'] - 1.96 * std_err
    all_df['yhat_upper'] = all_df['yhat'] + 1.96 * std_err

    return all_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']], model


def forecast_arima(df, date_col='date', price_col='numeric_price', periods=3, freq='M', order=(1, 1, 1)):
    """
    Forecast using ARIMA model.
    """
    monthly_df = prepare_monthly_data(df, date_col, price_col, freq)

    if monthly_df.empty or len(monthly_df) < 5:
        raise ValueError("Not enough valid monthly data for ARIMA (need at least 5).")

    monthly_df = monthly_df.sort_values('ds').reset_index(drop=True)
    y = monthly_df['y'].values

    model = ARIMA(y, order=order).fit()

    # Forecast
    forecast_result = model.get_forecast(steps=periods)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Build result DataFrame
    last_date = monthly_df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    future_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': conf_int[:, 0],
        'yhat_upper': conf_int[:, 1]
    })

    historical_df = monthly_df.copy()
    historical_df['yhat'] = model.fittedvalues
    # For ARIMA, first few values may be NaN due to differencing
    historical_df['yhat_lower'] = np.nan
    historical_df['yhat_upper'] = np.nan

    all_df = pd.concat([historical_df, future_df], ignore_index=True)

    return all_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']], model


def plot_forecast_comparison(forecasts_dict):
    """
    Plot multiple forecast models on the same chart for comparison.
    
    Args:
        forecasts_dict: dict of {model_name: forecast_df}
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    colors = {
        'Prophet': '#1f77b4',
        'Linear Regression': '#ff7f0e',
        'Exponential Smoothing': '#2ca02c',
        'ARIMA': '#d62728'
    }
    
    # Plot actuals once (from first model)
    first_key = list(forecasts_dict.keys())[0]
    first_df = forecasts_dict[first_key]
    actuals = first_df.dropna(subset=['y'])
    
    fig.add_trace(go.Scatter(
        x=actuals['ds'],
        y=actuals['y'],
        mode='markers+lines',
        name='Actual',
        line=dict(color='black', width=2),
        marker=dict(size=8)
    ))
    
    # Plot each model's forecast
    for model_name, forecast_df in forecasts_dict.items():
        color = colors.get(model_name, '#7f7f7f')
        
        # Confidence interval (shaded)
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            name=f'{model_name} CI'
        ))
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name=model_name,
            line=dict(color=color, width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Forecast Model Comparison',
        xaxis_title='Date',
        yaxis_title='Avg Price (RM)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig