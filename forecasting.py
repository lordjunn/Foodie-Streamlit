from prophet import Prophet
import pandas as pd

def forecast_prices(df, date_col='date', price_col='numeric_price', periods=3, freq='M', smooth=False):
    """
    Forecast next few months of average prices using Prophet.
    - Aggregates by month only (not by restaurant)
    - Handles missing months safely (does not require zero filling)
    - Optional smoothing to reduce random spikes
    """

    # --- Clean and prep data ---
    df = df.dropna(subset=[date_col, price_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # --- Monthly average price ---
    monthly_df = (
        df.groupby(pd.Grouper(key=date_col, freq=freq))[price_col]
          .agg(['mean', 'count'])
          .reset_index()
          .rename(columns={'mean': 'y', 'count': 'n'})
    )

    # --- Drop very sparse months (too few data points) ---
    monthly_df = monthly_df[monthly_df['n'] >= 3]  # adjust threshold if needed
    monthly_df = monthly_df[['date', 'y']].rename(columns={'date': 'ds'})

    if monthly_df.empty or len(monthly_df) < 3:
        raise ValueError("Not enough valid monthly data to forecast.")

    # --- Optional: light smoothing ---
    if smooth:
        monthly_df['y'] = monthly_df['y'].rolling(window=3, min_periods=1).mean()

    # --- Fit Prophet (trend-only) ---
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05  # less wiggly trend
    )
    model.fit(monthly_df)

    # --- Forecast future months ---
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # --- Merge actuals + predictions for plotting ---
    merged = monthly_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds', how='outer'
    ).sort_values('ds')

    return merged, model


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