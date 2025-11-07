from prophet import Prophet
import pandas as pd

def forecast_prices(df, date_col='date', price_col='numeric_price', periods=3, freq='M'):
    """Forecast next few months of average prices using Prophet."""
    # Ensure datetime
    df = df.dropna(subset=[date_col, price_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Aggregate monthly
    monthly_df = df.groupby(pd.Grouper(key=date_col, freq=freq))[price_col].mean().reset_index()
    monthly_df.columns = ['ds', 'y']

    # Fit Prophet model
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(monthly_df)

    # Predict future
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Merge actual and forecasted values for plotting
    merged = monthly_df.merge(forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='outer')
    return merged, model
