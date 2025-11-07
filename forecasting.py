from prophet import Prophet
import pandas as pd

def forecast_prices(df, date_col='date', price_col='numeric_price', periods=3, freq='M'):
    """Forecast next few months of average prices using Prophet with missing month handling."""
    
    # Drop missing values
    df = df.dropna(subset=[date_col, price_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Aggregate monthly
    monthly_df = df.groupby(pd.Grouper(key=date_col, freq=freq))[price_col].mean().reset_index()
    monthly_df.columns = ['ds', 'y']

    # Fill missing months
    all_months = pd.date_range(start=monthly_df['ds'].min(),
                               end=monthly_df['ds'].max(), freq=freq)
    monthly_df = monthly_df.set_index('ds').reindex(all_months).rename_axis('ds').reset_index()
    monthly_df['y'] = monthly_df['y'].fillna(0)  # or use .interpolate() for smooth trend

    # Fit Prophet
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False,
                    changepoint_prior_scale=0.05)  # less sensitive
    model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    model.fit(monthly_df)

    # Forecast future
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Merge for plotting
    merged = monthly_df.merge(forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='outer')

    return merged, model
