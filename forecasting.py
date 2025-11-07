from prophet import Prophet
import pandas as pd

def forecast_prices(df, date_col='date', price_col='numeric_price', periods=3, freq='M'):
    """
    Forecast next few months of average prices using Prophet.
    - Aggregates by month only (not by restaurant)
    - Handles missing months safely (no zero filling)
    - Smooths slightly to reduce random spikes
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
    ##monthly_df = monthly_df[monthly_df['n'] >= 3]  # you can adjust this threshold; likely not needed
    monthly_df = monthly_df[['date', 'y']].rename(columns={'date': 'ds'})

    if monthly_df.empty or len(monthly_df) < 3:
        raise ValueError("Not enough valid monthly data to forecast.")

    # --- Fill missing months smoothly ---
    monthly_df = monthly_df.set_index('ds').asfreq(freq)
    monthly_df['y'] = monthly_df['y'].interpolate(method='linear', limit_direction='both')

    # --- Optional: Light smoothing (3-month rolling mean) ---
    monthly_df['y'] = monthly_df['y'].rolling(window=3, min_periods=1).mean()

    # --- Fit Prophet (trend-only, mild changepoint flexibility) ---
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.02  # less wiggly trend
    )
    model.fit(monthly_df.reset_index())

    # --- Forecast future months ---
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # --- Merge actuals + predictions for plotting ---
    merged = monthly_df.reset_index().merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds', how='outer'
    )

    return merged, model
