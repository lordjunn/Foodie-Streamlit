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