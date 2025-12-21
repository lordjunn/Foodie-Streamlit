import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm

def plot_box_price_by_category(df):
    fig = px.box(
        df, 
        x='meal_category', 
        y='numeric_price',
        color='meal_category', 
        points="all",
        hover_data=['dish_name', 'restaurant_name']
    )
    fig.update_layout(xaxis_title="Meal Category", yaxis_title="Price")
    return fig

def plot_qq_normality_check(numeric_prices):
    sorted_prices = np.sort(numeric_prices)
    theoretical_quantiles = np.sort(np.random.normal(numeric_prices.mean(), numeric_prices.std(), len(sorted_prices)))
    qq_df = pd.DataFrame({
        "Theoretical Quantiles": theoretical_quantiles,
        "Sample Quantiles": sorted_prices
    })
    fig = px.scatter(
        qq_df, 
        x="Theoretical Quantiles", 
        y="Sample Quantiles",
        hover_data=[qq_df.index]
    )
    fig.add_shape(
        type="line",
        x0=qq_df["Theoretical Quantiles"].min(),
        y0=qq_df["Theoretical Quantiles"].min(),
        x1=qq_df["Theoretical Quantiles"].max(),
        y1=qq_df["Theoretical Quantiles"].max(),
        line=dict(color="red", dash="dash")
    )
    fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    return fig

def plot_price_distribution(numeric_prices):
    fig = px.histogram(numeric_prices, nbins=20, opacity=0.7, marginal=None)
    fig.update_traces(name='Prices', marker_color='blue')

    # Normal curve overlay
    mean, std = numeric_prices.mean(), numeric_prices.std()
    x_vals = np.linspace(numeric_prices.min(), numeric_prices.max(), 200)
    bin_width = (numeric_prices.max() - numeric_prices.min()) / 20
    y_vals = norm.pdf(x_vals, mean, std) * len(numeric_prices) * bin_width

    fig.add_scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='red'), name='Normal Curve')
    return fig

def plot_prices_over_time(df):
    # Ensure numeric_price is float
    df = df.copy()
    df['numeric_price'] = pd.to_numeric(df['numeric_price'], errors='coerce')
    df = df.dropna(subset=['numeric_price'])
    
    if len(df) < 2:
        return None

    fig = px.scatter(
        df,
        x='date',
        y='numeric_price',
        color='meal_category',
        hover_data=['dish_name', 'restaurant_name'],
        trendline='lowess',
        title="Food Prices Over Time"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Meal Type"
    )
    return fig

def plot_calendar_heatmap(df):
    # Count meals per day
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    fig = px.scatter(
        daily_counts,
        x="date",
        y=[1] * len(daily_counts), # Dummy Y axis
        size="count",
        color="count",
        color_continuous_scale="Reds",
        title="Meal Frequency Intensity",
        height=250
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig

def plot_forecast(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(width=0),
        name='Confidence Interval'
    ))

    fig.update_layout(title='Predicted Monthly Average Price', xaxis_title='Date', yaxis_title='Price (RM)')
    return fig
