import streamlit as st

from forecasting import (
    forecast_prices,
    forecast_linear_regression,
    forecast_exponential_smoothing,
    forecast_arima,
)
import plots


def render_forecasting(filtered_df):
    st.subheader("🔮 Forecast Next Month Prices")

    if filtered_df is None or filtered_df.empty:
        st.info("Not enough data to forecast.")
        return

    periods = st.slider("Months to Forecast", 1, 24, 3)

    st.markdown("### Select Models to Compare")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        use_prophet = st.checkbox("Prophet", value=True)
    with col_m2:
        use_linear = st.checkbox("Linear Regression", value=True)
    with col_m3:
        use_exp = st.checkbox("Exponential Smoothing", value=True)
    with col_m4:
        use_arima = st.checkbox("ARIMA", value=True)

    forecasts = {}
    errors = []

    if use_prophet:
        try:
            forecast_df, _ = forecast_prices(filtered_df, periods=periods, smooth=True)
            forecasts["Prophet"] = forecast_df
        except Exception as exc:
            errors.append(f"Prophet: {exc}")

    if use_linear:
        try:
            forecast_df, _ = forecast_linear_regression(filtered_df, periods=periods)
            forecasts["Linear Regression"] = forecast_df
        except Exception as exc:
            errors.append(f"Linear Regression: {exc}")

    if use_exp:
        try:
            forecast_df, _ = forecast_exponential_smoothing(filtered_df, periods=periods)
            forecasts["Exponential Smoothing"] = forecast_df
        except Exception as exc:
            errors.append(f"Exponential Smoothing: {exc}")

    if use_arima:
        try:
            forecast_df, _ = forecast_arima(filtered_df, periods=periods)
            forecasts["ARIMA"] = forecast_df
        except Exception as exc:
            errors.append(f"ARIMA: {exc}")

    if errors:
        with st.expander("⚠️ Model Errors", expanded=False):
            for err in errors:
                st.warning(err)

    if forecasts:
        st.markdown("### 📊 Model Comparison")
        fig_compare = plots.plot_forecast_comparison(forecasts)
        st.plotly_chart(fig_compare, width="stretch")

        st.markdown("### 📈 Individual Model Forecasts")
        for model_name, forecast_df in forecasts.items():
            with st.expander(f"{model_name} Details", expanded=False):
                fig = plots.plot_forecast(forecast_df)
                fig.update_layout(title=f"{model_name} Forecast")
                st.plotly_chart(fig, width="stretch")

                st.dataframe(
                    forecast_df.style.format(
                        {
                            "y": "{:.2f}",
                            "yhat": "{:.2f}",
                            "yhat_lower": "{:.2f}",
                            "yhat_upper": "{:.2f}",
                        }
                    )
                )
    else:
        st.warning("No models could generate forecasts. Try adjusting your data filters.")
