import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_dashboard(filtered_df):
    st.header("🏠 Dashboard Overview")

    if filtered_df is None or filtered_df.empty:
        st.info("No data to display. Adjust your filters or scrape more data.")
        return

    prices_valid = filtered_df["numeric_price"].dropna()
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Meals", f"{len(filtered_df):,}")
    d2.metric(
        "Total Spend",
        f"RM {prices_valid.sum():,.2f}" if not prices_valid.empty else "—",
    )
    d3.metric(
        "Avg per Meal",
        f"RM {prices_valid.mean():.2f}" if not prices_valid.empty else "—",
    )
    d4.metric("Unique Restaurants", filtered_df["restaurant_name"].nunique())
    date_min = filtered_df["date"].min()
    date_max = filtered_df["date"].max()
    if pd.notna(date_min) and pd.notna(date_max):
        d5.metric("Date Range", f"{date_min.strftime('%b %Y')} – {date_max.strftime('%b %Y')}")
    else:
        d5.metric("Date Range", "—")

    st.markdown("---")

    st.subheader("📈 Monthly Spending Trend")
    trend_df = filtered_df.dropna(subset=["numeric_price", "date"]).copy()
    if not trend_df.empty:
        trend_df["year_month"] = trend_df["date"].dt.to_period("M").astype(str)
        monthly_agg = (
            trend_df.groupby("year_month")
            .agg(
                total=("numeric_price", "sum"),
                avg=("numeric_price", "mean"),
                count=("numeric_price", "count"),
            )
            .reset_index()
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=monthly_agg["year_month"],
                y=monthly_agg["total"],
                name="Total Spend (RM)",
                marker_color="#636EFA",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_agg["year_month"],
                y=monthly_agg["avg"],
                name="Avg Price (RM)",
                yaxis="y2",
                line=dict(color="#EF553B", width=3),
                mode="lines+markers",
            )
        )
        fig_trend.update_layout(
            yaxis=dict(title="Total Spend (RM)"),
            yaxis2=dict(title="Avg Price (RM)", overlaying="y", side="right"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=400,
        )
        st.plotly_chart(fig_trend, width="stretch")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🏆 Top 10 Restaurants")
        top_rest = filtered_df["restaurant_name"].value_counts().head(10).reset_index()
        top_rest.columns = ["Restaurant", "Visits"]
        fig_rest = px.bar(
            top_rest,
            x="Visits",
            y="Restaurant",
            orientation="h",
            color="Visits",
            color_continuous_scale="Blues",
        )
        fig_rest.update_layout(yaxis=dict(autorange="reversed"), height=400, showlegend=False)
        st.plotly_chart(fig_rest, width="stretch")

    with col_r:
        st.subheader("🍜 Top 10 Most Ordered Dishes")
        top_dish = filtered_df["dish_name"].value_counts().head(10).reset_index()
        top_dish.columns = ["Dish", "Orders"]
        fig_dish = px.bar(
            top_dish,
            x="Orders",
            y="Dish",
            orientation="h",
            color="Orders",
            color_continuous_scale="Oranges",
        )
        fig_dish.update_layout(yaxis=dict(autorange="reversed"), height=400, showlegend=False)
        st.plotly_chart(fig_dish, width="stretch")

    col_dow, col_mt = st.columns(2)
    with col_dow:
        st.subheader("📅 Meals by Day of Week")
        dow_df = filtered_df.dropna(subset=["date"]).copy()
        if not dow_df.empty:
            dow_df["day_of_week"] = dow_df["date"].dt.day_name()
            dow_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            dow_counts = dow_df["day_of_week"].value_counts().reindex(dow_order).fillna(0)
            fig_dow = px.bar(
                x=dow_counts.index,
                y=dow_counts.values,
                labels={"x": "Day", "y": "Meals"},
                color=dow_counts.values,
                color_continuous_scale="Viridis",
            )
            fig_dow.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_dow, width="stretch")

    with col_mt:
        st.subheader("🍽️ Meal Type Distribution")
        mt_counts = filtered_df["meal_type"].value_counts()
        if not mt_counts.empty:
            fig_mt = px.pie(values=mt_counts.values, names=mt_counts.index, hole=0.4)
            fig_mt.update_layout(height=350)
            st.plotly_chart(fig_mt, width="stretch")

    st.markdown("---")
    st.subheader("🏅 Personal Records")
    if not prices_valid.empty:
        rec1, rec2, rec3, rec4 = st.columns(4)
        most_exp = filtered_df.loc[filtered_df["numeric_price"].idxmax()]
        cheapest_mask = filtered_df["numeric_price"] > 0
        if cheapest_mask.any():
            cheapest = filtered_df.loc[
                filtered_df.loc[cheapest_mask, "numeric_price"].idxmin()
            ]
        else:
            cheapest = filtered_df.loc[filtered_df["numeric_price"].idxmin()]

        rec1.metric(
            "💰 Most Expensive",
            f"RM {most_exp['numeric_price']:.2f}",
            most_exp["dish_name"][:30],
        )
        rec2.metric(
            "🪙 Cheapest",
            f"RM {cheapest['numeric_price']:.2f}",
            cheapest["dish_name"][:30],
        )

        most_visited = filtered_df["restaurant_name"].value_counts()
        if not most_visited.empty:
            rec3.metric(
                "🏠 Most Visited",
                most_visited.index[0][:20],
                f"{most_visited.iloc[0]} visits",
            )

        most_ordered = filtered_df["dish_name"].value_counts()
        if not most_ordered.empty:
            rec4.metric(
                "🔁 Most Ordered",
                most_ordered.index[0][:20],
                f"{most_ordered.iloc[0]} times",
            )
