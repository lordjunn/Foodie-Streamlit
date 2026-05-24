import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render_compare(filtered_df):
    st.header("🔄 Month-to-Month Comparison")

    comp_df = filtered_df.dropna(subset=["numeric_price", "date"]).copy()
    if comp_df.empty:
        st.info("Not enough data for comparison. Adjust filters or scrape more data.")
        return

    comp_df["year_month"] = comp_df["date"].dt.to_period("M").astype(str)
    available_months = sorted(comp_df["year_month"].unique())

    if len(available_months) < 2:
        st.warning("Need at least 2 months of data to compare.")
        return

    c1, c2 = st.columns(2)
    with c1:
        month_a = st.selectbox(
            "Month A",
            available_months,
            index=max(0, len(available_months) - 2),
            key="comp_a",
        )
    with c2:
        month_b = st.selectbox(
            "Month B",
            available_months,
            index=len(available_months) - 1,
            key="comp_b",
        )

    df_a = comp_df[comp_df["year_month"] == month_a]
    df_b = comp_df[comp_df["year_month"] == month_b]

    st.markdown("### 📊 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)

    avg_a = df_a["numeric_price"].mean()
    avg_b = df_b["numeric_price"].mean()
    k1.metric(f"Avg Price ({month_a})", f"RM {avg_a:.2f}")
    k1.metric(f"Avg Price ({month_b})", f"RM {avg_b:.2f}", f"{avg_b - avg_a:+.2f}")

    count_a, count_b = len(df_a), len(df_b)
    k2.metric(f"Total Meals ({month_a})", f"{count_a}")
    k2.metric(f"Total Meals ({month_b})", f"{count_b}", f"{count_b - count_a:+d}")

    spend_a = df_a["numeric_price"].sum()
    spend_b = df_b["numeric_price"].sum()
    k3.metric(f"Total Spend ({month_a})", f"RM {spend_a:.2f}")
    k3.metric(
        f"Total Spend ({month_b})",
        f"RM {spend_b:.2f}",
        f"RM {spend_b - spend_a:+.2f}",
    )

    rest_a = df_a["restaurant_name"].nunique()
    rest_b = df_b["restaurant_name"].nunique()
    k4.metric(f"Restaurants ({month_a})", f"{rest_a}")
    k4.metric(f"Restaurants ({month_b})", f"{rest_b}", f"{rest_b - rest_a:+d}")

    st.markdown("---")

    st.subheader("📈 Price Distribution Comparison")
    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Histogram(x=df_a["numeric_price"], name=month_a, opacity=0.6, nbinsx=15)
    )
    fig_comp.add_trace(
        go.Histogram(x=df_b["numeric_price"], name=month_b, opacity=0.6, nbinsx=15)
    )
    fig_comp.update_layout(
        barmode="overlay",
        xaxis_title="Price (RM)",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.subheader("🏆 Top Restaurants Comparison")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown(f"**{month_a}**")
        rest_a_counts = df_a["restaurant_name"].value_counts().head(5).reset_index()
        rest_a_counts.columns = ["Restaurant", "Visits"]
        st.dataframe(rest_a_counts, width="stretch", hide_index=True)
    with cr2:
        st.markdown(f"**{month_b}**")
        rest_b_counts = df_b["restaurant_name"].value_counts().head(5).reset_index()
        rest_b_counts.columns = ["Restaurant", "Visits"]
        st.dataframe(rest_b_counts, width="stretch", hide_index=True)

    st.subheader("🍽️ Meal Type Comparison")
    cm1, cm2 = st.columns(2)
    with cm1:
        mt_a = df_a["meal_type"].value_counts()
        fig_ma = px.pie(values=mt_a.values, names=mt_a.index, title=month_a, hole=0.4)
        st.plotly_chart(fig_ma, width="stretch")
    with cm2:
        mt_b = df_b["meal_type"].value_counts()
        fig_mb = px.pie(values=mt_b.values, names=mt_b.index, title=month_b, hole=0.4)
        st.plotly_chart(fig_mb, width="stretch")
