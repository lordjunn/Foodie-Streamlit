import streamlit as st


def render_kpis(filtered_df):
    if filtered_df is None or filtered_df.empty:
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Meals", f"{len(filtered_df)}")
    if "numeric_price" in filtered_df.columns and not filtered_df["numeric_price"].dropna().empty:
        m2.metric("Avg Price", f"RM {filtered_df['numeric_price'].mean():.2f}")
        m3.metric("Highest Price", f"RM {filtered_df['numeric_price'].max():.2f}")
    else:
        m2.metric("Avg Price", "-")
        m3.metric("Highest Price", "-")
    m4.metric(
        "Most Visited",
        filtered_df["restaurant_name"].mode()[0] if not filtered_df.empty else "-",
    )
    st.markdown("---")
