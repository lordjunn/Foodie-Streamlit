import streamlit as st
import pandas as pd
import numpy as np

from helpers import normalize_meal_type
import plots


def _highlight_change(val):
    if pd.isna(val):
        return ""
    color = "green" if val > 0 else ("red" if val < 0 else "gray")
    return f"color: {color}; font-weight: bold;"


def _format_period_label(period):
    return period.to_timestamp().strftime("%b %Y")


def _format_period_range(periods):
    if not periods:
        return ""
    start_label = _format_period_label(periods[0])
    end_label = _format_period_label(periods[-1])
    if start_label == end_label:
        return start_label
    return f"{start_label} to {end_label}"


def _build_restaurant_inflation_table(df, window_months=3, min_meals=3):
    months = sorted(df["year_month"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame(), "", ""

    window_months = max(1, min(window_months, len(months)))
    start_months = months[:window_months]
    end_months = months[-window_months:]

    start_df = df[df["year_month"].isin(start_months)]
    end_df = df[df["year_month"].isin(end_months)]

    start_stats = (
        start_df.groupby("restaurant_name")["numeric_price"]
        .agg(Start_Avg="mean", Start_Meals="count")
    )
    end_stats = (
        end_df.groupby("restaurant_name")["numeric_price"]
        .agg(End_Avg="mean", End_Meals="count")
    )

    table = start_stats.join(end_stats, how="inner")
    if table.empty:
        return pd.DataFrame(), _format_period_range(start_months), _format_period_range(end_months)

    table = table[(table["Start_Meals"] >= min_meals) & (table["End_Meals"] >= min_meals)]
    if table.empty:
        return pd.DataFrame(), _format_period_range(start_months), _format_period_range(end_months)

    table["Change (RM)"] = table["End_Avg"] - table["Start_Avg"]
    table["Change %"] = np.where(
        table["Start_Avg"] > 0,
        (table["Change (RM)"] / table["Start_Avg"]) * 100,
        np.nan,
    )

    table = table.reset_index().rename(columns={"restaurant_name": "Restaurant"})
    return table, _format_period_range(start_months), _format_period_range(end_months)


def render_data_stats(filtered_df, convert_df):
    display_df = filtered_df.copy()
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(
            display_df["date"], errors="coerce", dayfirst=True
        )

    st.dataframe(
        display_df,
        column_config={
            "date": st.column_config.DateColumn("date", format="DD/MM/YYYY")
        },
    )
    csv = convert_df(filtered_df)
    st.download_button("💾 Download Filtered CSV", csv, "filtered_menu_items.csv", "text/csv")

    st.subheader("📊 Quantitative Summary")
    df_stats = filtered_df.dropna(subset=["numeric_price"]).copy()
    if not df_stats.empty:
        df_stats["meal_category"] = df_stats["meal_type"].apply(normalize_meal_type)
        grouped_stats = (
            df_stats.groupby("meal_category")["numeric_price"]
            .agg(["mean", "median", "min", "max", "var", "std", "count"])
            .rename(
                columns={
                    "mean": "Mean",
                    "median": "Median",
                    "min": "Min",
                    "max": "Max",
                    "var": "Variance",
                    "std": "Std Dev",
                    "count": "Count",
                }
            )
        )

        total_stats = pd.DataFrame(
            {
                "Mean": [df_stats["numeric_price"].mean()],
                "Median": [df_stats["numeric_price"].median()],
                "Min": [df_stats["numeric_price"].min()],
                "Max": [df_stats["numeric_price"].max()],
                "Variance": [df_stats["numeric_price"].var()],
                "Std Dev": [df_stats["numeric_price"].std()],
                "Count": [df_stats["numeric_price"].count()],
            },
            index=["🍽️ Grand Total"],
        )

        full_stats = pd.concat([grouped_stats, total_stats])
        st.dataframe(
            full_stats.style.format(
                {
                    "Mean": "{:.2f}",
                    "Median": "{:.2f}",
                    "Min": "{:.2f}",
                    "Max": "{:.2f}",
                    "Variance": "{:.2f}",
                    "Std Dev": "{:.2f}",
                    "Count": "{:,.0f}",
                }
            )
        )

    st.subheader("📅 Monthly KPI Breakdown")
    if not filtered_df.empty and "numeric_price" in filtered_df.columns:
        monthly_df = filtered_df.dropna(subset=["numeric_price", "date"]).copy()
        monthly_df["year_month"] = monthly_df["date"].dt.to_period("M").astype(str)

        monthly_kpi = (
            monthly_df.groupby("year_month")
            .agg(
                Total_Meals=("numeric_price", "count"),
                Avg_Price=("numeric_price", "mean"),
                Min_Price=("numeric_price", "min"),
                Max_Price=("numeric_price", "max"),
                Most_Visited=(
                    "restaurant_name",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "-",
                ),
            )
            .reset_index()
        )

        monthly_kpi["Δ Avg"] = monthly_kpi["Avg_Price"].diff().round(2)
        monthly_kpi["% Change Avg"] = (
            monthly_kpi["Avg_Price"].pct_change() * 100
        ).round(2)
        monthly_kpi = monthly_kpi.set_index("year_month")

        st.dataframe(
            monthly_kpi.style.format(
                {
                    "Total_Meals": "{:,.0f}",
                    "Avg_Price": "RM {:.2f}",
                    "Min_Price": "RM {:.2f}",
                    "Max_Price": "RM {:.2f}",
                    "Δ Avg": "{:+.2f}",
                    "% Change Avg": "{:+.2f}%",
                }
            ).map(_highlight_change, subset=["Δ Avg", "% Change Avg"])
        )

    st.subheader("🏷️ Restaurant Inflation Tracker")
    inflation_df = filtered_df.dropna(
        subset=["numeric_price", "date", "restaurant_name"]
    ).copy()
    if not inflation_df.empty:
        inflation_df["date"] = pd.to_datetime(
            inflation_df["date"], errors="coerce", dayfirst=True
        )
        inflation_df = inflation_df.dropna(subset=["date"])
        inflation_df["year_month"] = inflation_df["date"].dt.to_period("M")

    if inflation_df.empty:
        st.info("Not enough data to compute restaurant inflation.")
    else:
        months = sorted(inflation_df["year_month"].dropna().unique())
        if len(months) < 2:
            st.info("Need at least 2 months of data to compare restaurant prices.")
        else:
            max_window = min(6, len(months))
            default_window = min(3, max_window)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                window_months = st.slider(
                    "Window months",
                    min_value=1,
                    max_value=max_window,
                    value=default_window,
                    key="inflation_window",
                )
            with col_b:
                min_meals = st.slider(
                    "Min meals per period",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="inflation_min_meals",
                )
            with col_c:
                sort_choice = st.selectbox(
                    "Sort",
                    [
                        "Biggest increase",
                        "Biggest decrease",
                        "Largest absolute change",
                    ],
                    key="inflation_sort",
                )

            top_n = st.slider(
                "Top rows to show",
                min_value=5,
                max_value=30,
                value=15,
                key="inflation_top_n",
            )

            table, start_label, end_label = _build_restaurant_inflation_table(
                inflation_df, window_months=window_months, min_meals=min_meals
            )

            if table.empty:
                st.info("Not enough restaurant activity in both periods.")
            else:
                st.caption(f"Comparing {start_label} vs {end_label}.")
                if sort_choice == "Biggest decrease":
                    table = table.sort_values("Change %", ascending=True, na_position="last")
                elif sort_choice == "Largest absolute change":
                    table = table.assign(_abs_change=table["Change %"].abs())
                    table = table.sort_values("_abs_change", ascending=False, na_position="last")
                else:
                    table = table.sort_values("Change %", ascending=False, na_position="last")

                display_table = table.head(top_n).drop(columns=["_abs_change"], errors="ignore")
                st.dataframe(
                    display_table.style.format(
                        {
                            "Start_Avg": "RM {:.2f}",
                            "End_Avg": "RM {:.2f}",
                            "Change (RM)": "{:+.2f}",
                            "Change %": "{:+.2f}%",
                            "Start_Meals": "{:,.0f}",
                            "End_Meals": "{:,.0f}",
                        }
                    ).map(_highlight_change, subset=["Change (RM)", "Change %"]),
                    width="stretch",
                )

    st.subheader("🏆 Top Restaurants and Dishes")
    if not filtered_df.empty:
        count_df = filtered_df.dropna(subset=["date"]).copy()
        if not count_df.empty:
            count_df["year_month"] = count_df["date"].dt.to_period("M").astype(str)
            total_meals = len(count_df)

            rest_total = count_df["restaurant_name"].value_counts().rename("Total Orders")
            rest_monthly = (
                count_df.groupby(["restaurant_name", "year_month"])\
                .size()
                .rename("Monthly Orders")
            )
            rest_max = rest_monthly.groupby("restaurant_name").max().rename("Max in Month")
            rest_table = pd.concat([rest_total, rest_max], axis=1).fillna(0)
            rest_table["Share %"] = (rest_table["Total Orders"] / total_meals * 100).round(2)
            rest_table = rest_table.reset_index().rename(columns={"index": "Restaurant"})
            rest_table["Total Orders"] = rest_table["Total Orders"].astype(int)
            rest_table["Max in Month"] = rest_table["Max in Month"].astype(int)
            rest_table = rest_table.sort_values("Total Orders", ascending=False)

            dish_total = count_df["dish_name"].value_counts().rename("Total Orders")
            dish_monthly = (
                count_df.groupby(["dish_name", "year_month"])\
                .size()
                .rename("Monthly Orders")
            )
            dish_max = dish_monthly.groupby("dish_name").max().rename("Max in Month")
            dish_table = pd.concat([dish_total, dish_max], axis=1).fillna(0)
            dish_table["Share %"] = (dish_table["Total Orders"] / total_meals * 100).round(2)
            dish_table = dish_table.reset_index().rename(columns={"index": "Dish"})
            dish_table["Total Orders"] = dish_table["Total Orders"].astype(int)
            dish_table["Max in Month"] = dish_table["Max in Month"].astype(int)
            dish_table = dish_table.sort_values("Total Orders", ascending=False)

            max_entries = max(len(rest_table), len(dish_table))
            if max_entries > 0:
                controls_col, _ = st.columns([1, 3])
                with controls_col:
                    show_all = st.checkbox("Show all", value=False, key="top_lists_all")
                    if show_all:
                        top_n = max_entries
                    else:
                        top_n = st.slider(
                            "Top list size",
                            min_value=5,
                            max_value=max(5, min(50, max_entries)),
                            value=min(10, max_entries),
                            step=1,
                            key="top_lists_size",
                        )
            else:
                top_n = 0

            if top_n:
                rest_table = rest_table.head(top_n)
                dish_table = dish_table.head(top_n)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### 🏠 Top Restaurants")
                st.dataframe(
                    rest_table.style.format({"Share %": "{:.2f}%"})
                )
            with col_right:
                st.markdown("### 🍜 Top Dishes")
                st.dataframe(
                    dish_table.style.format({"Share %": "{:.2f}%"})
                )
        else:
            st.info("Not enough dated entries to build top lists.")

    st.subheader("📊 Monthly LOWESS Summary (by Meal Type)")
    if not filtered_df.empty and "numeric_price" in filtered_df.columns:
        lowess_source = filtered_df.copy()
        lowess_source["meal_category"] = lowess_source["meal_type"].apply(
            normalize_meal_type
        )
        lowess_source["date"] = pd.to_datetime(
            lowess_source["date"], errors="coerce", dayfirst=True
        )
        fig_time = plots.plot_prices_over_time(lowess_source)

        if fig_time:
            lowess_traces = [
                t for t in fig_time.data if getattr(t, "mode", None) == "lines"
            ]

            if len(lowess_traces) == 0:
                st.info(
                    "No LOWESS trendlines found — try enabling trendline='lowess' in the plot."
                )
            else:
                cols = st.columns(2)
                col_index = 0

                for trace in lowess_traces:
                    meal_type = trace.name.replace("(lowess)", "").strip()
                    lowess_x = pd.to_datetime(trace.x, dayfirst=True)
                    lowess_y = trace.y

                    lowess_df = pd.DataFrame({"date": lowess_x, "lowess_price": lowess_y})
                    lowess_df["year_month"] = lowess_df["date"].dt.to_period("M").astype(str)

                    monthly_summary = (
                        lowess_df.groupby("year_month")["lowess_price"]
                        .agg(["min", "max", "mean"])
                        .rename(
                            columns={
                                "min": "LOWESS Min",
                                "max": "LOWESS Max",
                                "mean": "LOWESS Avg",
                            }
                        )
                        .round(2)
                    )

                    monthly_summary["Δ Avg"] = monthly_summary["LOWESS Avg"].diff().round(2)
                    monthly_summary["% Change Avg"] = (
                        monthly_summary["LOWESS Avg"].pct_change() * 100
                    ).round(2)

                    with cols[col_index]:
                        st.markdown(f"### 🍽️ {meal_type}")
                        st.dataframe(
                            monthly_summary.style.format(
                                {
                                    "LOWESS Min": "{:.2f}",
                                    "LOWESS Max": "{:.2f}",
                                    "LOWESS Avg": "{:.2f}",
                                    "Δ Avg": "{:+.2f}",
                                    "% Change Avg": "{:+.2f}%",
                                }
                            ).map(_highlight_change, subset=["Δ Avg", "% Change Avg"])
                        )

                    col_index = (col_index + 1) % 2
                    if col_index == 0:
                        cols = st.columns(2)
        else:
            st.info("Not enough data points to generate LOWESS summary.")

    st.subheader("📈 Monthly KPI by Meal Type")
    if not filtered_df.empty and "numeric_price" in filtered_df.columns:
        mt_df = filtered_df.dropna(subset=["numeric_price", "date"]).copy()
        mt_df["meal_category"] = mt_df["meal_type"].apply(normalize_meal_type)
        mt_df["year_month"] = mt_df["date"].dt.to_period("M").astype(str)

        grouped_mt = (
            mt_df.groupby(["meal_category", "year_month"])
            .agg(
                Total_Meals=("numeric_price", "count"),
                Avg_Price=("numeric_price", "mean"),
                Min_Price=("numeric_price", "min"),
                Max_Price=("numeric_price", "max"),
                Most_Visited=(
                    "restaurant_name",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "-",
                ),
            )
            .reset_index()
        )

        meal_types = grouped_mt["meal_category"].unique()
        cols = st.columns(2)
        col_index = 0

        for meal in meal_types:
            with cols[col_index]:
                st.markdown(f"### 🍽️ {meal}")
                mdf = (
                    grouped_mt[grouped_mt["meal_category"] == meal]
                    .set_index("year_month")
                    .drop(columns=["meal_category"])
                )
                mdf["Δ Avg"] = mdf["Avg_Price"].diff().round(2)
                mdf["% Change Avg"] = (mdf["Avg_Price"].pct_change() * 100).round(2)

                st.dataframe(
                    mdf.style.format(
                        {
                            "Total_Meals": "{:,.0f}",
                            "Avg_Price": "RM {:.2f}",
                            "Min_Price": "RM {:.2f}",
                            "Max_Price": "RM {:.2f}",
                            "Δ Avg": "{:+.2f}",
                            "% Change Avg": "{:+.2f}%",
                        }
                    ).map(_highlight_change, subset=["Δ Avg", "% Change Avg"])
                )

            col_index = (col_index + 1) % 2
            if col_index == 0:
                cols = st.columns(2)
