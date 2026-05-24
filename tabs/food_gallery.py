import streamlit as st
import pandas as pd
import plotly.express as px


def render_food_gallery(filtered_df, default_img):
    image_width = 220
    st.header("🍽️ Food Gallery")

    if filtered_df is None or filtered_df.empty:
        st.info("No data to display. Adjust filters or scrape more data.")
        return

    gallery_df = filtered_df.copy()

    if "image_url" not in gallery_df.columns:
        gallery_df["image_url"] = None

    gc1, gc2 = st.columns(2)
    with gc1:
        view_mode = st.radio(
            "View Mode",
            ["All Dishes", "By Dish", "By Restaurant"],
            horizontal=True,
            key="gallery_view",
        )
    with gc2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Date (Newest)",
                "Date (Oldest)",
                "Price (High → Low)",
                "Price (Low → High)",
                "A → Z",
                "Z → A",
            ],
            key="gallery_sort",
        )

    gallery_df["image_url"] = gallery_df["image_url"].fillna("").astype(str).str.strip()
    invalid_mask = ~gallery_df["image_url"].str.startswith(("http://", "https://"))
    gallery_df.loc[invalid_mask, "image_url"] = ""
    gallery_df.loc[gallery_df["image_url"] == "", "image_url"] = default_img

    if view_mode == "By Dish":
        dish_opts = sorted(gallery_df["dish_name"].dropna().unique())
        if dish_opts:
            sel_dish = st.selectbox("Select Dish", dish_opts, key="gal_dish")
            gallery_df = gallery_df[gallery_df["dish_name"] == sel_dish]
        else:
            st.warning("No dishes found.")
    elif view_mode == "By Restaurant":
        rest_opts = sorted(gallery_df["restaurant_name"].dropna().unique())
        if rest_opts:
            sel_rest = st.selectbox("Select Restaurant", rest_opts, key="gal_rest")
            gallery_df = gallery_df[gallery_df["restaurant_name"] == sel_rest]
        else:
            st.warning("No restaurants found.")

    sort_cfg = {
        "Date (Newest)": ("date", False),
        "Date (Oldest)": ("date", True),
        "Price (High → Low)": ("numeric_price", False),
        "Price (Low → High)": ("numeric_price", True),
        "A → Z": ("dish_name", True),
        "Z → A": ("dish_name", False),
    }
    sort_col, sort_asc = sort_cfg[sort_by]
    gallery_df = gallery_df.sort_values(sort_col, ascending=sort_asc, na_position="last")

    max_gallery = 100
    if len(gallery_df) > max_gallery:
        st.info(
            f"Showing first {max_gallery} of {len(gallery_df)} items. Use filters to narrow down."
        )
        gallery_df = gallery_df.head(max_gallery)

    show_trend = st.checkbox("📈 Show Price Trends", key="gal_trend")
    if show_trend and not gallery_df.empty:
        trend_df = gallery_df.dropna(subset=["numeric_price", "date"]).copy()
        trend_df["label"] = trend_df["dish_name"] + " [" + trend_df["restaurant_name"] + "]"
        top_labels = trend_df["label"].value_counts().head(15).index
        trend_df = trend_df[trend_df["label"].isin(top_labels)]
        if not trend_df.empty:
            fig = px.line(
                trend_df.sort_values("date"),
                x="date",
                y="numeric_price",
                color="label",
                markers=True,
                title=f"Price Trends (Top {len(top_labels)} most ordered)",
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (RM)",
                legend_title="Dish [Restaurant]",
                height=500,
            )
            st.plotly_chart(fig, width="stretch")

    st.caption(f"Showing {len(gallery_df)} entries")

    if gallery_df.empty:
        st.info("No items match your current selection.")
        return

    per_page = 24
    total_pages = max(1, -(-len(gallery_df) // per_page))
    page = st.number_input("Page", 1, total_pages, 1, key="gal_page")
    start = (page - 1) * per_page
    page_df = gallery_df.iloc[start : start + per_page]

    if view_mode == "By Dish":
        for rest_name, grp in page_df.groupby("restaurant_name", sort=False):
            st.subheader(f"📍 {rest_name}")
            grp = grp.sort_values("date")
            with st.container(height=420):
                for _, row in grp.iterrows():
                    ci, ct = st.columns([1, 2])
                    with ci:
                        img = row.get("image_url")
                        if pd.notna(img) and img:
                            st.image(img, width=image_width)
                        else:
                            st.markdown("🍽️ *No image*")
                    with ct:
                        date_label = (
                            row["date"].strftime("%d %b %Y")
                            if pd.notna(row.get("date"))
                            else "—"
                        )
                        price_label = (
                            f"RM {row['numeric_price']:.2f}"
                            if pd.notna(row.get("numeric_price"))
                            else row.get("price", "—")
                        )
                        st.markdown(
                            f"**{date_label}** · {price_label} · _{row.get('meal_type', '')}_"
                        )
                        desc = row.get("description", "")
                        if pd.notna(desc) and desc not in ("No description", ""):
                            st.markdown(desc, unsafe_allow_html=True)
                    st.divider()

    elif view_mode == "By Restaurant":
        with st.container(height=600):
            for _, row in page_df.iterrows():
                ci, ct = st.columns([1, 2])
                with ci:
                    img = row.get("image_url")
                    if pd.notna(img) and img:
                        st.image(img, width=image_width)
                    else:
                        st.markdown("🍽️ *No image*")
                with ct:
                    date_label = (
                        row["date"].strftime("%d %b %Y")
                        if pd.notna(row.get("date"))
                        else "—"
                    )
                    price_label = (
                        f"RM {row['numeric_price']:.2f}"
                        if pd.notna(row.get("numeric_price"))
                        else row.get("price", "—")
                    )
                    st.markdown(f"**{row['dish_name']}** · {price_label}")
                    st.caption(f"📅 {date_label} · {row.get('meal_type', '')}")
                    desc = row.get("description", "")
                    if pd.notna(desc) and desc not in ("No description", ""):
                        st.markdown(desc, unsafe_allow_html=True)
                st.divider()

    else:
        cols_count = 3
        with st.container(height=600):
            for i in range(0, len(page_df), cols_count):
                cols = st.columns(cols_count)
                for j, col in enumerate(cols):
                    if i + j >= len(page_df):
                        break
                    row = page_df.iloc[i + j]
                    with col:
                        img = row.get("image_url")
                        if pd.notna(img) and img:
                            st.image(img, width=image_width)
                        date_label = (
                            row["date"].strftime("%d %b %Y")
                            if pd.notna(row.get("date"))
                            else "—"
                        )
                        price_label = (
                            f"RM {row['numeric_price']:.2f}"
                            if pd.notna(row.get("numeric_price"))
                            else row.get("price", "—")
                        )
                        st.markdown(f"**{row['dish_name']}**")
                        st.caption(
                            f"📍 {row['restaurant_name']} · {price_label} · {date_label}"
                        )
