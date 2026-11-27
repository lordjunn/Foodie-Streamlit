import requests
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Fast-path coordinates for common Malaysian towns / areas in dataset
KNOWN_AREA_COORDS = {
    "MMU Cyberjaya": (2.9274, 101.6418),
    "Cyberjaya": (2.9213, 101.6559),
    "Puchong": (3.0333, 101.6167),
    "Petaling Jaya": (3.1073, 101.6067),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Subang Jaya": (3.0565, 101.5851),
    "Putrajaya": (2.9264, 101.6964),
    "Seri Kembangan": (3.0235, 101.7042),
    "Damansara": (3.1350, 101.6210),
    "Bangsar": (3.1292, 101.6710),
}


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_area(area_name: str) -> tuple[float, float] | None:
    """Resolve an area name to (latitude, longitude) using known map or Nominatim."""
    if not area_name:
        return None

    cleaned = area_name.strip()
    for known, coords in KNOWN_AREA_COORDS.items():
        if known.lower() in cleaned.lower() or cleaned.lower() in known.lower():
            return coords

    # Fallback: Query OpenStreetMap Nominatim
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "Foodie-Streamlit-Recommender/1.0 (Personal food log explorer)"}
    params = {
        "q": f"{cleaned}, Malaysia",
        "format": "json",
        "limit": 1,
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning(f"Nominatim geocoding failed for {area_name}: {e}")

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nearby_food_venues(lat: float, lon: float, radius: int = 1500, limit: int = 8) -> list[dict]:
    """Query OpenStreetMap Overpass API for real restaurant/cafe amenities near coordinates."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:8];
    (
      node["amenity"="restaurant"](around:{radius},{lat},{lon});
      node["amenity"="cafe"](around:{radius},{lat},{lon});
      node["amenity"="fast_food"](around:{radius},{lat},{lon});
    );
    out body {limit * 2};
    """
    try:
        resp = requests.post(overpass_url, data={"data": query}, timeout=8)
        if resp.status_code != 200:
            return []

        elements = resp.json().get("elements", [])
        venues = []
        seen_names = set()

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name or name.strip() in seen_names:
                continue

            seen_names.add(name.strip())
            cuisine = tags.get("cuisine", "Local / General")
            amenity = tags.get("amenity", "restaurant").replace("_", " ").title()
            street = tags.get("addr:street", "")

            venues.append({
                "name": name.strip(),
                "cuisine": cuisine,
                "type": amenity,
                "street": street,
                "lat": el.get("lat"),
                "lon": el.get("lon"),
            })

            if len(venues) >= limit:
                break

        return venues
    except Exception as e:
        logger.warning(f"Overpass API query failed: {e}")
        return []
