"""
Script to extract unique restaurants from food.csv and categorize them
into standardized location/area tags based on Malaysian geography,
MMU Cyberjaya campus context, and home dining.
"""

import csv
import json
from collections import Counter
from pathlib import Path

CSV_FILE = Path("food.csv")
OUTPUT_JSON = Path("restaurant_areas.json")

# Standardized location mapping for Malaysian geography & MMU campus context
KNOWN_RESTAURANT_AREAS = {
    # MMU Cyberjaya (Campus cafeteria spots, student clubs, campus events, booths)
    "Komatra": "MMU Cyberjaya",
    "Dapo Sahang": "MMU Cyberjaya",
    "HTC": "MMU Cyberjaya",
    "HSH": "MMU Cyberjaya",
    "Home Sweet Home": "MMU Cyberjaya",
    "Warung Membawang": "MMU Cyberjaya",
    "Cita Rasa Delima": "MMU Cyberjaya",
    "Royal Tandoori": "MMU Cyberjaya",
    "The Royal Tandoori": "MMU Cyberjaya",
    "Starbees": "MMU Cyberjaya",
    "Abang Izz": "MMU Cyberjaya",
    "Mira Kitchen": "MMU Cyberjaya",
    "Azeey Kopitiam": "MMU Cyberjaya",
    "Rasa Shiokkk": "MMU Cyberjaya",
    "Tuah Anas": "MMU Cyberjaya",
    "Deens Cafe": "MMU Cyberjaya",
    "Pak Ngah": "MMU Cyberjaya",
    "Criss Burger": "MMU Cyberjaya",
    "Bakery": "MMU Cyberjaya",
    "BBC": "MMU Cyberjaya",
    "Hirup Mee Kopitiam": "MMU Cyberjaya",
    "Switt Kreving": "MMU Cyberjaya",
    "AGK.Letop": "MMU Cyberjaya",
    "Win Week Aug 2022": "MMU Cyberjaya",
    "Win Week Oct 2022": "MMU Cyberjaya",
    "GDC Game Jam": "MMU Cyberjaya",
    "GDC AGM": "MMU Cyberjaya",
    "SRC": "MMU Cyberjaya",
    "SRC CW": "MMU Cyberjaya",
    "CodeNection 2025": "MMU Cyberjaya",
    "SEAMUN": "MMU Cyberjaya",
    "Meowcha.MMU": "MMU Cyberjaya",
    "Vendor #1": "MMU Cyberjaya",
    "Vendor #2": "MMU Cyberjaya",
    "KUDRAT": "MMU Cyberjaya",
    "Thailand booth": "MMU Cyberjaya",
    "MDA Catering": "MMU Cyberjaya",
    "Ai-Sandwich": "MMU Cyberjaya",

    # Cyberjaya (Commercial spots, shop lots, and malls outside campus)
    "Sky Malaysia": "Cyberjaya",
    "Mama Homecook Mixed Rice": "Cyberjaya",
    "AS Nasi Kandar": "Cyberjaya",
    "Hinz Kitchen": "Cyberjaya",
    "Dapur Chilli": "Cyberjaya",
    "McD Edusphere": "Cyberjaya",
    "KFC Edusphere": "Cyberjaya",
    "McD Prima Avenue": "Cyberjaya",
    "Domino's Prima Avenue": "Cyberjaya",
    "Domino's Cyberjaya Galleria": "Cyberjaya",
    "Restoran Al Bidayah": "Cyberjaya",
    "Fu Jee Ulu Yam Loh Mee": "Cyberjaya",
    "Burger King": "Cyberjaya",
    "Salam Noodles": "Cyberjaya",
    "Warung Hanna Crystalville": "Cyberjaya",
    "Ri Yue Ming": "Cyberjaya",
    "Famous Bistro Cafe": "Cyberjaya",
    "Chef Yas": "Cyberjaya",
    "Ummifa": "Cyberjaya",
    "Cosmo Kitchen": "Cyberjaya",
    "RMY Solutions": "Cyberjaya",
    "Rice Home Restaurant": "Cyberjaya",
    "Glaze Eatery": "Cyberjaya",
    "Subway": "Cyberjaya",
    "Chennai Spice Express": "Cyberjaya",
    "Pizza Hut": "Cyberjaya",
    "IZ Kitchen": "Cyberjaya",
    "Nur Char Kuey Teow Penang": "Cyberjaya",
    "10 Gram": "Cyberjaya",
    "Ayam Gepuk Pak Gembus Glomac": "Cyberjaya",
    "Mr Mee": "Cyberjaya",
    "Faizal Burger": "Cyberjaya",
    "Pasta Syok": "Cyberjaya",
    "Ayam Gepuk Shilla": "Cyberjaya",
    "Taco Bell": "Cyberjaya",
    "GO Noodle House": "Cyberjaya",
    "Restoran Nasi Kandar D\"Tas Maju": "Cyberjaya",
    "A & W": "Cyberjaya",
    "Thai Luck": "Cyberjaya",
    "Bungkus Kaw Kaw": "Cyberjaya",
    "Diem Canteen": "Cyberjaya",
    "Ayam Geprek Indo Asli": "Cyberjaya",
    "Restoran Indorasa": "Cyberjaya",
    "Aurora Express": "Cyberjaya",
    "Jin Man Cyberjaya": "Cyberjaya",
    "Ayak Gepuk Tok Mat Cyberjaya": "Cyberjaya",
    "Jinja Shiokkk": "Cyberjaya",
    "Ayam Gepuk Mas Rio": "Cyberjaya",
    "Amir Damascus": "Cyberjaya",

    # Puchong (Internship spots, Bandar Puteri, Kinrara, etc.)
    "Double D Restaurant Puchong": "Puchong",
    "Soon Hing Bdr Puteri": "Puchong",
    "BP72 Mixed Rice": "Puchong",
    "3V's Bistro Nasi Kandar": "Puchong",
    "Ramai Food Centre": "Puchong",
    "Wang Bu Liao": "Puchong",
    "Taste Memories": "Puchong",
    "Old Town Puchong Yong Tou Fu Restaurant": "Puchong",
    "Puteri Mart": "Puchong",
    "Nasi Lemak BK4": "Puchong",
    "Face To Face Noodle House": "Puchong",
    "Mr Bistro": "Puchong",
    "Holiao Noodles Puchong": "Puchong",
    "LBM Puchong": "Puchong",
    "Yuan Fong 3 Kopitiam": "Puchong",
    "Nasi Kandar Kok Siong": "Puchong",
    "Kopi Home": "Puchong",
    "S11 Puteri 5": "Puchong",
    "Restoran EFJC": "Puchong",
    "Restoran Wai Wai": "Puchong",
    "Thai Grandma Canteen": "Puchong",
    "Ayam Gepuk Pak Gembus": "Puchong",
    "Ayam Gepuk Top Global": "Puchong",

    # Kuala Lumpur
    "Connaught Night Market": "Kuala Lumpur",
    "Isetan KLCC": "Kuala Lumpur",
    "Norazlina Chong Char Kuey Teow": "Kuala Lumpur",
    "NAK NAK TRX": "Kuala Lumpur",
    "Wok Grill, NU Sentral": "Kuala Lumpur",
    "MadChili Pan Mee, Berjaya Times Square": "Kuala Lumpur",
    "Ken Chan Curry, NU Sentral": "Kuala Lumpur",
    "Ttukbaegi MyTown": "Kuala Lumpur",
    "Boat Noodle": "Kuala Lumpur",
    "Suria Food Court": "Kuala Lumpur",
    "Uncle Soon Fried Rice": "Kuala Lumpur",
    "McD Jalan Tun Perak": "Kuala Lumpur",
    "Texas Chicken": "Kuala Lumpur",
    "Don Don Donki": "Kuala Lumpur",

    # Petaling Jaya
    "Food District - Amin Chicken": "Petaling Jaya",
    "Bananabro": "Petaling Jaya",
    "Restoran D.U. Cafe (新百乐茶餐室)": "Petaling Jaya",
    "Grab": "Petaling Jaya",

    # Putrajaya (IOI City Mall)
    "Jollibee": "Putrajaya",
    "Mr Dakgalbi": "Putrajaya",
    "Little Wok": "Putrajaya",
    "Sukiya": "Putrajaya",

    # Seri Kembangan
    "Jasir Bistro": "Seri Kembangan",

    # Home / Instant
    "Instant Noodles": "Home / Instant",
    "Home-Cooked: Lakefront Homes": "Home / Instant",
    "Home-Cooked: Mutiara Ville": "Home / Instant",
    "Chicken Flavour": "Home / Instant",
    "Spicy": "Home / Instant",

    # Unknown / Other
    "No restaurant name": "Unknown / Other",
    "Food delivery": "Unknown / Other",
    "?": "Unknown / Other",
    "DONDON BREAK": "Unknown / Other",
    "Pop Meals": "Unknown / Other",
}


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"'{CSV_FILE}' not found in the current directory.")

    # 1. Load food.csv and extract unique, non-null values from restaurant_name
    unique_restaurants = set()
    with open(CSV_FILE, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = (row.get("restaurant_name") or "").strip()
            if val:
                unique_restaurants.add(val)

    print(f"Loaded {CSV_FILE}: Found {len(unique_restaurants)} unique restaurant names.\n")

    # 2. Categorize each unique restaurant
    mapping = {}
    for r in sorted(unique_restaurants):
        area = KNOWN_RESTAURANT_AREAS.get(r, "Unknown / Other")
        mapping[r] = area

    # 3. Save resulting mapping to restaurant_areas.json
    with open(OUTPUT_JSON, mode="w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved restaurant area mapping to '{OUTPUT_JSON}'.\n")

    # 4. Print summary of unique location tags and counts
    summary = Counter(mapping.values())
    print("=" * 45)
    print(" Location Tag Summary")
    print("=" * 45)
    for tag, count in summary.most_common():
        print(f" - {tag:<22}: {count:>3} restaurants")
    print("-" * 45)
    print(f" Total Unique Restaurants: {len(mapping):>3}")
    print("=" * 45)


if __name__ == "__main__":
    main()
