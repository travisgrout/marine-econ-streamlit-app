import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from textwrap import wrap
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Ocean Economy Estimates from Public QCEW Data",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """
    Loads, cleans, and prepares the dataset from a CSV file.
    This function is cached to enhance performance.
    """
    try:
        df = pd.read_csv("DORADO_combined_sectors.csv")
        
        # Rename columns for consistency
        rename_dict = {
            "NQ_establishments": "NQ_Establishments",
            "NQ_employment": "NQ_Employment",
            "NQ_wages": "NQ_Wages",
            "NQ_rw": "NQ_RealWages", 
        }
        df.rename(columns=rename_dict, inplace=True)
        
        # Explicitly convert all potential metric columns to numeric types.
        metric_cols_to_convert = [
            'NQ_Establishments', 'NQ_Employment', 'NQ_Wages', 'NQ_RealWages', 'NQ_GDP', 'NQ_RealGDP',
            'Establishments', 'Employment', 'Wages', 'GDP', 'RealGDP'
        ]
        
        for col in metric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except FileNotFoundError:
        st.error("Error: The data file 'DORADO_combined_sectors.csv' was not found. Please ensure it is in the same directory as the app script.")
        return None

# Load the data
dorado_results = load_data()

# --- Helper Functions ---
def format_value(x, metric):
    """Formats numbers with commas and appropriate currency symbols."""
    if pd.isna(x):
        return "N/A"
    if metric in ["Wages (not inflation-adjusted)", "Real Wages", "GDP (nominal)", "Real GDP"]:
        return f"${x:,.0f}"
    else:
        return f"{x:,.0f}"

def get_sector_colors(n):
    """Provides a list of distinct, colorblind-friendly colors."""
    base_colors = [
        "#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
        "#CC6677", "#AA4499", "#882255", "#E69F00", "#56B4E9",
        "#009E73", "#F0E442"
    ]
    return base_colors[:n] if n <= len(base_colors) else alt.themes.get().schemes['tableau20'][:n]

# --- NEW: Function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    """
    Converts a Pandas DataFrame to a CSV string, encoded in UTF-8.
    This function is cached to improve performance.
    """
    return df.to_csv(index=False).encode('utf-8')


# --- Data Dictionaries for Expanders ---
SECTOR_DESCRIPTIONS = {
    "Living Resources": {
        "description": "The Living Resources sector includes industries engaged in the harvesting, processing, or selling of marine life. This encompasses commercial fishing, aquaculture (such as fish hatcheries and shellfish farming), seafood processing and packaging, and wholesale or retail seafood markets.",
        "table": pd.DataFrame([
            {"NAICS Code": "11251", "Years": "All years", "Description": "Fish Hatcheries and Aquaculture"},
            {"NAICS Code": "11411", "Years": "All years", "Description": "Fishing"},
            {"NAICS Code": "311710", "Years": "2012 - present", "Description": "Seafood Product Preparation and Packaging"},
            {"NAICS Code": "424460", "Years": "All years", "Description": "Fish and Seafood Merchant Wholesalers"},
            {"NAICS Code": "445250", "Years": "2022 - present", "Description": "Fish and Seafood Retailers"},
            {"NAICS Code": "311711", "Years": "2001 - 2011", "Description": "Seafood Canning"},
            {"NAICS Code": "311712", "Years": "2001 - 2011", "Description": "Fresh and Frozen Seafood Processing"},
            {"NAICS Code": "445220", "Years": "2001 - 2021", "Description": "Fish and Seafood Markets"},
        ])
    },
    "Marine Construction": {
        "description": "The Marine Construction sector is composed of establishments involved in heavy and civil engineering construction that is related to the marine environment, such as dredging, pier construction, and beach nourishment.",
        "table": pd.DataFrame([
            {"NAICS Code": "237990", "Years": "All years", "Description": "Other Heavy and Civil Engineering Construction"}
        ])
    },
    "Marine Transportation": {
        "description": "The Marine Transportation sector includes industries that provide transportation for freight and passengers on the deep sea, coastal waters, or the Great Lakes. It also covers support activities essential for water transport, such as port and harbor operations, marine cargo handling, and navigational services. The manufacturing of search and navigation equipment and warehousing services are also included in this sector.",
        "table": pd.DataFrame([
            {"NAICS Code": "334511", "Years": "All years", "Description": "Search, Detection, Navigation, Guidance, Aeronautical, and Nautical System and Instrument Manufacturing"},
            {"NAICS Code": "48311", "Years": "All years", "Description": "Marine Freight and Passenger Transport"},
            {"NAICS Code": "4883", "Years": "All years", "Description": "Marine Transportation Services"},
            {"NAICS Code": "4931", "Years": "All years", "Description": "Warehousing"},
        ])
    },
    "Offshore Mineral Resources": {
        "description": "The Offshore Mineral Resources sector consists of industries involved in the exploration and extraction of minerals from the seafloor. This includes the extraction of crude petroleum and natural gas, the mining of sand and gravel, and support activities such as drilling and geophysical exploration.",
        "table": pd.DataFrame([
            {"NAICS Code": "211120", "Years": "2017 - present", "Description": "Crude Petroleum Extraction"},
            {"NAICS Code": "211130", "Years": "2017 - present", "Description": "Natural Gas Extraction"},
            {"NAICS Code": "212321", "Years": "All years", "Description": "Construction Sand and Gravel Mining"},
            {"NAICS Code": "212322", "Years": "All years", "Description": "Industrial Sand Mining"},
            {"NAICS Code": "213111", "Years": "All years", "Description": "Drilling Oil and Gas Wells"},
            {"NAICS Code": "213112", "Years": "All years", "Description": "Support Activities for Oil and Gas Operations"},
            {"NAICS Code": "541360", "Years": "All years", "Description": "Geophysical Surveying and Mapping Services"},
            {"NAICS Code": "211111", "Years": "2001 - 2016", "Description": "Crude Petroleum and Natural Gas Extraction"},
            {"NAICS Code": "211112", "Years": "2001 - 2016", "Description": "Natural Gas Liquid Extraction"},
        ])
    },
    "Ship and Boat Building": {
        "description": "The Ship and Boat Building sector is composed of establishments that build, repair, and maintain ships and recreational boats.",
        "table": pd.DataFrame([
            {"NAICS Code": "33661", "Years": "All years", "Description": "Ship and Boat Building"}
        ])
    },
    "Tourism and Recreation": {
        "description": "The Tourism and Recreation sector comprises a diverse group of industries that provide goods and services to people enjoying coastal recreation. This includes businesses such as full-service and limited-service restaurants, hotels and motels, marinas, boat dealers, and charter fishing operations. It also includes scenic water tours, sporting goods manufacturers, recreational instruction, and attractions like aquaria and nature parks. Note that four of the codes used in this sector are not in the original ENOW dataset.",
        "table": pd.DataFrame([
            {"NAICS Code": "339920", "Years": "All years", "Description": "Sporting and Athletic Goods Manufacturing"},
            {"NAICS Code": "441222", "Years": "All years", "Description": "Boat Dealers"},
            {"NAICS Code": "487210", "Years": "All years", "Description": "Scenic and Sightseeing Transportation, Water"},
            {"NAICS Code": "487990", "Years": "All years", "Description": "Scenic and Sightseeing Transportation, Other"},
            {"NAICS Code": "532284", "Years": "2017 - present", "Description": "Recreational Goods Rental"},
            {"NAICS Code": "611620", "Years": "All years", "Description": "Sports and Recreation Instruction"},
            {"NAICS Code": "712130", "Years": "All years", "Description": "Zoos and Botanical Gardens"},
            {"NAICS Code": "712190", "Years": "All years", "Description": "Nature Parks and Other Similar Institutions"},
            {"NAICS Code": "713110", "Years": "All years", "Description": "Amusement and Theme Parks"},
            {"NAICS Code": "713930", "Years": "All years", "Description": "Marinas"},
            {"NAICS Code": "713990", "Years": "All years", "Description": "All Other Amusement and Recreation Industries"},
            {"NAICS Code": "721110", "Years": "All years", "Description": "Hotels (except Casino Hotels) and Motels"},
            {"NAICS Code": "721191", "Years": "All years", "Description": "Bed-and-Breakfast Inns"},
            {"NAICS Code": "721199", "Years": "All years", "Description": "All Other Traveler Accommodation"},
            {"NAICS Code": "721211", "Years": "All years", "Description": "RV (Recreational Vehicle) Parks and Campgrounds"},
            {"NAICS Code": "721214", "Years": "All years", "Description": "Recreational and Vacation Camps (except Campgrounds)"},
            {"NAICS Code": "722410", "Years": "All years", "Description": "Drinking Places (Alcoholic Beverages)"},
            {"NAICS Code": "722511", "Years": "2012 - present", "Description": "Full-Service Restaurants"},
            {"NAICS Code": "722513", "Years": "2012 - present", "Description": "Limited-Service Restaurants"},
            {"NAICS Code": "722514", "Years": "2012 - present", "Description": "Cafeterias, Grill Buffets, and Buffets"},
            {"NAICS Code": "722515", "Years": "2012 - present", "Description": "Snack and Nonalcoholic Beverage Bars"},
            {"NAICS Code": "532292", "Years": "2001 - 2016", "Description": "Recreational Goods Rental"},
            {"NAICS Code": "722110", "Years": "2001 - 2011", "Description": "Full-Service Restaurants"},
            {"NAICS Code": "722211", "Years": "2001 - 2011", "Description": "Limited-Service Restaurants"},
            {"NAICS Code": "722212", "Years": "2001 - 2011", "Description": "Cafeterias, Grill Buffets, and Buffets"},
            {"NAICS Code": "722213", "Years": "2001 - 2011", "Description": "Snack and Nonalcoholic Beverage Bars"},
        ])
    }
}

# --- Dictionary for Metric Descriptions ---
METRIC_DESCRIPTIONS = {
    "Employment": "Employment estimates in Open ENOW are based on the sum of annual average employment reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates employment in the Louisiana Marine Transportation Sector based on reported annual average employment in four NAICS codes (334511, 483
