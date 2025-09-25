import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from textwrap import wrap
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Ocean Economy Estimates from Public QCEW Data",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_comparison_data():
    """
    Loads, cleans, and prepares the comparison dataset from enow_version_comparisons.csv.
    This data is used for the "Compare to original ENOW" mode.
    """
    try:
        df = pd.read_csv("enow_version_comparisons.csv")

        # RENAME COLUMNS FOR CONSISTENCY
        rename_dict = {
            "Open_establishments": "Open_Establishments",
            "Open_employment": "Open_Employment",
            "Open_wages": "Open_Wages",
            "Open_GDP": "Open_GDP",
            "Open_RealGDP": "Open_RealGDP",
            "oldENOW_establishments": "oldENOW_Establishments",
            "oldENOW_employment": "oldENOW_Employment",
            "oldENOW_wages": "oldENOW_Wages",
            "oldENOW_GDP": "oldENOW_GDP",
            "oldENOW_RealGDP": "oldENOW_RealGDP",
            "noimpute_establishments": "noimpute_Establishments",
            "noimpute_employment": "noimpute_Employment",
            "noimpute_wages": "noimpute_Wages",
            "noimpute_GDP": "noimpute_GDP",
            "noimpute_RealGDP": "noimpute_RealGDP"
        }
        df.rename(columns=rename_dict, inplace=True)

        # CONVERT METRIC COLUMNS TO NUMERIC
        metric_cols_to_convert = [
            'Open_Establishments', 'Open_Employment', 'Open_Wages', 'Open_GDP', 'Open_RealGDP',
            'oldENOW_Establishments', 'oldENOW_Employment', 'oldENOW_Wages', 'oldENOW_GDP', 'oldENOW_RealGDP',
            'noimpute_Establishments', 'noimpute_Employment', 'noimpute_Wages', 'noimpute_GDP', 'noimpute_RealGDP'
        ]
        for col in metric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_open_enow_data():
    """
    Loads, cleans, and prepares the new Open ENOW dataset from openENOWinput.csv.
    This data is used for the "State Estimates", "County Estimates", and "Regional Estimates" modes.
    """
    try:
        df = pd.read_csv("openENOWinput.csv")
        rename_dict = {
            "geoType": "GeoScale", "geoName": "GeoName", "state": "StateAbbrv",
            "year": "Year", "enowSector": "OceanSector", "establishments": "Open_Establishments",
            "employment": "Open_Employment", "wages": "Open_Wages", "real_wages": "Open_RealWages",
            "gdp": "Open_GDP", "rgdp": "Open_RealGDP"
        }
        df.rename(columns=rename_dict, inplace=True)
        df_original = pd.read_csv("openENOWinput.csv")
        if 'geoType' in df_original.columns:
            df['geoType'] = df_original['geoType']
        if 'stateName' in df_original.columns:
            df['stateName'] = df_original['stateName']
        metric_cols_to_convert = [
            'Open_Establishments', 'Open_Employment', 'Open_Wages', 'Open_RealWages', 'Open_GDP', 'Open_RealGDP'
        ]
        for col in metric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None

# Load both potential data sources
comparison_data = load_comparison_data()
open_enow_data = load_open_enow_data()

# --- Helper Functions ---
def format_value(x, metric):
    """Formats numbers with commas and appropriate currency symbols."""
    if pd.isna(x):
        return "N/A"
    if metric in ["Wages (not inflation-adjusted)", "Real Wages", "GDP (nominal)", "Real GDP", "Wages", "GDP"]:
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

# --- Function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    """
    Converts a Pandas DataFrame to a CSV string, encoded in UTF-8.
    """
    return df.to_csv(index=False).encode('utf-8')


# --- Data Dictionaries for Expanders ---
SECTOR_DESCRIPTIONS = {
    "Living Resources": {
        "description": "The Living Resources sector includes industries engaged in the harvesting, processing, or selling of marine life...",
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
        "description": "The Marine Construction sector is composed of establishments involved in heavy and civil engineering construction that is related to the marine environment...",
        "table": pd.DataFrame([
            {"NAICS Code": "237990", "Years": "All years", "Description": "Other Heavy and Civil Engineering Construction"}
        ])
    },
    "Marine Transportation": {
        "description": "The Marine Transportation sector includes industries that provide transportation for freight and passengers on the deep sea, coastal waters, or the Great Lakes...",
        "table": pd.DataFrame([
            {"NAICS Code": "334511", "Years": "All years", "Description": "Search, Detection, Navigation, Guidance, Aeronautical, and Nautical System and Instrument Manufacturing"},
            {"NAICS Code": "48311", "Years": "All years", "Description": "Marine Freight and Passenger Transport"},
            {"NAICS Code": "4883", "Years": "All years", "Description": "Marine Transportation Services"},
            {"NAICS Code": "4931", "Years": "All years", "Description": "Warehousing"},
        ])
    },
    "Offshore Mineral Resources": {
        "description": "The Offshore Mineral Resources sector consists of industries involved in the exploration and extraction of minerals from the seafloor...",
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
        "description": "The Tourism and Recreation sector comprises a diverse group of industries in coastal zip codes...",
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
METRIC_DESCRIPTIONS = {
    "Employment": "Employment estimates in Open ENOW are based on the sum of annual average employment reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates employment in the Louisiana Marine Transportation Sector based on reported annual average employment in four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline. To address gaps in public county-level QCEW data, Open ENOW imputes missing values based on data from other years or broader economic sectors.",
    "Wages (not inflation-adjusted)": "Open ENOW estimates wages paid to workers based on the sum of total annual wages paid reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates wages in the Louisiana Marine Transportation Sector based on reported annual wages paid in four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline. To address gaps in public county-level QCEW data, Open ENOW imputes missing values based on data from other years or broader economic sectors.",
    "Real Wages": "Open ENOW reports inflation-adjusted real wages in 2024 dollars. To estimate real wages, Open ENOW adjusts its nominal wage estimates for changes in the consumer price index (CPI).",
    "Establishments": "Open ENOW estimates the number of employers in a given marine sector based on the sum of establishments reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates the number of establishments in the Louisiana Marine Transportation Sector based on QCEW data for four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline.",
    "GDP (nominal)": "Open ENOW estimates a sector's contribution to GDP based on the average ratio of wages paid to GDP reported for the relevant industry in the Bureau of Economic Analysis (BEA) GDP by industry in current dollars (SAGDP2) table.",
    "Real GDP": "Real GDP is reported in 2017 dollars. Open ENOW estimates a sector's contribution to Real GDP based on the average ratio of wages paid to GDP reported for the relevant industry in the Bureau of Economic Analysis (BEA) Real GDP by industry in chained dollars (SAGDP9) table."
}


# --- Main Application ---
METRIC_MAP = {
    "Employment": "Employment",
    "Wages (not inflation-adjusted)": "Wages",
    "Real Wages": "RealWages",
    "Establishments": "Establishments",
    "GDP (nominal)": "GDP",
    "Real GDP": "RealGDP"
}

st.sidebar.image("open_ENOW_logo.png", width=200)

# --- START: CODE FOR POP-UP WINDOW ---
popover = st.sidebar.popover("What is Open ENOW?")
popover.markdown("""
This web app is a proof of concept. It displays preliminary results from an attempt to use publicly-available data to track economic activity in six sectors that depend on the oceans and Great Lakes. The Open ENOW dataset currently covers 30 coastal states and the years 2001-2023. **Neither the results, nor the underlying methods, have undergone peer review.**

**How is Open ENOW different from the original ENOW dataset?**

Open ENOW will, if developed into a publicly-released product, bridge a temporary gap in the Economics: National Ocean Watch (ENOW) dataset. The original ENOW dataset draws on establishment-level microdata collected by the Bureau of Labor Statistics (BLS). Due to resource constraints, BLS cannot currently support updates to the ENOW dataset.

The original ENOW dataset includes the data years 2005-2021. It does not capture substantial growth and changes in the ocean and Great Lakes economies since 2021 and, without annual updates, will become less and less relevant to users who want to understand current conditions and trends in marine economies. Open ENOW addresses this problem by creating “ENOW-like” estimates from public Quarterly Census of Employment and Wages (QCEW) data.

Open ENOW covers the same states and economic sectors as the original ENOW and reports the same economic metrics. Like ENOW, it is a useful tool for understanding state, regional, and national marine economies. Understanding the type of economic activities that depend on the oceans and Great Lakes can help to guide planning, management, and policy decisions. However, Open ENOW is different from the original ENOW dataset in several important respects:

* Open ENOW draws on less detailed data than the original ENOW dataset and uses imputed values to fill in data gaps. As a result, it is less authoritative than the original ENOW dataset.
* Open ENOW does not currently cover individual counties. The original ENOW dataset reports on marine economic activity in about 475 coastal U.S. counties.
* Open ENOW reports on a slightly different set of employers than the original ENOW.
""")
# --- END: CODE FOR POP-UP WINDOW ---

st.sidebar.header("Display Mode:")

# --- Custom Button Display Mode ---
# Map button labels to plot_mode values
button_map = {
    "States": "State Estimates from Public QCEW Data",
    "Counties": "County Estimates from Public QCEW Data",
    "Regions": "Regional Estimates from Public QCEW Data",
    "Compare": "Compare to original ENOW",
    "Error Analysis": "Error Analysis"
}

# Initialize session state for the plot mode
if 'plot_mode' not in st.session_state:
    st.session_state.plot_mode = button_map["States"]

# Custom CSS to style the primary button as NOAA Sea Blue and make all buttons larger
st.markdown("""
<style>
    /* Style for the selected (primary) button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #0085CA;
        color: white;
        border: 1px solid #0085CA;
        height: 3em;
    }
    /* Style for the unselected (secondary) buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# Function to handle button clicks and update state
def update_mode(mode_label):
    st.session_state.plot_mode = button_map[mode_label]

# Display buttons and handle state changes
cols1 = st.sidebar.columns(2)
with cols1[0]:
    is_selected = st.session_state.plot_mode == button_map["States"]
    st.button("States", on_click=update_mode, args=("States",), use_container_width=True, type="primary" if is_selected else "secondary")
with cols1[1]:
    is_selected = st.session_state.plot_mode == button_map["Counties"]
    st.button("Counties", on_click=update_mode, args=("Counties",), use_container_width=True, type="primary" if is_selected else "secondary")

cols2 = st.sidebar.columns(2)
with cols2[0]:
    is_selected = st.session_state.plot_mode == button_map["Regions"]
    st.button("Regions", on_click=update_mode, args=("Regions",), use_container_width=True, type="primary" if is_selected else "secondary")
with cols2[1]:
    is_selected = st.session_state.plot_mode == button_map["Compare"]
    st.button("Compare", on_click=update_mode, args=("Compare",), use_container_width=True, type="primary" if is_selected else "secondary", help="Compare to original ENOW")

# Add the fifth button for Error Analysis
cols3 = st.sidebar.columns(2)
with cols3[0]:
    is_selected = st.session_state.plot_mode == button_map["Error Analysis"]
    st.button("Error Analysis", on_click=update_mode, args=("Error Analysis",), use_container_width=True, type="primary" if is_selected else "secondary", help="Analyze differences between Open ENOW and original ENOW")


plot_mode = st.session_state.plot_mode


# --- Select Active DataFrame and Set Filters based on Mode ---
estimate_modes = [
    "State Estimates from Public QCEW Data",
    "County Estimates from Public QCEW Data",
    "Regional Estimates from Public QCEW Data"
]

# Initialize variables to be used later
selected_county_name = None
selected_state = None
geo_filter_type = None
all_geo_label = None
selected_geo = None

if plot_mode in estimate_modes:
    active_df = open_enow_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `openENOWinput.csv` is in the same directory as the app.")
        st.stop()

    if plot_mode == "State Estimates from Public QCEW Data":
        # Filter for State-level data aggregated by Sector
        active_df = active_df[(active_df['GeoScale'] == 'State') & (active_df['aggregation'] == 'Sector')].copy()
        geo_label = "Select State:"
        all_geo_label = "All Coastal States"
        geo_filter_type = 'State'
        geo_names = active_df["GeoName"].dropna().unique()
        unique_geos = [all_geo_label] + sorted(geo_names)
        selected_geo = st.sidebar.selectbox(geo_label, unique_geos)

    elif plot_mode == "County Estimates from Public QCEW Data":
        # Filter for County-level data
        active_df = active_df[(active_df['geoType'] == 'County') & (active_df['aggregation'] == 'Sector')].copy()
        geo_filter_type = 'County'
        all_geo_label = None # No "all" option for counties

        state_label = "Select State:"
        # Ensure stateName column exists and is used for filtering
        if 'stateName' in active_df.columns:
            state_names = sorted(active_df['stateName'].dropna().unique())
            selected_state = st.sidebar.selectbox(state_label, state_names)

            county_label = "Select County:"
            if selected_state:
                # Filter by state before getting unique county names
                county_names = sorted(
                    active_df[active_df['stateName'] == selected_state]['GeoName'].dropna().unique()
                )
                selected_county_name = st.sidebar.selectbox(county_label, county_names)
            else:
                selected_county_name = st.sidebar.selectbox(county_label, [])

            # Set selected_geo to the county for unified logic later
            selected_geo = selected_county_name
        else:
            st.warning("The 'stateName' column is not available in the data for county estimates.")
            selected_state = None
            selected_county_name = None


    else: # Regional Estimates from Public QCEW Data
        # Filter for Region-level data aggregated by Sector
        active_df = active_df[(active_df['GeoScale'] == 'Region') & (active_df['aggregation'] == 'Sector')].copy()
        geo_label = "Select Region:"
        all_geo_label = "All Regions"
        geo_filter_type = 'Region'
        geo_names = active_df["GeoName"].dropna().unique()
        unique_geos = [all_geo_label] + sorted(geo_names)
        selected_geo = st.sidebar.selectbox(geo_label, unique_geos)
    
    # --- DYNAMIC FILTERS FOR ESTIMATE MODES ---
    ocean_sectors = active_df["OceanSector"].dropna().unique()
    unique_sectors = ["All Marine Sectors"] + sorted(ocean_sectors)
    selected_sector = st.sidebar.selectbox("Select Sector:", unique_sectors)

    sorted_sector_names = sorted(ocean_sectors)
    colors_list = get_sector_colors(len(sorted_sector_names))
    sector_color_map = dict(zip(sorted_sector_names, colors_list))

    metric_choices = list(METRIC_MAP.keys())
    selected_display_metric = st.sidebar.selectbox("Select Metric:", metric_choices)
    selected_metric_internal = METRIC_MAP[selected_display_metric]

    min_year, max_year = int(active_df["Year"].min()), int(active_df["Year"].max())
    default_end_year = max_year
    default_start_year = max(min_year, max_year - 9)
    default_range = (default_start_year, default_end_year)
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=default_range,
        step=1
    )

    # --- DYNAMIC TITLE FOR ESTIMATE MODES ---
    title_sector_part = "All Marine Sectors" if selected_sector == "All Marine Sectors" else f"{selected_sector} Sector"
    if plot_mode == "County Estimates from Public QCEW Data":
        if selected_county_name and selected_state:
            plot_title = f"{selected_display_metric}: {title_sector_part} in {selected_county_name}, {selected_state}"
        else:
            plot_title = "Please select a state and county to view estimates"
    else:
        plot_title = f"{selected_display_metric}: {title_sector_part} in {selected_geo}"
    st.title(plot_title)

    # --- PLOTTING AND FILTERING FOR ESTIMATE MODES ---
    is_gdp_metric = selected_display_metric in ["GDP (nominal)", "Real GDP"]
    if is_gdp_metric:
        gdp_col_to_check = f"Open_{selected_metric_internal}"
        if not active_df.empty:
            gdp_is_missing_for_max_year = active_df.loc[active_df['Year'] == max_year, gdp_col_to_check].isnull().all()
            if gdp_is_missing_for_max_year:
                st.info(f"💡 GDP estimates are not yet available for {max_year}.")
    
    base_filtered_df = active_df[
        (active_df["Year"] >= year_range[0]) &
        (active_df["Year"] <= year_range[1])
    ]
    if plot_mode == "County Estimates from Public QCEW Data":
        if selected_county_name and selected_state:
            base_filtered_df = base_filtered_df[
                (base_filtered_df["GeoName"] == selected_county_name) &
                (base_filtered_df["stateName"] == selected_state)
            ]
        else:
            base_filtered_df = pd.DataFrame()
    elif all_geo_label and selected_geo == all_geo_label:
        pass
    else:
        base_filtered_df = base_filtered_df[base_filtered_df["GeoName"] == selected_geo]

    if selected_sector != "All Marine Sectors":
        base_filtered_df = base_filtered_df[base_filtered_df["OceanSector"] == selected_sector]

    y_label_map = {
        "GDP (nominal)": "GDP ($ millions)", "Real GDP": "Real GDP ($ millions, 2017)",
        "Wages (not inflation-adjusted)": "Wages ($ millions)", "Real Wages": "Real Wages ($ millions, 2024)",
        "Employment": "Employment (Number of Jobs)", "Establishments": "Establishments (Count)"
    }
    y_label = y_label_map.get(selected_display_metric, selected_display_metric)
    is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)", "Real Wages"]
    tooltip_format = '$,.0f' if is_currency else ',.0f'
    
    open_metric_col = f"Open_{selected_metric_internal}"
    
    summary_message = None
    latest_year = year_range[1]
    latest_year_data = base_filtered_df[base_filtered_df['Year'] == latest_year]

    if not latest_year_data.empty:
        latest_value = latest_year_data[open_metric_col].sum()
        if pd.notna(latest_value) and latest_value > 0:
            formatted_value = format_value(latest_value, selected_display_metric)
            summary_text_templates = {
                "Employment": f"Approximately <strong>{formatted_value}</strong> people were employed in the selected sector(s) in <strong>{latest_year}</strong>.",
                "Wages (not inflation-adjusted)": f"Workers in the selected sector(s) earned about <strong>{formatted_value}</strong> in total annual wages in <strong>{latest_year}</strong>.",
                "Real Wages": f"Workers in the selected sector(s) earned about <strong>{formatted_value}</strong> in total annual wages in <strong>{latest_year}</strong>, adjusted for inflation.",
                "Establishments": f"There were about <strong>{formatted_value}</strong> establishments in the selected sector(s) in <strong>{latest_year}</strong>.",
                "GDP (nominal)": f"The selected sector(s) contributed about <strong>{formatted_value}</strong> to GDP in <strong>{latest_year}</strong>.",
                "Real GDP": f"The selected sector(s) contributed about <strong>{formatted_value}</strong> to GDP in <strong>{latest_year}</strong> (in chained 2017 dollars)."
            }
            summary_message = summary_text_templates.get(selected_display_metric)

    if selected_sector == "All Marine Sectors":
        plot_df = base_filtered_df[["Year", "OceanSector", open_metric_col]].copy()
        plot_df.rename(columns={open_metric_col: "Estimate_value"}, inplace=True)
        plot_df.dropna(subset=["Estimate_value"], inplace=True)
        if is_currency:
            plot_df["Estimate_value"] /= 1e6
        if not plot_df.empty:
            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('sum(Estimate_value):Q', title=y_label, stack='zero'),
                color=alt.Color('OceanSector:N', scale=alt.Scale(domain=sorted_sector_names, range=colors_list), legend=alt.Legend(title="Sectors")),
                tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('OceanSector:N', title='Sector'), alt.Tooltip('sum(Estimate_value):Q', title=selected_display_metric, format=tooltip_format)]
            ).configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(symbolLimit=len(sorted_sector_names))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    else:
        if all_geo_label and selected_geo == all_geo_label:
            source_df = base_filtered_df[['Year', 'GeoName', open_metric_col]].copy()
            source_df.dropna(subset=[open_metric_col], inplace=True)
            if not source_df.empty and source_df[open_metric_col].sum() > 0:
                source_df['rank'] = source_df.groupby('Year')[open_metric_col].rank(method='first', ascending=False)
                other_geo_text = f"All Other {geo_filter_type}s"
                source_df['GeoContribution'] = np.where(source_df['rank'] <= 3, source_df['GeoName'], other_geo_text)
                plot_df_geos = source_df.groupby(['Year', 'GeoContribution'])[open_metric_col].sum().reset_index()
                plot_df_geos.rename(columns={open_metric_col: "Estimate_value"}, inplace=True)
                if is_currency:
                    plot_df_geos["Estimate_value"] /= 1e6
                unique_contributors = sorted([c for c in plot_df_geos['GeoContribution'].unique() if c != other_geo_text])
                sort_order = unique_contributors + [other_geo_text]
                color_range = get_sector_colors(len(unique_contributors)) + ["#A5AAAF"]
                chart = alt.Chart(plot_df_geos).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label, stack='zero'),
                    color=alt.Color('GeoContribution:N', legend=alt.Legend(title=f"{geo_filter_type} Contribution", orient="right"), sort=sort_order, scale=alt.Scale(domain=sort_order, range=color_range)),
                    tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('GeoContribution:N', title='Contribution'), alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)]
                ).configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(symbolLimit=31)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")
        else:
            bar_df = base_filtered_df.groupby("Year")[open_metric_col].sum().reset_index()
            bar_df.rename(columns={open_metric_col: 'Estimate_value'}, inplace=True)
            if not bar_df.empty:
                if is_currency:
                    bar_df["Estimate_value"] /= 1e6
                sector_color = sector_color_map.get(selected_sector, "#808080")
                chart = alt.Chart(bar_df).mark_bar(color=sector_color).encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label, stack='zero'),
                    tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)]
                ).properties(height=500).configure_axis(labelFontSize=14, titleFontSize=16)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")

    if summary_message:
        st.markdown(f"<p style='font-size: 24px; text-align: center; font-weight: normal;'>{summary_message}</p>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""<style>div[data-testid="stExpander"] summary {font-size: 1.75rem;}</style>""", unsafe_allow_html=True)
    expander_title = "Coastal Geographies in Open ENOW"
    if plot_mode == "State Estimates from Public QCEW Data" and selected_geo != "All Coastal States":
        expander_title = f"{selected_geo} Coastal Geographies in Open ENOW"
    if plot_mode != "County Estimates from Public QCEW Data":
        with st.expander(expander_title):
            st.divider()
            if plot_mode == "Regional Estimates from Public QCEW Data":
                st.write("Open ENOW splits coastal states into 8 regions...")
            elif plot_mode == "State Estimates from Public QCEW Data":
                if selected_geo == "All Coastal States":
                    st.write("Open ENOW includes all 30 U.S. states with a coastline...")
                else:
                    map_filename = f"ENOW state maps/Map_{selected_geo.replace(' ', '_')}.jpg"
                    if os.path.exists(map_filename): st.image(map_filename, use_container_width=True)
    metric_expander_title = f"{selected_display_metric} in Open ENOW"
    with st.expander(metric_expander_title):
        st.divider()
        st.write(METRIC_DESCRIPTIONS.get(selected_display_metric, "No description available."))

# --- START: REVISED 'Error Analysis' MODE ---
elif plot_mode == "Error Analysis":
    active_df = comparison_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `enow_version_comparisons.csv` is in the same directory.")
        st.stop()

    st.title("Error Analysis: Open ENOW vs. Original ENOW")

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plot Configuration")

    # Granularity filters
    selected_agg = st.sidebar.radio("Aggregation Level:", ("Sector", "Industry"), index=0)
    selected_geoscale = st.sidebar.radio("Geographic Scale:", ("State", "County"), index=0)

    # Metric selection for axes
    y_axis_choice = st.sidebar.selectbox(
        "Y-Axis (Error Metric):",
        ("Mean Percent Difference", "Mean Absolute Error", "Root Mean Squared Error"),
        index=0
    )
    x_axis_choice = st.sidebar.selectbox(
        "X-Axis (Economic Metric):",
        ("Employment", "Wages", "GDP"),
        index=0
    )

    # Grouping for trendlines and colors
    grouping_choice = st.sidebar.selectbox(
        "Group By:",
        ("OceanSector", "OceanIndustry", "Year"), # NEW: Added "Year"
        index=0
    )

    # Data filters
    st.sidebar.markdown("---")
    st.sidebar.header("Data Filters")

    min_year, max_year = int(active_df["Year"].min()), int(active_df["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range:", min_year, max_year, (min_year, max_year), 1
    )
    
    # Create a map from state name to state abbreviation for filtering
    state_df = active_df[active_df['GeoScale'] == 'State']
    state_names = ["All Coastal States"] + sorted(state_df["GeoName"].dropna().unique())
    state_abbr_map = {"All Coastal States": "All"}
    state_abbr_map.update(pd.Series(state_df.state.values, index=state_df.GeoName).to_dict())

    selected_state_name = st.sidebar.selectbox("Filter by State:", state_names)
    selected_state_abbr = state_abbr_map[selected_state_name]


    sector_names = ["All Marine Sectors"] + sorted(active_df["OceanSector"].dropna().unique())
    selected_sector_filter = st.sidebar.selectbox("Filter by Sector:", sector_names)

    # --- DATA PROCESSING ---
    
    # Base filtering
    filtered_df = active_df[
        (active_df['aggregation'] == selected_agg) &
        (active_df['GeoScale'] == selected_geoscale) &
        (active_df['Year'] >= year_range[0]) &
        (active_df['Year'] <= year_range[1])
    ].copy()

    # UPDATED: State filtering logic
    if selected_state_name != "All Coastal States":
        # This now correctly filters for counties within the selected state
        filtered_df = filtered_df[filtered_df['state'] == selected_state_abbr]

    if selected_sector_filter != "All Marine Sectors":
        filtered_df = filtered_df[filtered_df['OceanSector'] == selected_sector_filter]

    # Define metric columns based on user choice
    x_metric_map = {
        "Employment": "Employment",
        "Wages": "Wages",
        "GDP": "GDP"
    }
    metric_suffix = x_metric_map[x_axis_choice]
    open_col = f"Open_{metric_suffix}"
    enow_col = f"oldENOW_{metric_suffix}"

    # Calculate metrics for each group
    results = []
    
    # Each point on the plot represents a GeoName within a chosen group
    grouping_cols = [grouping_choice, 'GeoName']
    
    for name, group_df in filtered_df.groupby(grouping_cols):
        
        # Ensure we have data to compare
        group_df = group_df.dropna(subset=[enow_col, open_col])
        if group_df.empty:
            continue

        # Avoid division by zero for percent difference
        valid_enow = group_df[group_df[enow_col] != 0]
        if valid_enow.empty:
             mpd = np.nan
        else:
             pct_diff = 100 * (valid_enow[open_col] - valid_enow[enow_col]) / valid_enow[enow_col]
             mpd = pct_diff.mean()

        mae = mean_absolute_error(group_df[enow_col], group_df[open_col])
        rmse = np.sqrt(mean_squared_error(group_df[enow_col], group_df[open_col]))
        
        x_val = (group_df[enow_col].mean() + group_df[open_col].mean()) / 2

        result_row = {
            grouping_choice: name[0],
            'GeoName': name[1],
            'X_Value': x_val,
            'Mean Percent Difference': mpd,
            'Mean Absolute Error': mae,
            'Root Mean Squared Error': rmse
        }
        results.append(result_row)
        
    if results:
        results_df = pd.DataFrame(results)
        # Assign the chosen Y-axis metric to the 'Y_Value' column for plotting
        results_df['Y_Value'] = results_df[y_axis_choice]
        results_df = results_df.dropna(subset=['Y_Value', 'X_Value'])
        
        # UPDATED: Filter for positive values before plotting on a log scale
        results_df = results_df[results_df['X_Value'] > 0]
        
        # --- PLOTTING ---
        st.subheader(f"Plot of {y_axis_choice} vs. Average {x_axis_choice}")
        
        # If grouping by Year, ensure Year is treated as discrete for coloring
        color_encoding_type = 'O' if grouping_choice == 'Year' else 'N'

        scatter = alt.Chart(results_df).mark_circle(size=100, opacity=0.8).encode(
            # UPDATED: Added scale=alt.Scale(type="log") to the X-axis encoding
            x=alt.X('X_Value:Q', 
                    scale=alt.Scale(type="log"), 
                    title=f'Mean of Original and Open ENOW {x_axis_choice} (Log Scale)'),
            y=alt.Y('Y_Value:Q', title=y_axis_choice),
            color=alt.Color(f'{grouping_choice}:{color_encoding_type}', legend=alt.Legend(title="Group")),
            tooltip=[
                alt.Tooltip(f'{grouping_choice}:{color_encoding_type}', title='Group'),
                alt.Tooltip('GeoName:N', title='Geography'),
                alt.Tooltip('X_Value:Q', title=f'Mean {x_axis_choice}', format=',.0f'),
                alt.Tooltip('Y_Value:Q', title=y_axis_choice, format='.2f')
            ]
        ).interactive()

        trend = scatter.transform_regression(
            'X_Value', 'Y_Value', groupby=[grouping_choice]
        ).mark_line()

        st.altair_chart((scatter + trend), use_container_width=True)
        
        # --- SUMMARY STATISTICS TABLE ---
        st.subheader("Summary Statistics by Group")
        
        # Prepare table for display
        summary_table = results_df[[
            grouping_choice,
            'GeoName',
            'Mean Percent Difference',
            'Mean Absolute Error',
            'Root Mean Squared Error'
        ]].copy()
        
        # Formatting for better readability
        summary_table['Mean Percent Difference'] = summary_table['Mean Percent Difference'].map('{:,.2f}%'.format)
        
        currency_metrics = ['Mean Absolute Error', 'Root Mean Squared Error']
        for col in currency_metrics:
            if x_axis_choice in ["Wages", "GDP"]:
                 summary_table[col] = summary_table[col].apply(lambda x: f"${x:,.0f}")
            else:
                 summary_table[col] = summary_table[col].apply(lambda x: f"{x:,.0f}")

        st.dataframe(summary_table, use_container_width=True)

    else:
        st.warning("No data available for the selected filters. Please broaden your criteria.")

# --- END: REVISED 'Error Analysis' MODE ---

else:  # "Compare to original ENOW"
    active_df = comparison_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `enow_version_comparisons.csv` is in the same directory.")
        st.stop()

    # --- STATE AND COUNTY FILTERS ---
    state_df = active_df[active_df['GeoScale'] == 'State']
    state_names = ["All Coastal States"] + sorted(state_df["GeoName"].dropna().unique())
    
    state_abbr_map = {"All Coastal States": "All"}
    state_abbr_map.update(pd.Series(state_df.state.values, index=state_df.GeoName).to_dict())

    selected_state_name = st.sidebar.selectbox("Select State:", state_names, key='compare_state')
    selected_state_abbr = state_abbr_map[selected_state_name]

    if selected_state_name == "All Coastal States":
        selected_county = "All Coastal Counties"
        st.sidebar.selectbox("Select County:", [selected_county], disabled=True)
    else:
        county_list = ["All Coastal Counties"] + sorted(
            active_df[(active_df['GeoScale'] == 'County') & (active_df['state'] == selected_state_abbr)]['GeoName'].unique()
        )
        def on_county_change():
            st.session_state.compare_industry = "All Marine Industries"
        selected_county = st.sidebar.selectbox("Select County:", county_list, key='compare_county', on_change=on_county_change)

    # --- SECTOR AND INDUSTRY FILTERS ---
    ocean_sectors = ["All Marine Sectors"] + sorted(active_df["OceanSector"].dropna().unique())
    selected_sector = st.sidebar.selectbox("Select Sector:", ocean_sectors, key='compare_sector')

    if selected_sector == "All Marine Sectors":
        selected_industry = "All Marine Industries"
        st.sidebar.selectbox("Select Industry:", [selected_industry], disabled=True)
    else:
        industry_list = ["All Marine Industries"] + sorted(
            active_df[(active_df['aggregation'] == 'Industry') & (active_df['OceanSector'] == selected_sector)]['OceanIndustry'].unique()
        )
        def on_industry_change():
            st.session_state.compare_county = "All Coastal Counties"
        selected_industry = st.sidebar.selectbox("Select Industry:", industry_list, key='compare_industry', on_change=on_industry_change)

    # --- METRIC AND YEAR FILTERS ---
    metric_choices = {k: v for k, v in METRIC_MAP.items() if v != "RealWages"}
    selected_display_metric = st.sidebar.selectbox("Select Metric:", list(metric_choices.keys()))
    selected_metric_internal = METRIC_MAP[selected_display_metric]

    min_year, max_year = int(active_df["Year"].min()), int(active_df["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range:", min_year, max_year, (max(min_year, 2012), min(max_year, 2021)), 1
    )

    # --- DYNAMIC TITLE ---
    geo_title_part = selected_state_name
    if selected_county != "All Coastal Counties":
        geo_title_part = f"{selected_county}, {selected_state_abbr}"

    econ_title_part = selected_sector
    if selected_industry != "All Marine Industries":
        econ_title_part = selected_industry

    if econ_title_part == "All Marine Sectors" and geo_title_part != "All Coastal States":
         econ_title_part = "All Marine Sectors"
    elif econ_title_part == "All Marine Sectors" and geo_title_part == "All Coastal States":
        econ_title_part = "the Marine Economy"

    
    st.title(f"{selected_display_metric} in {econ_title_part} in {geo_title_part}")

    # --- DATA FILTERING LOGIC ---
    base_filtered_df = active_df[
        (active_df["Year"] >= year_range[0]) & (active_df["Year"] <= year_range[1])
    ].copy()

    # GEOGRAPHY FILTERING
    if selected_county != "All Coastal Counties":
        base_filtered_df = base_filtered_df[
            (base_filtered_df['GeoScale'] == 'County') & 
            (base_filtered_df['GeoName'] == selected_county)
        ]
    else:
        base_filtered_df = base_filtered_df[base_filtered_df['GeoScale'] == 'State']
        if selected_state_name != "All Coastal States":
            base_filtered_df = base_filtered_df[base_filtered_df['GeoName'] == selected_state_name]

    # ECONOMIC AGGREGATION FILTERING
    if selected_industry != "All Marine Industries":
        base_filtered_df = base_filtered_df[
            (base_filtered_df['aggregation'] == 'Industry') &
            (base_filtered_df['OceanIndustry'] == selected_industry)
        ]
    else:
        base_filtered_df = base_filtered_df[base_filtered_df['aggregation'] == 'Sector']
        if selected_sector != "All Marine Sectors":
            base_filtered_df = base_filtered_df[base_filtered_df['OceanSector'] == selected_sector]

    # --- PLOTTING AND VISUALIZATION ---
    y_label_map = {"GDP (nominal)": "GDP ($ millions)", "Real GDP": "Real GDP ($ millions, 2017)", "Wages (not inflation-adjusted)": "Wages ($ millions)", "Employment": "Employment (Number of Jobs)", "Establishments": "Establishments (Count)"}
    y_label = y_label_map.get(selected_display_metric, selected_display_metric)
    is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)"]
    tooltip_format = '$,.0f' if is_currency else ',.0f'

    open_metric_col = f"Open_{selected_metric_internal}"
    enow_metric_col = f"oldENOW_{selected_metric_internal}"
    noimpute_metric_col = f"noimpute_{selected_metric_internal}"
    
    # Ensure all required columns exist before proceeding
    required_cols = ["Year", enow_metric_col, open_metric_col, noimpute_metric_col]
    if not all(col in base_filtered_df.columns for col in required_cols):
        st.warning("One or more data columns required for the chart are missing.")
        st.stop()

    plot_df = base_filtered_df[required_cols].copy()
    plot_df.rename(columns={
        enow_metric_col: "Original ENOW",
        open_metric_col: "Open ENOW Estimate",
        noimpute_metric_col: "Public QCEW data, no imputed values"
    }, inplace=True)
    
    if is_currency:
        plot_df.iloc[:, 1:] /= 1e6

    compare_df = plot_df.groupby("Year").sum(min_count=1).reset_index()
    
    # Replace 0 with NaN so that lines with no data don't drop to zero
    cols_to_check = ["Original ENOW", "Open ENOW Estimate", "Public QCEW data, no imputed values"]
    for col in cols_to_check:
        if col in compare_df.columns:
            compare_df[col] = compare_df[col].replace({0: np.nan})

    long_form_df = compare_df.melt('Year', var_name='Source', value_name='Value')
    long_form_df.dropna(subset=['Value'], inplace=True)


    if not long_form_df.empty:
        base = alt.Chart(long_form_df).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Value:Q', title=y_label, scale=alt.Scale(zero=True)),
            color=alt.Color('Source:N',
                            scale=alt.Scale(
                                domain=['Original ENOW', 'Open ENOW Estimate', 'Public QCEW data, no imputed values'],
                                range=['#D55E00', '#0072B2', '#117733']
                            ),
                            legend=alt.Legend(title="Data Source", orient="bottom")),
            tooltip=[
                alt.Tooltip('Year:O', title='Year'), alt.Tooltip('Source:N', title='Source'),
                alt.Tooltip('Value:Q', title=selected_display_metric, format=tooltip_format)
            ]
        )
        line = base.mark_line()
        points = base.mark_point(size=80, filled=True)
        chart = (line + points).properties(height=500).configure_axis(labelFontSize=14, titleFontSize=16).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.divider()
        csv_data_compare = convert_df_to_csv(compare_df)
        file_name_compare = f"Comparison_data.csv"
        st.download_button(
           label="📥 Download Comparison Data as CSV",
           data=csv_data_compare,
           file_name=file_name_compare,
           mime='text/csv',
        )
        
        # --- Summary Statistics ---
        st.subheader("Summary Statistics")

        # For Open ENOW Estimate
        valid_compare_open = compare_df.dropna(subset=["Original ENOW", "Open ENOW Estimate"])
        if not valid_compare_open.empty:
            mae_open = mean_absolute_error(valid_compare_open["Original ENOW"], valid_compare_open["Open ENOW Estimate"])
            rmse_open = np.sqrt(mean_squared_error(valid_compare_open["Original ENOW"], valid_compare_open["Open ENOW Estimate"]))
            diff_open = valid_compare_open["Open ENOW Estimate"] - valid_compare_open["Original ENOW"]
            pct_diff_open = (100 * diff_open / valid_compare_open["Original ENOW"]).replace([np.inf, -np.inf], np.nan)
            
            st.markdown("##### Open ENOW Estimate (with imputed values)")
            summary_text_open = f"""
- **Mean Absolute Error:** {format_value(mae_open, selected_display_metric)}
- **Root Mean Squared Error:** {format_value(rmse_open, selected_display_metric)}
- **Mean Percent Difference:** {pct_diff_open.mean():.2f}%
"""
            st.markdown(summary_text_open)
        else:
            st.markdown("##### Open ENOW Estimate (with imputed values)")
            st.warning("Not enough overlapping data to calculate statistics.")
            
        # For No Imputation Estimate
        valid_compare_noimpute = compare_df.dropna(subset=["Original ENOW", "Public QCEW data, no imputed values"])
        if not valid_compare_noimpute.empty:
            mae_noimpute = mean_absolute_error(valid_compare_noimpute["Original ENOW"], valid_compare_noimpute["Public QCEW data, no imputed values"])
            rmse_noimpute = np.sqrt(mean_squared_error(valid_compare_noimpute["Original ENOW"], valid_compare_noimpute["Public QCEW data, no imputed values"]))
            diff_noimpute = valid_compare_noimpute["Public QCEW data, no imputed values"] - valid_compare_noimpute["Original ENOW"]
            pct_diff_noimpute = (100 * diff_noimpute / valid_compare_noimpute["Original ENOW"]).replace([np.inf, -np.inf], np.nan)
            
            st.markdown("##### Public QCEW Estimate (no imputed values)")
            summary_text_noimpute = f"""
- **Mean Absolute Error:** {format_value(mae_noimpute, selected_display_metric)}
- **Root Mean Squared Error:** {format_value(rmse_noimpute, selected_display_metric)}
- **Mean Percent Difference:** {pct_diff_noimpute.mean():.2f}%
"""
            st.markdown(summary_text_noimpute)
        else:
            st.markdown("##### Public QCEW Estimate (no imputed values)")
            st.warning("Not enough overlapping data to calculate statistics.")

    else:
        st.warning("No overlapping data available to compare for the selected filters.")


