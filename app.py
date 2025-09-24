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
            "oldENOW_RealGDP": "oldENOW_RealGDP"
        }
        df.rename(columns=rename_dict, inplace=True)

        # CONVERT METRIC COLUMNS TO NUMERIC
        metric_cols_to_convert = [
            'Open_Establishments', 'Open_Employment', 'Open_Wages', 'Open_GDP', 'Open_RealGDP',
            'oldENOW_Establishments', 'oldENOW_Employment', 'oldENOW_Wages', 'oldENOW_GDP', 'oldENOW_RealGDP'
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
    if pd.isna(x): return "N/A"
    if metric in ["Wages (not inflation-adjusted)", "Real Wages", "GDP (nominal)", "Real GDP"]:
        return f"${x:,.0f}"
    return f"{x:,.0f}"

def get_sector_colors(n):
    base_colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
    return base_colors[:n] if n <= len(base_colors) else alt.themes.get().schemes['tableau20'][:n]

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Data Dictionaries for Expanders (Unchanged) ---
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
    "Employment": "Employment estimates in Open ENOW are based on the sum of annual average employment reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates employment in the Louisiana Marine Transportation Sector based on reported annual average employment in four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline. To address gaps in public county-level QCEW data, Open ENOW imputes missing values based on data from other years or broader economic sectors.",    "Wages (not inflation-adjusted)": "Open ENOW estimates wages paid to workers based on the sum of total annual wages paid reported in the Quarterly Census of Employment and Wages (QCEW)...",
    "Wages (not inflation-adjusted)": "Open ENOW estimates wages paid to workers based on the sum of total annual wages paid reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates wages in the Louisiana Marine Transportation Sector based on reported annual wages paid in four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline. To address gaps in public county-level QCEW data, Open ENOW imputes missing values based on data from other years or broader economic sectors.",
    "Real Wages": "Open ENOW reports inflation-adjusted real wages in 2024 dollars. To estimate real wages, Open ENOW adjusts its nominal wage estimates for changes in the consumer price index (CPI).",
    "Establishments": "Open ENOW estimates the number of employers in a given marine sector based on the sum of establishments reported in the Quarterly Census of Employment and Wages (QCEW) for a given set of NAICS codes and set of coastal counties. For example, Open ENOW estimates the number of establishments in the Louisiana Marine Transportation Sector based on QCEW data for four NAICS codes (334511, 48311, 4883, and 4931) in 18 Louisiana parishes on or near the coastline.",
    "GDP (nominal)": "Open ENOW estimates a sector's contribution to GDP based on the average ratio of wages paid to GDP reported for the relevant industry in the Bureau of Economic Analysis (BEA) GDP by industry in current dollars (SAGDP2) table.",
    "Real GDP": "Real GDP is reported in 2017 dollars. Open ENOW estimates a sector's contribution to Real GDP based on the average ratio of wages paid to GDP reported for the relevant industry in the Bureau of Economic Analysis (BEA) Real GDP by industry in chained dollars (SAGDP9) table."
}

# --- Main Application ---
METRIC_MAP = {
    "Employment": "Employment", "Wages (not inflation-adjusted)": "Wages",
    "Real Wages": "RealWages", "Establishments": "Establishments",
    "GDP (nominal)": "GDP", "Real GDP": "RealGDP"
}

st.sidebar.image("open_ENOW_logo.png", width=200)

# --- Pop-up Window ---
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
button_map = {"States": "State Estimates from Public QCEW Data", "Counties": "County Estimates from Public QCEW Data", "Regions": "Regional Estimates from Public QCEW Data", "Compare": "Compare to original ENOW"}
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


def update_mode(mode_label):
    st.session_state.plot_mode = button_map[mode_label]

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


plot_mode = st.session_state.plot_mode

# --- Select Active DataFrame and Set Filters based on Mode ---
estimate_modes = ["State Estimates from Public QCEW Data", "County Estimates from Public QCEW Data", "Regional Estimates from Public QCEW Data"]

if plot_mode in estimate_modes:
    # ... (Logic for estimate modes omitted for brevity) ...
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
                selected_county = st.sidebar.selectbox(county_label, county_names)
            else:
                selected_county = st.sidebar.selectbox(county_label, [])

            # Set selected_geo to the county for unified logic later
            selected_geo = selected_county
        else:
            st.warning("The 'stateName' column is not available in the data for county estimates.")
            selected_state = None
            selected_county = None


    else: # Regional Estimates from Public QCEW Data
        # Filter for Region-level data aggregated by Sector
        active_df = active_df[(active_df['GeoScale'] == 'Region') & (active_df['aggregation'] == 'Sector')].copy()
        geo_label = "Select Region:"
        all_geo_label = "All Regions"
        geo_filter_type = 'Region'
        geo_names = active_df["GeoName"].dropna().unique()
        unique_geos = [all_geo_label] + sorted(geo_names)
        selected_geo = st.sidebar.selectbox(geo_label, unique_geos)

else:  # "Compare to original ENOW"
    active_df = comparison_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `enow_version_comparisons.csv` is in the same directory.")
        st.stop()

    # --- STATE AND COUNTY FILTERS ---
    state_df = active_df[active_df['GeoScale'] == 'State']
    state_names = ["All Coastal States"] + sorted(state_df["GeoName"].dropna().unique())
    
    # Map state names to abbreviations for filtering
    state_abbr_map = {"All Coastal States": "All"}
    state_abbr_map.update(pd.Series(state_df.state.values, index=state_df.GeoName).to_dict())

    selected_state_name = st.sidebar.selectbox("Select State:", state_names, key='compare_state')
    selected_state_abbr = state_abbr_map[selected_state_name]

    # County Selector
    if selected_state_name == "All Coastal States":
        selected_county = "All Coastal Counties"
        st.sidebar.selectbox("Select County:", [selected_county], disabled=True)
    else:
        county_list = ["All Coastal Counties"] + sorted(
            active_df[(active_df['GeoScale'] == 'County') & (active_df['state'] == selected_state_abbr)]['GeoName'].unique()
        )
        # Reset industry selection if a county is chosen
        def on_county_change():
            st.session_state.compare_industry = "All Marine Industries"
        
        selected_county = st.sidebar.selectbox("Select County:", county_list, key='compare_county', on_change=on_county_change)


    # --- SECTOR AND INDUSTRY FILTERS ---
    ocean_sectors = ["All Marine Sectors"] + sorted(active_df["OceanSector"].dropna().unique())
    selected_sector = st.sidebar.selectbox("Select Sector:", ocean_sectors, key='compare_sector')

    # Industry Selector
    if selected_sector == "All Marine Sectors":
        selected_industry = "All Marine Industries"
        st.sidebar.selectbox("Select Industry:", [selected_industry], disabled=True)
    else:
        industry_list = ["All Marine Industries"] + sorted(
            active_df[(active_df['aggregation'] == 'Industry') & (active_df['OceanSector'] == selected_sector)]['OceanIndustry'].unique()
        )
        # Reset county selection if an industry is chosen
        def on_industry_change():
            st.session_state.compare_county = "All Coastal Counties"
            # Note: A full reset of the state might be needed if counties should clear
            # For now, just resetting the county dropdown content is sufficient
        
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

    if econ_title_part == "All Marine Sectors":
        econ_title_part = "All Marine Sectors"
    
    st.title(f"{selected_display_metric}: {econ_title_part} in {geo_title_part}")

    # --- DATA FILTERING LOGIC ---
    base_filtered_df = active_df[
        (active_df["Year"] >= year_range[0]) & (active_df["Year"] <= year_range[1])
    ]

    # Filter by geography
    if selected_county != "All Coastal Counties":
        base_filtered_df = base_filtered_df[
            (base_filtered_df['GeoScale'] == 'County') & 
            (base_filtered_df['GeoName'] == selected_county)
        ]
    elif selected_state_name != "All Coastal States":
        if selected_industry != "All Marine Industries":
             base_filtered_df = base_filtered_df[base_filtered_df['state'] == selected_state_abbr]
        else:
            base_filtered_df = base_filtered_df[
                (base_filtered_df['GeoScale'] == 'State') & 
                (base_filtered_df['GeoName'] == selected_state_name)
            ]

    # Filter by economic sector/industry
    if selected_industry != "All Marine Industries":
        base_filtered_df = base_filtered_df[
            (base_filtered_df['aggregation'] == 'Industry') &
            (base_filtered_df['OceanIndustry'] == selected_industry)
        ]
    elif selected_sector != "All Marine Sectors":
        base_filtered_df = base_filtered_df[base_filtered_df['OceanSector'] == selected_sector]


    # --- PLOTTING AND VISUALIZATION ---
    y_label_map = {"GDP (nominal)": "GDP ($ millions)", "Real GDP": "Real GDP ($ millions, 2017)", "Wages (not inflation-adjusted)": "Wages ($ millions)", "Employment": "Employment (Number of Jobs)", "Establishments": "Establishments (Count)"}
    y_label = y_label_map.get(selected_display_metric, selected_display_metric)
    is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)"]
    tooltip_format = '$,.0f' if is_currency else ',.0f'

    open_metric_col = f"Open_{selected_metric_internal}"
    enow_metric_col = f"oldENOW_{selected_metric_internal}"

    plot_df = base_filtered_df[["Year", enow_metric_col, open_metric_col]].copy()
    plot_df.rename(columns={
        enow_metric_col: "Original ENOW",
        open_metric_col: "Open ENOW Estimate"
    }, inplace=True)
    
    if is_currency:
        plot_df[["Original ENOW", "Open ENOW Estimate"]] /= 1e6

    compare_df = plot_df.groupby("Year")[["Original ENOW", "Open ENOW Estimate"]].sum(min_count=1).reset_index()
    compare_df.dropna(subset=["Original ENOW", "Open ENOW Estimate"], how='all', inplace=True)
    long_form_df = compare_df.melt('Year', var_name='Source', value_name='Value')

    if not long_form_df.empty and long_form_df['Value'].notna().any():
        base = alt.Chart(long_form_df).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Value:Q', title=y_label, scale=alt.Scale(zero=True)),
            color=alt.Color('Source:N',
                            scale=alt.Scale(domain=['Original ENOW', 'Open ENOW Estimate'], range=['#D55E00', '#0072B2']),
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

        # --- Data Download Section ---
        st.divider()
        csv_data_compare = convert_df_to_csv(compare_df)
        file_name_compare = f"Comparison_{selected_geo.replace(' ', '_')}_{selected_sector.replace(' ', '_')}_{selected_display_metric.replace(' ', '_')}_{year_range[0]}_to_{year_range[1]}.csv"

        st.download_button(
           label="📥 Download Comparison Data as CSV",
           data=csv_data_compare,
           file_name=file_name_compare,
           mime='text/csv',
        )

        # --- Summary Statistics ---
        valid_compare_df = compare_df.dropna(subset=["Original ENOW", "Open ENOW Estimate"])
        if not valid_compare_df.empty:
            mae = mean_absolute_error(valid_compare_df["Original ENOW"], valid_compare_df["Open ENOW Estimate"])
            rmse = np.sqrt(mean_squared_error(valid_compare_df["Original ENOW"], valid_compare_df["Open ENOW Estimate"]))
            diff = valid_compare_df["Open ENOW Estimate"] - valid_compare_df["Original ENOW"]
            # Avoid division by zero for percent difference
            pct_diff = (100 * diff / valid_compare_df["Original ENOW"]).replace([np.inf, -np.inf], np.nan)

            summary_text = f"""
Mean Absolute Error: {format_value(mae, selected_display_metric)}
Root Mean Squared Error: {format_value(rmse, selected_display_metric)}
Mean Percent Difference: {pct_diff.mean():.2f}%
"""
        else:
            summary_text = "Not enough data points with both ENOW and Estimate values to calculate summary statistics."


        st.subheader("Summary Statistics")
        st.code(summary_text, language='text')

    else:
        st.warning("No overlapping data available to compare for the selected filters.")
