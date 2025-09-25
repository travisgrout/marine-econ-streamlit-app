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

# --- Sidebar ---
st.sidebar.image("open_ENOW_logo.png", width=200)
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

# --- Custom CSS for Sidebar and Tabs ---
st.markdown("""
<style>
    /* Set sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #C6E6F0;
    }

    /* --- Custom Tab Styling --- */
    /* (1) Make tab titles larger */
    button[data-baseweb="tab"] {
        font-size: 20rem; /* Increase font size */
        font-weight: 800;   /* Make font bolder */
        padding: 10px 15px; /* Add some padding */
    }

    /* (2) Change the highlight color of the selected tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #003087 !important; /* NOAA Sky Blue text */
        border-bottom-color: #003087 !important; /* NOAA Sky Blue underline */
    }
</style>
""", unsafe_allow_html=True)

# --- Create Tabs for Main Navigation ---
tab_states, tab_counties, tab_regions, tab_compare, tab_error = st.tabs([
    "State Estimates",
    "County Estimates",
    "Regional Estimates",
    "Compare to ENOW",
    "Error Analysis"
])

# --- State Estimates Tab ---
with tab_states:
    st.sidebar.header("State Display Filters")
    
    # Data loading for this tab
    active_df = open_enow_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `openENOWinput.csv` is in the same directory as the app.")
        st.stop()
    
    # Filter data for this specific view
    states_df = active_df[(active_df['GeoScale'] == 'State') & (active_df['aggregation'] == 'Sector')].copy()

    # --- Sidebar Filters for States ---
    geo_label = "Select State:"
    all_geo_label = "All Coastal States"
    geo_names = states_df["GeoName"].dropna().unique()
    unique_geos = [all_geo_label] + sorted(geo_names)
    selected_geo = st.sidebar.selectbox(geo_label, unique_geos, key="state_geo")

    ocean_sectors = states_df["OceanSector"].dropna().unique()
    unique_sectors = ["All Marine Sectors"] + sorted(ocean_sectors)
    selected_sector = st.sidebar.selectbox("Select Sector:", unique_sectors, key="state_sector")

    metric_choices = list(METRIC_MAP.keys())
    selected_display_metric = st.sidebar.selectbox("Select Metric:", metric_choices, key="state_metric")
    selected_metric_internal = METRIC_MAP[selected_display_metric]

    min_year, max_year = int(states_df["Year"].min()), int(states_df["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, max_year - 9), max_year),
        step=1,
        key="state_year"
    )

    # --- Main Page Content for States ---
    title_sector_part = "All Marine Sectors" if selected_sector == "All Marine Sectors" else f"{selected_sector} Sector"
    plot_title = f"{selected_display_metric}: {title_sector_part} in {selected_geo}"
    st.title(plot_title)

    # Filter data based on selections
    base_filtered_df = states_df[
        (states_df["Year"] >= year_range[0]) &
        (states_df["Year"] <= year_range[1])
    ]
    if selected_geo != all_geo_label:
        base_filtered_df = base_filtered_df[base_filtered_df["GeoName"] == selected_geo]
    
    if selected_sector != "All Marine Sectors":
        base_filtered_df = base_filtered_df[base_filtered_df["OceanSector"] == selected_sector]

    # ... (Plotting logic for states, similar to your original code) ...
    # This section contains the chart rendering logic for the State Estimates tab.
    y_label_map = {
        "GDP (nominal)": "GDP ($ millions)", "Real GDP": "Real GDP ($ millions, 2017)",
        "Wages (not inflation-adjusted)": "Wages ($ millions)", "Real Wages": "Real Wages ($ millions, 2024)",
        "Employment": "Employment (Number of Jobs)", "Establishments": "Establishments (Count)"
    }
    y_label = y_label_map.get(selected_display_metric, selected_display_metric)
    is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)", "Real Wages"]
    tooltip_format = '$,.0f' if is_currency else ',.0f'
    open_metric_col = f"Open_{selected_metric_internal}"

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
                color=alt.Color('OceanSector:N', scale=alt.Scale(domain=sorted(ocean_sectors), range=get_sector_colors(len(ocean_sectors)))),
                tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('OceanSector:N', title='Sector'), alt.Tooltip('sum(Estimate_value):Q', title=selected_display_metric, format=tooltip_format)]
            ).configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(symbolLimit=len(ocean_sectors))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    else: # A single sector is selected
        if selected_geo == all_geo_label: # All states, single sector
             source_df = base_filtered_df[['Year', 'GeoName', open_metric_col]].copy()
             source_df.dropna(subset=[open_metric_col], inplace=True)
             if not source_df.empty and source_df[open_metric_col].sum() > 0:
                source_df['rank'] = source_df.groupby('Year')[open_metric_col].rank(method='first', ascending=False)
                other_geo_text = "All Other States"
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
                    color=alt.Color('GeoContribution:N', legend=alt.Legend(title="State Contribution", orient="right"), sort=sort_order, scale=alt.Scale(domain=sort_order, range=color_range)),
                    tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('GeoContribution:N', title='Contribution'), alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)]
                ).configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(symbolLimit=31)
                st.altair_chart(chart, use_container_width=True)
             else:
                st.warning("No data available for the selected filters.")
        else: # Single state, single sector
            bar_df = base_filtered_df.groupby("Year")[open_metric_col].sum().reset_index()
            bar_df.rename(columns={open_metric_col: 'Estimate_value'}, inplace=True)
            if not bar_df.empty:
                if is_currency:
                    bar_df["Estimate_value"] /= 1e6
                sector_color_map = dict(zip(sorted(ocean_sectors), get_sector_colors(len(ocean_sectors))))
                sector_color = sector_color_map.get(selected_sector, "#808080")
                chart = alt.Chart(bar_df).mark_bar(color=sector_color).encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label, stack='zero'),
                    tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)]
                ).properties(height=500).configure_axis(labelFontSize=14, titleFontSize=16)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")

# --- County Estimates Tab ---
with tab_counties:
    st.sidebar.header("County Display Filters")
    # This section would contain the logic and filters for the County display.
    # To keep this example concise, it's left as a placeholder.
    st.info("The County Estimates display is under construction.")


# --- Regional Estimates Tab ---
with tab_regions:
    st.sidebar.header("Regional Display Filters")
    # This section would contain the logic and filters for the Regional display.
    st.info("The Regional Estimates display is under construction.")


# --- Compare to ENOW Tab ---
with tab_compare:
    st.sidebar.header("Comparison Filters")
    
    # Data loading for this tab
    active_df = comparison_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `enow_version_comparisons.csv` is in the same directory.")
        st.stop()

    # --- Sidebar Filters for Comparison ---
    state_df = active_df[active_df['GeoScale'] == 'State']
    state_names = ["All Coastal States"] + sorted(state_df["GeoName"].dropna().unique())
    state_abbr_map = {"All Coastal States": "All"}
    state_abbr_map.update(pd.Series(state_df.state.values, index=state_df.GeoName).to_dict())

    selected_state_name = st.sidebar.selectbox("Select State:", state_names, key='compare_state')
    selected_state_abbr = state_abbr_map[selected_state_name]

    if selected_state_name == "All Coastal States":
        selected_county = "All Coastal Counties"
        st.sidebar.selectbox("Select County:", [selected_county], disabled=True, key='compare_county_disabled')
    else:
        county_list = ["All Coastal Counties"] + sorted(
            active_df[(active_df['GeoScale'] == 'County') & (active_df['state'] == selected_state_abbr)]['GeoName'].unique()
        )
        selected_county = st.sidebar.selectbox("Select County:", county_list, key='compare_county')

    # ... (Rest of the compare logic, similar to your original code) ...
    st.title(f"Comparing ENOW Versions for {selected_state_name}")
    st.info("The Compare to ENOW display is under construction.")


# --- Error Analysis Tab ---
with tab_error:
    st.sidebar.header("Error Analysis Filters")

    # Data loading for this tab
    active_df = comparison_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `enow_version_comparisons.csv` is in the same directory.")
        st.stop()
        
    # --- Sidebar Filters for Error Analysis ---
    selected_agg = st.sidebar.radio("Aggregation Level:", ("Sector", "Industry"), index=0, key="error_agg")
    # ... (Rest of the error analysis filters and logic) ...
    st.title("Error Analysis: Open ENOW vs. Original ENOW")
    st.info("The Error Analysis display is under construction.")


