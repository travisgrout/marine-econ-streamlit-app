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
def load_dorado_data():
    """
    Loads, cleans, and prepares the original combined dataset (DORADO).
    This data is used for the "Compare to original ENOW" mode, as it contains both
    official ENOW data and the older estimates.
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
        # This error will be handled in the main app logic if this mode is selected.
        return None

@st.cache_data
def load_open_enow_data():
    """
    Loads, cleans, and prepares the new Open ENOW dataset from openENOWinput.csv.
    This data is used for the "State Estimates" and "Regional Estimates" modes.
    """
    try:
        # NOTE: The file name was changed from "openENOWinput.csv" to match the rest of the script.
        # If your file is named differently, please adjust it here.
        df = pd.read_csv("openENOWinput.csv")
        
        # Rename columns to match the app's expected internal names
        rename_dict = {
            "geoType": "GeoScale",
            "geoName": "GeoName",
            "state": "StateAbbrv",
            "year": "Year",
            "enowSector": "OceanSector",
            "establishments": "NQ_Establishments",
            "employment": "NQ_Employment",
            "wages": "NQ_Wages",
            "real_wages": "NQ_RealWages",
            "gdp": "NQ_GDP",
            "rgdp": "NQ_RealGDP"
        }
        df.rename(columns=rename_dict, inplace=True)
        
        # Explicitly convert metric columns to numeric types
        metric_cols_to_convert = [
            'NQ_Establishments', 'NQ_Employment', 'NQ_Wages', 'NQ_RealWages', 'NQ_GDP', 'NQ_RealGDP'
        ]
        
        for col in metric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except FileNotFoundError:
        # This error will be handled in the main app logic if this mode is selected.
        return None

# Load both potential data sources
dorado_data = load_dorado_data()
open_enow_data = load_open_enow_data()

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

# --- Function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    """
    Converts a Pandas DataFrame to a CSV string, encoded in UTF-8.
    This function is cached to improve performance.
    """
    return df.to_csv(index=False).encode('utf-8')


# --- Data Dictionaries for Expanders (Unchanged) ---
SECTOR_DESCRIPTIONS = {
    # This dictionary remains unchanged...
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
METRIC_DESCRIPTIONS = {
    # This dictionary remains unchanged...
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

# --- START: CODE FOR POP-UP WINDOW (Unchanged) ---
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

st.sidebar.header("Filters")
# --- START: MODIFICATION FOR NEW DISPLAY MODE ---
plot_mode = st.sidebar.radio(
    "Display Mode:",
    (
        "State Estimates from Public QCEW Data",
        "Regional Estimates from Public QCEW Data", # New mode added here
        "Compare to original ENOW"
    ),
    index=0
)
# --- END: MODIFICATION FOR NEW DISPLAY MODE ---

# --- Select Active DataFrame and Set Filters based on Mode ---
estimate_modes = ["State Estimates from Public QCEW Data", "Regional Estimates from Public QCEW Data"]

if plot_mode in estimate_modes:
    active_df = open_enow_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `openENOWinput.csv` is in the same directory as the app.")
        st.stop()
    
    # --- START: MODIFICATION FOR DYNAMIC GEOGRAPHY FILTER ---
    if plot_mode == "State Estimates from Public QCEW Data":
        geo_label = "Select State:"
        geo_filter_type = "State"
        all_geo_label = "All Coastal States"
        geo_df = active_df[active_df['GeoScale'] == geo_filter_type]
    else: # Regional Estimates from Public QCEW Data
        geo_label = "Select Region:"
        geo_filter_type = "Region"
        all_geo_label = "All Regions"
        geo_df = active_df[active_df['GeoScale'] == geo_filter_type]
        
    geo_names = geo_df["GeoName"].dropna().unique()
    unique_geos = [all_geo_label] + sorted(geo_names)
    # --- END: MODIFICATION FOR DYNAMIC GEOGRAPHY FILTER ---

else:  # "Compare to original ENOW"
    active_df = dorado_data
    if active_df is None:
        st.error("❌ **Data not found!** Please make sure `DORADO_combined_sectors.csv` is in the same directory as the app.")
        st.stop()
    
    geo_label = "Select State:"
    all_geo_label = "All Coastal States"
    geo_names = active_df["GeoName"].dropna().unique()
    unique_geos = [all_geo_label] + sorted(geo_names)


# --- Dynamically create sidebar filters from the active dataframe ---
ocean_sectors = active_df["OceanSector"].dropna().unique()
unique_sectors = ["All Marine Sectors"] + sorted(ocean_sectors)

sorted_sector_names = sorted(ocean_sectors)
colors_list = get_sector_colors(len(sorted_sector_names))
sector_color_map = dict(zip(sorted_sector_names, colors_list))

if plot_mode in estimate_modes:
    metric_choices = list(METRIC_MAP.keys())
else:
    # "Real Wages" is not available in the original ENOW data for comparison
    compare_metrics = {k: v for k, v in METRIC_MAP.items() if v != "RealWages"}
    metric_choices = list(compare_metrics.keys())

# --- MODIFICATION: Use dynamic labels for geography selection ---
selected_geo = st.sidebar.selectbox(geo_label, unique_geos)
selected_sector = st.sidebar.selectbox("Select Marine Sector:", unique_sectors)
selected_display_metric = st.sidebar.selectbox("Select Metric:", metric_choices)
selected_metric_internal = METRIC_MAP[selected_display_metric]

min_year, max_year = int(active_df["Year"].min()), int(active_df["Year"].max())

# Set the default year range based on the selected plot mode.
if plot_mode in estimate_modes:
    default_end_year = max_year
    default_start_year = max(min_year, max_year - 9) 
    default_range = (default_start_year, default_end_year)
else:  # "Compare to original ENOW" - This range is fixed by the ENOW dataset's availability
    default_start_year = max(min_year, 2012)
    default_end_year = min(max_year, 2021)
    default_range = (default_start_year, default_end_year)
    
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=default_range,
    step=1
)

# --- Dynamic Title ---
if selected_sector == "All Marine Sectors":
    title_sector_part = selected_sector
else:
    title_sector_part = f"{selected_sector} Sector"

plot_title = f"{selected_display_metric}: {title_sector_part} in {selected_geo}"
st.title(plot_title)

# --- START: CODE BLOCK FOR GDP BANNER ---
is_gdp_metric = selected_display_metric in ["GDP (nominal)", "Real GDP"]

if is_gdp_metric:
    gdp_col_to_check = f"NQ_{selected_metric_internal}"
    # Check if all GDP values for the max_year are null/NaN in the active dataframe
    gdp_is_missing_for_max_year = active_df.loc[active_df['Year'] == max_year, gdp_col_to_check].isnull().all()
    
    if gdp_is_missing_for_max_year:
        st.info(f"💡 GDP estimates are not yet available for {max_year}.")
# --- END: GDP BANNER CODE BLOCK ---

# --- Base Data Filtering ---
# First, filter by year range
base_filtered_df = active_df[
    (active_df["Year"] >= year_range[0]) &
    (active_df["Year"] <= year_range[1])
]

# --- MODIFICATION: Handle geography filtering with new dynamic logic ---
if selected_geo == all_geo_label:
    # For estimates modes, ensure we only sum the correct geography type to avoid double counting
    if plot_mode in estimate_modes:
        base_filtered_df = base_filtered_df[base_filtered_df['GeoScale'] == geo_filter_type]
    # For "Compare" mode, no change is needed; it includes everything by default
else:
    # Filter for the specific geography (state or region) selected by the user
    base_filtered_df = base_filtered_df[base_filtered_df["GeoName"] == selected_geo]

# Then, filter by sector
if selected_sector != "All Marine Sectors":
    base_filtered_df = base_filtered_df[base_filtered_df["OceanSector"] == selected_sector]


# --- Plotting and Visualization ---
y_label_map = {
    "GDP (nominal)": "GDP ($ millions)",
    "Real GDP": "Real GDP ($ millions, 2017)",
    "Wages (not inflation-adjusted)": "Wages ($ millions)",
    "Real Wages": "Real Wages ($ millions, 2024)",
    "Employment": "Employment (Number of Jobs)",
    "Establishments": "Establishments (Count)"
}
y_label = y_label_map.get(selected_display_metric, selected_display_metric)

is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)", "Real Wages"]
tooltip_format = '$,.0f' if is_currency else ',.0f'

# --- MODIFICATION: Combined logic for both State and Regional Estimate Modes ---
if plot_mode in estimate_modes:
    
    nq_metric_col = f"NQ_{selected_metric_internal}"

    # --- START: LATEST YEAR SUMMARY CALCULATION ---
    summary_message = None
    latest_year = year_range[1]
    
    latest_year_data = base_filtered_df[base_filtered_df['Year'] == latest_year]

    if not latest_year_data.empty:
        latest_value = latest_year_data[nq_metric_col].sum()

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
    # --- END: LATEST YEAR SUMMARY CALCULATION ---
    
    # CASE 1: ALL MARINE SECTORS (STACKED BY SECTOR)
    if selected_sector == "All Marine Sectors":
        plot_df = base_filtered_df[["Year", "OceanSector", nq_metric_col]].copy()
        plot_df.rename(columns={nq_metric_col: "Estimate_value"}, inplace=True)
        plot_df.dropna(subset=["Estimate_value"], inplace=True)
        
        if is_currency:
            plot_df["Estimate_value"] /= 1e6
        
        if not plot_df.empty:
            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('sum(Estimate_value):Q', title=y_label),
                color=alt.Color('OceanSector:N', 
                                scale=alt.Scale(domain=sorted_sector_names, range=colors_list),
                                legend=alt.Legend(title="Sectors")),
                tooltip=[
                    alt.Tooltip('Year:O', title='Year'),
                    alt.Tooltip('OceanSector:N', title='Sector'),
                    alt.Tooltip('sum(Estimate_value):Q', title=selected_display_metric, format=tooltip_format)
                ]
            ).configure_axis(
                labelFontSize=14,
                titleFontSize=16
            ).configure_legend(
                symbolLimit=len(sorted_sector_names)
            )
            st.altair_chart(chart, use_container_width=True)

            # --- Data Download for All Sectors Chart ---
            download_df = plot_df.groupby(['Year', 'OceanSector'])['Estimate_value'].sum().reset_index()
            download_df.rename(columns={'Estimate_value': y_label}, inplace=True)
            csv_data = convert_df_to_csv(download_df)
            file_name = f"{selected_geo.replace(' ', '_')}_{selected_sector.replace(' ', '_')}_{selected_display_metric.replace(' ', '_')}_{year_range[0]}_to_{year_range[1]}.csv"
            st.download_button(
               label="📥 Download Selected Data as CSV",
               data=csv_data,
               file_name=file_name,
               mime='text/csv',
            )
        else:
            st.warning("No data available for the selected filters.")

    # CASE 2 & 3: A SINGLE MARINE SECTOR IS SELECTED
    else:
        # CASE 2: SINGLE SECTOR, ALL GEOGRAPHIES (STACKED BY STATE/REGION)
        if selected_geo == all_geo_label:
            source_df = base_filtered_df[['Year', 'GeoName', nq_metric_col]].copy()
            source_df.dropna(subset=[nq_metric_col], inplace=True)

            if not source_df.empty and source_df[nq_metric_col].sum() > 0:
                source_df['rank'] = source_df.groupby('Year')[nq_metric_col].rank(method='first', ascending=False)
                
                # Dynamic placeholder text for the legend
                other_geo_text = f"All Other {geo_filter_type}s"
                source_df['GeoContribution'] = np.where(source_df['rank'] <= 3, source_df['GeoName'], other_geo_text)

                plot_df_geos = source_df.groupby(['Year', 'GeoContribution'])[nq_metric_col].sum().reset_index()
                plot_df_geos.rename(columns={nq_metric_col: "Estimate_value"}, inplace=True)
                
                if is_currency:
                    plot_df_geos["Estimate_value"] /= 1e6
                                        
                geo_color_palette = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
                other_geos_color = "#A5AAAF"

                unique_contributors = sorted([c for c in plot_df_geos['GeoContribution'].unique() if c != other_geo_text])
                
                sort_order = unique_contributors + [other_geo_text]
                
                color_domain = sort_order
                color_range = geo_color_palette[:len(unique_contributors)] + [other_geos_color]

                chart = alt.Chart(plot_df_geos).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label, stack='zero'),
                    color=alt.Color('GeoContribution:N',
                                    legend=alt.Legend(title=f"{geo_filter_type} Contribution", orient="right"),
                                    sort=sort_order, 
                                    scale=alt.Scale(domain=color_domain, range=color_range)
                                    ),
                    tooltip=[
                        alt.Tooltip('Year:O', title='Year'),
                        alt.Tooltip('GeoContribution:N', title='Contribution'),
                        alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)
                    ]
                ).configure_axis(
                    labelFontSize=14,
                    titleFontSize=16
                ).configure_legend(
                    symbolLimit=31 
                )
                
                st.altair_chart(chart, use_container_width=True)

                # --- Data Download for Geo Contribution Chart ---
                download_df = plot_df_geos.rename(columns={'Estimate_value': y_label, 'GeoContribution': f'{geo_filter_type} Contribution'})
                csv_data = convert_df_to_csv(download_df)
                file_name = f"By_{geo_filter_type}_{selected_sector.replace(' ', '_')}_{selected_display_metric.replace(' ', '_')}_{year_range[0]}_to_{year_range[1]}.csv"
                st.download_button(
                    label=f"📥 Download {geo_filter_type} Contribution Data as CSV",
                    data=csv_data,
                    file_name=file_name,
                    mime='text/csv',
                )
            else:
                st.warning("No data available for the selected filters.")

        # CASE 3: SINGLE SECTOR, SINGLE GEOGRAPHY (SIMPLE BAR CHART)
        else: 
            bar_df = base_filtered_df.groupby("Year")[nq_metric_col].sum().reset_index()
            bar_df.rename(columns={nq_metric_col: 'Estimate_value'}, inplace=True)
            
            if not bar_df.empty:
                if is_currency:
                    bar_df["Estimate_value"] /= 1e6
                
                sector_color = sector_color_map.get(selected_sector, "#808080")
                chart = alt.Chart(bar_df).mark_bar(color=sector_color).encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label),
                    tooltip=[
                        alt.Tooltip('Year:O', title='Year'),
                        alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)
                    ]
                ).properties(
                    height=500
                ).configure_axis(
                    labelFontSize=14,
                    titleFontSize=16
                )
                st.altair_chart(chart, use_container_width=True)

                # --- Data Download for Single Sector/Geo Chart ---
                download_df = bar_df
                download_df.insert(1, 'OceanSector', selected_sector) 
                download_df.rename(columns={'Estimate_value': y_label}, inplace=True)
                csv_data = convert_df_to_csv(download_df)
                file_name = f"{selected_geo.replace(' ', '_')}_{selected_sector.replace(' ', '_')}_{selected_display_metric.replace(' ', '_')}_{year_range[0]}_to_{year_range[1]}.csv"
                st.download_button(
                   label="📥 Download Selected Data as CSV",
                   data=csv_data,
                   file_name=file_name,
                   mime='text/csv',
                )
            else:
                st.warning("No data available for the selected filters.")
    
    # --- START: DISPLAY LATEST YEAR SUMMARY TEXT ---
    if summary_message:
        st.markdown(
            f"<p style='font-size: 24px; text-align: center; font-weight: normal;'>{summary_message}</p>",
            unsafe_allow_html=True
        )
    # --- END: DISPLAY LATEST YEAR SUMMARY TEXT ---

    # --- Expandable sections appear below all charts in these modes ---
    st.divider()

    # --- START: MODIFICATION FOR COASTAL GEOGRAPHIES EXPANDER ---
    # This section now conditionally displays content based on the plot mode.
    st.markdown("""
        <style>
        div[data-testid="stExpander"] summary {
            font-size: 1.75rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
    expander_title = "Coastal Geographies in Open ENOW"
    if plot_mode == "State Estimates from Public QCEW Data" and selected_geo != "All Coastal States":
        expander_title = f"{selected_geo} Coastal Geographies in Open ENOW"

    with st.expander(expander_title):
        st.divider()
        
        # --- NEW: Logic for Regional Mode ---
        if plot_mode == "Regional Estimates from Public QCEW Data":
            st.write("This is a placeholder for regional geography information, which will be added in a future update.")
        
        # --- ORIGINAL: Logic for State Mode ---
        elif plot_mode == "State Estimates from Public QCEW Data":
            if selected_geo == "All Coastal States":
                st.write("""Open ENOW includes all 30 U.S. states with a coastline on the ocean or the Great Lakes. Within those states, Open ENOW aggregates data for all counties on or near the coastline. Open ENOW relies on state-level instead of county-level data for three states–Delaware, Hawaii, and Rhode Island–where all counties are on the coastline. Select a state from the drop-down menu to see the portion of that state considered "coastal" for the purpose of Open ENOW estimates.""")
            else:
                st.markdown("""
                    <style>
                    div[data-testid="stHorizontalBlock"] {
                        align-items: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                map_col, legend_col = st.columns([2, 1])

                with map_col:
                    map_filename = f"ENOW state maps/Map_{selected_geo.replace(' ', '_')}.jpg"

                    if os.path.exists(map_filename):
                        with st.container(border=True):
                            st.image(map_filename, use_container_width=True)
                    else:
                        st.warning(f"Map for {selected_geo} not found. Looked for: {map_filename}")

                with legend_col:
                    st.markdown("Open ENOW estimates marine economy establishments, employment, wages and GDP for the coastal portion of each state.")
                    
                    legend_html = """
                        <style>
                            .legend-item { display: flex; align-items: flex-start; margin-top: 15px; }
                            .legend-color-box { width: 25px; height: 25px; min-width: 25px; margin-right: 10px; border: 1px solid #333; }
                            .legend-text { font-size: 1.1rem; }
                        </style>
                        <div class="legend-item">
                            <div class="legend-color-box" style="background-color: #C6E6F0;"></div>
                            <span class="legend-text">Counties shaded in blue in this map are considered coastal for the purposes of estimating employment in the Living Resources, Marine Construction, Marine Transportation, Offshore Mineral Resources, and Ship and Boat Building sectors.</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color-box" style="background-color: #FFFF00;"></div>
                            <span class="legend-text">Zip codes shaded in yellow on this map are considered coastal for the purposes of the Tourism and Recreation sector.</span>
                        </div>
                    """
                    st.markdown(legend_html, unsafe_allow_html=True)
    # --- END: MODIFICATION FOR COASTAL GEOGRAPHIES EXPANDER ---

    # --- Expandable Section for Sector Details ---
    if selected_sector == "All Marine Sectors":
        with st.expander("Marine Sectors in Open ENOW"):
            st.divider()
            st.write("""
            Open ENOW tracks six economic sectors: Living Resources, Marine Construction, 
            Marine Transportation, Offshore Mineral Resources, Ship and Boat Building, and 
            Tourism and Recreation. For a detailed description of each sector, make a 
            selection from the drop-down menu on the left.
            """)
    else:
        if selected_sector in SECTOR_DESCRIPTIONS:
            expander_title = f"The {selected_sector} Sector in Open ENOW"
            with st.expander(expander_title):
                st.divider() 
                sector_info = SECTOR_DESCRIPTIONS[selected_sector]
                st.write(sector_info['description'])
                
                def style_naics_table(row):
                    highlight_codes = ["713110", "721199", "721214"]
                    yellow_style = 'background-color: #FFFF00'
                    gray_style = 'background-color: #f0f0f0'

                    if row['NAICS Code'] in highlight_codes:
                        return [yellow_style for _ in row]

                    years_val = row['Years']
                    is_active = (years_val == "All years") or (years_val.endswith("- present"))
                    if not is_active:
                        return [gray_style for _ in row]

                    return ['' for _ in row]

                st.dataframe(
                    sector_info['table'].style.apply(style_naics_table, axis=1), 
                    use_container_width=True, 
                    hide_index=True
                )
                
    # --- Expandable Section for Metric Details ---
    metric_expander_title = f"{selected_display_metric} in Open ENOW"
    with st.expander(metric_expander_title):
        st.divider()
        description = METRIC_DESCRIPTIONS.get(selected_display_metric, "No description available.")
        st.write(description)


# --- Mode 2: Compare to original ENOW (Unchanged logic, but now an elif) ---
elif plot_mode == "Compare to original ENOW":
    enow_metric_col = selected_metric_internal
    nq_metric_col = f"NQ_{selected_metric_internal}"

    plot_df = base_filtered_df[["Year", enow_metric_col, nq_metric_col]].copy()
    plot_df.rename(columns={
        enow_metric_col: "ENOW",
        nq_metric_col: "Estimate from public QCEW"
    }, inplace=True)
    
    if is_currency:
        plot_df[["ENOW", "Estimate from public QCEW"]] /= 1e6

    compare_df = plot_df.groupby("Year")[["ENOW", "Estimate from public QCEW"]].sum(min_count=1).reset_index()
    compare_df.dropna(subset=["ENOW", "Estimate from public QCEW"], how='all', inplace=True)
    
    long_form_df = compare_df.melt('Year', var_name='Source', value_name='Value')

    if not long_form_df.empty:
        base = alt.Chart(long_form_df).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Value:Q', title=y_label, scale=alt.Scale(zero=True)),
            color=alt.Color('Source:N', 
                            scale=alt.Scale(domain=['ENOW', 'Estimate from public QCEW'], range=['#D55E00', '#0072B2']),
                            legend=alt.Legend(title="Data Source", orient="bottom")),
            tooltip=[
                alt.Tooltip('Year:O', title='Year'),
                alt.Tooltip('Source:N', title='Source'),
                alt.Tooltip('Value:Q', title=selected_display_metric, format=tooltip_format)
            ]
        )
        
        line = base.mark_line()
        points = base.mark_point(size=80, filled=True)
        
        chart = (line + points).properties(
            height=500
        ).configure_axis(
            labelFontSize=14,
            titleFontSize=16
        ).interactive()

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
        diff = compare_df["Estimate from public QCEW"] - compare_df["ENOW"]
        pct_diff = (100 * diff / compare_df["ENOW"]).replace([np.inf, -np.inf], np.nan)
        
        summary_text = f"""
        Mean Difference: {format_value(diff.mean(), selected_display_metric)}
        Median Difference: {format_value(diff.median(), selected_display_metric)}
        Mean Percent Difference: {pct_diff.mean():.2f}%
        """
        st.subheader("Summary Statistics")
        st.code(summary_text, language='text')
    else:
        st.warning("No overlapping data available to compare for the selected filters.")

