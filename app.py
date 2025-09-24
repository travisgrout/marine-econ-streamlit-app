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

# --- Data Dictionaries (Omitted for brevity) ---
SECTOR_DESCRIPTIONS = {} # Your sector descriptions here
METRIC_DESCRIPTIONS = {} # Your metric descriptions here

# --- Main Application ---
METRIC_MAP = {
    "Employment": "Employment", "Wages (not inflation-adjusted)": "Wages",
    "Real Wages": "RealWages", "Establishments": "Establishments",
    "GDP (nominal)": "GDP", "Real GDP": "RealGDP"
}

st.sidebar.image("open_ENOW_logo.png", width=200)

# --- Pop-up Window (Omitted for brevity) ---
popover = st.sidebar.popover("What is Open ENOW?")
popover.markdown("...") # Your pop-up text here

st.sidebar.header("Display Mode:")

# --- Custom Button Display Mode ---
button_map = {"States": "State Estimates from Public QCEW Data", "Counties": "County Estimates from Public QCEW Data", "Regions": "Regional Estimates from Public QCEW Data", "Compare": "Compare to original ENOW"}
if 'plot_mode' not in st.session_state:
    st.session_state.plot_mode = button_map["States"]

st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Your CSS here

def update_mode(mode_label):
    st.session_state.plot_mode = button_map[mode_label]

cols1 = st.sidebar.columns(2)
# ... (Button rendering code omitted for brevity) ...

plot_mode = st.session_state.plot_mode

# --- Select Active DataFrame and Set Filters based on Mode ---
estimate_modes = ["State Estimates from Public QCEW Data", "County Estimates from Public QCEW Data", "Regional Estimates from Public QCEW Data"]

if plot_mode in estimate_modes:
    # ... (Logic for estimate modes omitted for brevity) ...
    active_df = open_enow_data
    # This section remains unchanged
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

        # ... (Summary Statistics and Download button logic remains the same) ...
    else:
        st.warning("No overlapping data available to compare for the selected filters.")
