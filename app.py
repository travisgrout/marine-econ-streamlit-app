import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from textwrap import wrap
import os # <-- ADDED: To check for file existence

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


# --- Main Application ---
if dorado_results is not None:
    st.title("Marine Economy estimates: states and sectors")

    METRIC_MAP = {
        "Employment": "Employment",
        "Wages (not inflation-adjusted)": "Wages",
        "Real Wages": "RealWages",
        "Establishments": "Establishments",
        "GDP (nominal)": "GDP",
        "Real GDP": "RealGDP"
    }
    
    st.sidebar.image("open_ENOW_logo.png", width=200)
    st.sidebar.header("Filters")
    plot_mode = st.sidebar.radio(
        "Display Mode:",
        ("Estimates from Public QCEW Data", "Compare to ENOW"),
        index=0
    )

    geo_names = dorado_results["GeoName"].dropna().unique()
    unique_states = ["All Coastal States"] + sorted(geo_names)

    ocean_sectors = dorado_results["OceanSector"].dropna().unique()
    unique_sectors = ["All Sectors"] + sorted(ocean_sectors)
    
    sorted_sector_names = sorted(ocean_sectors)
    colors_list = get_sector_colors(len(sorted_sector_names))
    sector_color_map = dict(zip(sorted_sector_names, colors_list))

    if plot_mode == "Estimates from Public QCEW Data":
        metric_choices = list(METRIC_MAP.keys())
    else:
        compare_metrics = {k: v for k, v in METRIC_MAP.items() if v != "RealWages"}
        metric_choices = list(compare_metrics.keys())

    selected_state = st.sidebar.selectbox("Select State:", unique_states)
    selected_sector = st.sidebar.selectbox("Select Marine Sector:", unique_sectors)
    selected_display_metric = st.sidebar.selectbox("Select Metric:", metric_choices)
    selected_metric_internal = METRIC_MAP[selected_display_metric]

    min_year, max_year = int(dorado_results["Year"].min()), int(dorado_results["Year"].max())
    
    # Set the default year range based on the selected plot mode.
    if plot_mode == "Estimates from Public QCEW Data":
        default_start_year = max(min_year, 2014)
        default_end_year = min(max_year, 2023)
        default_range = (default_start_year, default_end_year)
    else:  # "Compare to ENOW"
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

    # --- Base Data Filtering ---
    base_filtered_df = dorado_results[
        (dorado_results["Year"] >= year_range[0]) &
        (dorado_results["Year"] <= year_range[1])
    ]
    if selected_state != "All Coastal States":
        base_filtered_df = base_filtered_df[base_filtered_df["GeoName"] == selected_state]
    if selected_sector != "All Sectors":
        base_filtered_df = base_filtered_df[base_filtered_df["OceanSector"] == selected_sector]

    # --- Plotting and Visualization ---
    y_label_map = {
        "GDP (nominal)": "GDP ($ millions)",
        "Real GDP": "Real GDP (millions of 2017 USD)",
        "Wages (not inflation-adjusted)": "Wages ($ millions)",
        "Real Wages": "Real Wages ($ millions, 2017)",
        "Employment": "Employment (Number of Jobs)",
        "Establishments": "Establishments (Count)"
    }
    y_label = y_label_map.get(selected_display_metric, selected_display_metric)
    plot_title = f"Marine Economy {selected_display_metric} in {selected_state} - {selected_sector} Sector"
    
    is_currency = selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)", "Real Wages"]
    tooltip_format = '$,.0f' if is_currency else ',.0f'

    # --- Mode 1: Estimates from Public QCEW Data ---
    if plot_mode == "Estimates from Public QCEW Data":
        nq_metric_col = f"NQ_{selected_metric_internal}"
        
        plot_df = base_filtered_df[["Year", "OceanSector", nq_metric_col]].copy()
        plot_df.rename(columns={nq_metric_col: "Estimate_value"}, inplace=True)
        plot_df.dropna(subset=["Estimate_value"], inplace=True)
        
        if is_currency:
            plot_df["Estimate_value"] /= 1e6

        if selected_sector == "All Sectors":
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
                ).properties(
                    title=plot_title,
                    height=500
                ).configure_axis(
                    labelFontSize=14,
                    titleFontSize=16
                ).configure_legend(
                    symbolLimit=len(sorted_sector_names)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")

        else: # Single-sector chart
            bar_df = plot_df.groupby("Year")["Estimate_value"].sum().reset_index()
            if not bar_df.empty:
                sector_color = sector_color_map.get(selected_sector, "#808080")
                chart = alt.Chart(bar_df).mark_bar(color=sector_color).encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Estimate_value:Q', title=y_label),
                    tooltip=[
                        alt.Tooltip('Year:O', title='Year'),
                        alt.Tooltip('Estimate_value:Q', title=selected_display_metric, format=tooltip_format)
                    ]
                ).properties(
                    title=plot_title,
                    height=500
                ).configure_axis(
                    labelFontSize=14,
                    titleFontSize=16
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")

        # --- ADDED: Map and Legend Display ---
        if selected_state != "All Coastal States":

            # Added CSS to increase the expander title font size
            st.markdown("""
                <style>
                div[data-testid="stExpander"] summary {
                    font-size: 1.75rem;
                }
                </style>
                """, unsafe_allow_html=True)
            
            # Wrapped the entire map/legend section in an expander
            with st.expander("Coastal geographies in Open ENOW"):
                st.divider()  # Adds a horizontal line for separation inside the expander

                # CSS to vertically center the columns
                st.markdown("""
                    <style>
                    div[data-testid="stHorizontalBlock"] {
                        align-items: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                # Create two columns: 2/3 for the map, 1/3 for the legend
                map_col, legend_col = st.columns([2, 1])

                with map_col:
                    # Format the state name for the filename
                    map_filename = f"ENOW state maps/Map_{selected_state.replace(' ', '_')}.jpg"

                    # Check if the map file exists before trying to display it
                    if os.path.exists(map_filename):
                        with st.container(border=True):
                            st.image(map_filename, use_container_width=True)
                    else:
                        st.warning(f"Map for {selected_state} not found. Looked for: {map_filename}")

                with legend_col:
                    # Legend Title and custom HTML for colored boxes
                    st.markdown("Open ENOW estimates marine economy establishments, employment, wages and GDP for the coastal portion of each state.")
                    
                    legend_html = """
                        <style>
                            .legend-item {
                                display: flex;
                                align-items: flex-start;
                                margin-top: 15px;
                            }
                            .legend-color-box {
                                width: 25px;
                                height: 25px;
                                min-width: 25px;
                                margin-right: 10px;
                                border: 1px solid #333;
                            }
                            .legend-text {
                                font-size: 1.1rem; 
                            }
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

    # --- Mode 2: Compare to ENOW ---
    elif plot_mode == "Compare to ENOW":
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
                title=plot_title,
                height=500
            ).configure_axis(
                labelFontSize=14,
                titleFontSize=16
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

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
