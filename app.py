import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from textwrap import wrap

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
        
        # --- ROBUSTNESS FIX ---
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
    # ### MODIFICATION ###
    # Updated the metric names in the check
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
    return base_colors[:n] if n <= len(base_colors) else plt.cm.viridis(np.linspace(0, 1, n))

# --- Main Application ---
if dorado_results is not None:
    st.title("Marine Economy estimates: states and sectors")

    # ### MODIFICATION ###
    # Dictionary to map user-facing metric names to internal data names.
    METRIC_MAP = {
        "Employment": "Employment",
        "Wages (not inflation-adjusted)": "Wages",
        "Real Wages": "RealWages",
        "Establishments": "Establishments",
        "GDP (nominal)": "GDP",
        "Real GDP": "RealGDP"
    }

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

    # Conditionally set the metric choices based on the selected plot_mode.
    if plot_mode == "Estimates from Public QCEW Data":
        metric_choices = list(METRIC_MAP.keys())
    else:
        # Create a new dictionary for compare mode that excludes "Real Wages"
        compare_metrics = {k: v for k, v in METRIC_MAP.items() if v != "RealWages"}
        metric_choices = list(compare_metrics.keys())

    selected_state = st.sidebar.selectbox("Select State:", unique_states)
    selected_sector = st.sidebar.selectbox("Select Marine Sector:", unique_sectors)
    
    # ### MODIFICATION ###
    # The selected metric is now the user-facing display name
    selected_display_metric = st.sidebar.selectbox("Select Metric:", metric_choices)
    
    # ### MODIFICATION ###
    # We map the display name back to the internal name for data processing
    selected_metric_internal = METRIC_MAP[selected_display_metric]


    min_year, max_year = int(dorado_results["Year"].min()), int(dorado_results["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    # --- Base Data Filtering (applied to both modes) ---
    base_filtered_df = dorado_results[
        (dorado_results["Year"] >= year_range[0]) &
        (dorado_results["Year"] <= year_range[1])
    ]
    if selected_state != "All Coastal States":
        base_filtered_df = base_filtered_df[base_filtered_df["GeoName"] == selected_state]
    if selected_sector != "All Sectors":
        base_filtered_df = base_filtered_df[base_filtered_df["OceanSector"] == selected_sector]

    # --- Plotting and Visualization ---
    
    # ### MODIFICATION ###
    # y_label_map now uses the new display names as keys.
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
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Mode 1: Estimates from Public QCEW Data ---
    if plot_mode == "Estimates from Public QCEW Data":
        # We use the internal metric name for constructing the column name
        nq_metric_col = f"NQ_{selected_metric_internal}"
        
        plot_df = base_filtered_df[["Year", "OceanSector", nq_metric_col]].copy()
        plot_df.rename(columns={nq_metric_col: "Estimate_value"}, inplace=True)
        plot_df.dropna(subset=["Estimate_value"], inplace=True)
        
        # ### MODIFICATION ###
        # Check now uses the new display names
        if selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)", "Real Wages"]:
            plot_df["Estimate_value"] /= 1e6

        if selected_sector == "All Sectors":
            agg_df = plot_df.groupby(["Year", "OceanSector"])["Estimate_value"].sum().unstack()
            if not agg_df.empty:
                colors = get_sector_colors(len(agg_df.columns))
                agg_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
                wrapped_labels = ['\n'.join(wrap(l, 20)) for l in agg_df.columns]
                ax.legend(wrapped_labels, title="Sectors", bbox_to_anchor=(1.04, 1), loc="upper left")
                fig.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                st.warning("No data available for the selected filters.")
        else:
            bar_df = plot_df.groupby("Year")["Estimate_value"].sum().reset_index() if selected_state == "All Coastal States" else plot_df
            if not bar_df.empty:
                ax.bar(bar_df["Year"], bar_df["Estimate_value"], color="#0072B2", label="Estimate from public QCEW")
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
            else:
                st.warning("No data available for the selected filters.")

    # --- Mode 2: Compare to ENOW ---
    elif plot_mode == "Compare to ENOW":
        # We use the internal metric name for data selection
        enow_metric_col = selected_metric_internal
        nq_metric_col = f"NQ_{selected_metric_internal}"

        plot_df = base_filtered_df[["Year", "OceanSector", enow_metric_col, nq_metric_col]].copy()
        plot_df.rename(columns={
            enow_metric_col: "ENOW_value",
            nq_metric_col: "Estimate_value"
        }, inplace=True)
        
        # ### MODIFICATION ###
        # Check now uses the new display names
        if selected_display_metric in ["GDP (nominal)", "Real GDP", "Wages (not inflation-adjusted)"]:
            plot_df[["ENOW_value", "Estimate_value"]] = plot_df[["ENOW_value", "Estimate_value"]].div(1e6)

        compare_df = plot_df.groupby("Year")[["ENOW_value", "Estimate_value"]].sum(min_count=1).reset_index()
        compare_df.dropna(subset=["ENOW_value", "Estimate_value"], inplace=True)

        if not compare_df.empty:
            ax.plot(compare_df["Year"], compare_df["ENOW_value"], 'o-', color="#D55E00", label="ENOW", markersize=8)
            ax.plot(compare_df["Year"], compare_df["Estimate_value"], 's-', color="#0072B2", label="Estimate from public QCEW", markersize=8)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            
            # --- Summary Statistics ---
            diff = compare_df["Estimate_value"] - compare_df["ENOW_value"]
            pct_diff = (100 * diff / compare_df["ENOW_value"]).replace([np.inf, -np.inf], np.nan)
            
            summary_text = f"""
            Mean Difference: {format_value(diff.mean(), selected_display_metric)}
            Median Difference: {format_value(diff.median(), selected_display_metric)}
            Mean Percent Difference: {pct_diff.mean():.2f}%
            """
            st.subheader("Summary Statistics")
            st.code(summary_text, language='text')
        else:
            st.warning("No overlapping data available to compare for the selected filters.")

    # --- Common Plot Formatting ---
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('\n'.join(wrap(plot_title, 60)), fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if plot_mode == "Compare to ENOW" or selected_sector != "All Sectors":
        years_df = compare_df if plot_mode == "Compare to ENOW" else bar_df
        all_years = sorted(years_df["Year"].unique())
        if all_years:
            ax.set_xticks(all_years)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    st.pyplot(fig)
