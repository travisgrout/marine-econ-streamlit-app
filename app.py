import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from textwrap import wrap

# --- Page Configuration ---
st.set_page_config(
    page_title="ENOW Lite Estimates",
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
        # Rename columns for consistency, similar to the R script
        rename_dict = {
            "NQ_establishments": "NQ_Establishments",
            "NQ_employment": "NQ_Employment",
            "NQ_wages": "NQ_Wages",
        }
        df.rename(columns=rename_dict, inplace=True)
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
    if metric in ["Wages", "GDP", "RealGDP"]:
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
    st.title("ENOW Lite estimates: states and sectors")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Filters")
    plot_mode = st.sidebar.radio(
        "Display Mode:",
        ("Show Estimates for All Years", "Compare to ENOW"),
        index=0
    )

    unique_states = ["All States"] + sorted(dorado_results["GeoName"].unique())
    unique_sectors = ["All Sectors"] + sorted(dorado_results["OceanSector"].unique())
    metric_choices = ["Employment", "Wages", "Establishments", "GDP", "RealGDP"]

    selected_state = st.sidebar.selectbox("Select State:", unique_states)
    selected_sector = st.sidebar.selectbox("Select Ocean Sector:", unique_sectors)
    selected_metric = st.sidebar.selectbox("Select Metric:", metric_choices)

    min_year, max_year = int(dorado_results["Year"].min()), int(dorado_results["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(2012, 2021),
        step=1
    )

    # --- Data Filtering and Processing ---
    enow_metric_col = selected_metric
    nq_metric_col = f"NQ_{selected_metric}"

    # Base filtering on user selections
    filtered_df = dorado_results[
        (dorado_results["Year"] >= year_range[0]) &
        (dorado_results["Year"] <= year_range[1])
    ]
    if selected_state != "All States":
        filtered_df = filtered_df[filtered_df["GeoName"] == selected_state]
    if selected_sector != "All Sectors":
        filtered_df = filtered_df[filtered_df["OceanSector"] == selected_sector]

    # Select and rename columns
    plot_df = filtered_df[["Year", "OceanSector", enow_metric_col, nq_metric_col]].copy()
    plot_df.rename(columns={
        enow_metric_col: "ENOW_value",
        nq_metric_col: "Estimate_value"
    }, inplace=True)
    
    # Scale values for display
    if selected_metric in ["GDP", "RealGDP", "Wages"]:
        plot_df["ENOW_value"] /= 1e6
        plot_df["Estimate_value"] /= 1e6

    # --- Plotting and Statistics Logic ---
    st.subheader("Comparison Plot")

    y_label_map = {
        "GDP": "GDP ($ millions)",
        "RealGDP": "Real GDP ($ millions)",
        "Wages": "Wages ($ millions)",
        "Employment": "Employment (number of jobs)",
        "Establishments": "Establishments (count)"
    }
    y_label = y_label_map.get(selected_metric, selected_metric)
    plot_title = f"{selected_metric} in {selected_state} - {selected_sector}"
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plot Mode: Compare to ENOW ---
    if plot_mode == "Compare to ENOW":
        # Aggregate data if 'All' is selected
        if selected_state == "All States" or selected_sector == "All Sectors":
            compare_df = plot_df.groupby("Year")[["ENOW_value", "Estimate_value"]].sum().reset_index()
        else:
            compare_df = plot_df
        
        compare_df.dropna(subset=["ENOW_value", "Estimate_value"], inplace=True)

        if not compare_df.empty:
            ax.plot(compare_df["Year"], compare_df["ENOW_value"], 'o-', color="#D55E00", label="ENOW")
            ax.plot(compare_df["Year"], compare_df["Estimate_value"], 's-', color="#0072B2", label="Estimate from public QCEW")
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
            
            # --- Summary Statistics ---
            diff = compare_df["Estimate_value"] - compare_df["ENOW_value"]
            pct_diff = 100 * diff / compare_df["ENOW_value"]
            pct_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            summary_text = f"""
            Mean Difference: {format_value(diff.mean(), selected_metric)}
            Median Difference: {format_value(diff.median(), selected_metric)}
            Mean Percent Difference: {pct_diff.mean():.2f}%
            """
            st.subheader("Summary Statistics")
            st.code(summary_text, language='text')

        else:
            st.warning("No overlapping data available to compare for the selected filters.")

    # --- Plot Mode: Show Estimates for All Years ---
    else:
        # Stacked bar chart for "All Sectors"
        if selected_sector == "All Sectors":
            if selected_state == "All States":
                agg_df = plot_df.groupby(["Year", "OceanSector"])["Estimate_value"].sum().unstack()
            else:
                agg_df = plot_df.groupby(["Year", "OceanSector"])["Estimate_value"].sum().unstack()

            if not agg_df.empty:
                colors = get_sector_colors(len(agg_df.columns))
                agg_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
                
                # Wrap legend labels
                wrapped_labels = ['\n'.join(wrap(l, 20)) for l in agg_df.columns]
                ax.legend(wrapped_labels, title="Sectors", bbox_to_anchor=(1.04, 1), loc="upper left")
                fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            else:
                 st.warning("No data available for the selected filters.")

        # Simple bar chart for a single sector
        else:
            if selected_state == "All States":
                bar_df = plot_df.groupby("Year")["Estimate_value"].sum().reset_index()
            else:
                bar_df = plot_df
                
            if not bar_df.empty:
                ax.bar(bar_df["Year"], bar_df["Estimate_value"], color="#0072B2", label="Estimate from public QCEW")
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
            else:
                st.warning("No data available for the selected filters.")


    # --- Common Plot Formatting ---
    if plot_mode == "Compare to ENOW" or selected_sector != "All Sectors":
        ax.set_xticks(np.arange(year_range[0], year_range[1] + 1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format y-axis to have commas
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    st.pyplot(fig)
