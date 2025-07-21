import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="ENOW Stopgap Estimates",
    layout="wide"
)

# --- Data Loading and Preparation ---
@st.cache_data
def load_data():
    """
    Loads, cleans, and prepares the dataset.
    This function is cached to improve performance.
    """
    try:
        # Load the dataset from the CSV file
        df = pd.read_csv("DORADO_combined_sectors.csv")

        # Rename columns for consistency, similar to the R script
        rename_dict = {
            "NQ_establishments": "NQ_Establishments",
            "NQ_employment": "NQ_Employment",
            "NQ_wages": "NQ_Wages",
            "NP_establishments": "NP_Establishments",
            "NP_employment": "NP_Employment",
            "NP_wages": "NP_Wages"
        }
        df.rename(columns=rename_dict, inplace=True)
        return df
    except FileNotFoundError:
        # Display an error message if the data file is not found
        st.error("Error: 'DORADO_combined_sectors.csv' not found. Please make sure the data file is in the same directory as the app.py script.")
        return None

# Load the data using the cached function
dorado_results = load_data()

# --- Helper Functions ---
def format_value(x, metric):
    """
    Formats numbers with commas and dollar signs, similar to the R helper function.
    """
    if pd.isna(x):
        return "N/A"
    if metric in ["Wages", "GDP", "RealGDP"]:
        return f"${x:,.0f}"
    else:
        return f"{x:,.0f}"

# --- Main Application ---
if dorado_results is not None:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Filters")

    # Get unique values for dropdowns
    unique_states = ["All States"] + sorted(dorado_results["GeoName"].unique())
    unique_sectors = ["All Sectors"] + sorted(dorado_results["OceanSector"].unique())
    metric_choices = ["Employment", "Wages", "Establishments", "GDP", "RealGDP"]

    # Create widgets for user input
    selected_state = st.sidebar.selectbox("Select State:", unique_states)
    selected_sector = st.sidebar.selectbox("Select Ocean Sector:", unique_sectors)
    selected_metric = st.sidebar.selectbox("Select Metric:", metric_choices)
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=int(dorado_results["Year"].min()),
        max_value=int(dorado_results["Year"].max()),
        value=(2011, int(dorado_results["Year"].max())),
        step=1
    )

    # --- Data Filtering Logic ---
    # Filter data based on user selections
    filtered_df = dorado_results[
        (dorado_results["Year"] >= year_range[0]) &
        (dorado_results["Year"] <= year_range[1])
    ]

    if selected_state != "All States":
        filtered_df = filtered_df[filtered_df["GeoName"] == selected_state]

    if selected_sector != "All Sectors":
        filtered_df = filtered_df[filtered_df["OceanSector"] == selected_sector]

    # Define the metric columns to be used based on the selected metric
    nq_metric_col = f"NQ_{selected_metric}"
    np_metric_col = f"NP_{selected_metric}"
    enow_metric_col = selected_metric

    # Create a smaller DataFrame with only the necessary columns
    plot_df = filtered_df[["Year", enow_metric_col, nq_metric_col, np_metric_col]].copy()
    plot_df.rename(columns={
        enow_metric_col: "ENOW_value",
        nq_metric_col: "QCEW_w_imputation",
        np_metric_col: "NPStates_est"
    }, inplace=True)

    # If GDP or RealGDP, divide by 1 million for better readability on the chart
    if selected_metric in ["GDP", "RealGDP"]:
        plot_df["ENOW_value"] /= 1e6
        plot_df["QCEW_w_imputation"] /= 1e6
        plot_df["NPStates_est"] /= 1e6

    # If 'All States' or 'All Sectors' is selected, aggregate the data by year
    if selected_state == "All States" or selected_sector == "All Sectors":
        plot_df = plot_df.groupby("Year").sum().reset_index()

    # --- Main Panel Display ---
    st.title("ENOW Stopgap Estimates: States and Sectors")

    # --- Comparison Plot ---
    st.subheader("Comparison Plot")

    if not plot_df.empty:
        # Define plot labels and styles
        y_label_map = {
            "GDP": "GDP ($ millions)",
            "RealGDP": f"Real GDP ({year_range[0]} $ millions)",
            "Wages": "Wages (USD)",
            "Employment": "Employment (Number of Jobs)",
            "Establishments": "Establishments (Count)"
        }
        y_label = y_label_map.get(selected_metric, selected_metric)

        # Create the plot using Matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the data series
        ax.plot(plot_df["Year"], plot_df["ENOW_value"], 'o-r', label="ENOW")
        ax.plot(plot_df["Year"], plot_df["QCEW_w_imputation"], 's-b', label="Estimate: public QCEW data with imputed values")
        ax.plot(plot_df["Year"], plot_df["NPStates_est"], '^-g', label="Estimate: public QCEW, no imputation")

        # Formatting the plot
        ax.set_xlabel("Year")
        ax.set_ylabel(y_label)
        ax.set_title(f"{selected_metric} Comparison for {selected_state} - {selected_sector}", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc="best")

        # Format y-axis labels with commas and/or dollar signs
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format_value(x, selected_metric))
        )
        plt.xticks(plot_df["Year"].unique(), rotation=45)
        fig.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters. Please adjust your selections.")


    # --- Summary Statistics ---
    st.subheader("Summary Statistics")
    summary_df = plot_df.dropna()

    if not summary_df.empty:
        # Calculate differences and percentage differences
        diff_nq = summary_df["QCEW_w_imputation"] - summary_df["ENOW_value"]
        pct_diff_nq = 100 * diff_nq / summary_df["ENOW_value"]

        diff_np = summary_df["NPStates_est"] - summary_df["ENOW_value"]
        pct_diff_np = 100 * diff_np / summary_df["ENOW_value"]
        
        # Replace infinite values with NaN for proper calculation
        pct_diff_nq.replace([np.inf, -np.inf], np.nan, inplace=True)
        pct_diff_np.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Create the summary text
        summary_text = f"""
        Public QCEW with imputed values vs ENOW:
          - Mean Difference: {format_value(diff_nq.mean(), selected_metric)}
          - Median Difference: {format_value(diff_nq.median(), selected_metric)}
          - Mean Percent Difference: {pct_diff_nq.mean():.2f}%

        Non-Participating States method (no imputation) vs ENOW:
          - Mean Difference: {format_value(diff_np.mean(), selected_metric)}
          - Median Difference: {format_value(diff_np.median(), selected_metric)}
          - Mean Percent Difference: {pct_diff_np.mean():.2f}%
        """
        st.code(summary_text, language='text')
    else:
        st.warning("Cannot calculate summary statistics because there is no overlapping data for the selected filters.")

