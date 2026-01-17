import streamlit as st
import pandas as pd
import pydeck as pdk

# Set page config
st.set_page_config(page_title="Boston Parking Violations Dashboard", layout="wide")

# Custom CSS styling - City of Boston branding
st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        color: #001F3F;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-bottom: 4px solid #C41E3A;
        padding-bottom: 10px;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #001F3F;
        font-weight: 600;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #FFFFFF;
        border-left: 4px solid #0066CC;
        padding: 15px;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0066CC !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 4px !important;
        padding: 12px 24px !important;
        border: 2px solid #0066CC !important;
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #C41E3A !important;
        border-color: #C41E3A !important;
        box-shadow: 0 6px 12px rgba(196, 30, 58, 0.4) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #001F3F;
        font-weight: 600;
    }
    
    /* Text styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('trellint_tickets_deid_lpn_2024.csv') #calls the data from local csv file named 'trellint_tickets_deid_lpn_2024.csv' but the same csv is in the parking_violations_improvement folder
    
    # Drop rows with missing latitude/longitude
    df = df.dropna(subset=["latitude", "longitude"])
    # Filter to keep only unique deid_lpn
    df_unique = df[df['deid_lpn'].value_counts()[df['deid_lpn']].values == 1]
    return df_unique

df_unique = load_data()

# Get unique violation descriptions (filter out NaN values)
violation_types = sorted([v for v in df_unique['violation_desc_long'].unique() if pd.notna(v)])

# Dropdown filter
st.sidebar.header("Filters")
selected_violation = st.sidebar.selectbox(
    "Select Violation Type",
    ["All Violations"] + violation_types
)

# Display dynamic title with selected violation type
if selected_violation == "All Violations":
    st.title("Boston Parking Violations Heatmap (One-Time Offenders): All Violations")
else:
    st.title(f"Boston Parking Violations Heatmap: {selected_violation}")

# Filter data based on selection
if selected_violation == "All Violations":
    filtered_df = df_unique
else:
    filtered_df = df_unique[df_unique['violation_desc_long'] == selected_violation]

# Display stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Violations", f"{len(filtered_df):,}")
with col2:
    st.metric("Unique License Plates", f"{filtered_df['deid_lpn'].nunique():,}")
with col3:
    st.metric("% of Dataset", f"{(len(filtered_df)/len(df_unique)*100):.1f}%")

# Create heatmap
layer = pdk.Layer(
    "HeatmapLayer",
    data=filtered_df[["longitude", "latitude"]].dropna(),
    get_position='[longitude, latitude]',
    radius_pixels=12,
    intensity=1.5,
    threshold=0.3
)

view_state = pdk.ViewState(latitude=42.36, longitude=-71.06, zoom=13)

# Display map
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light"))
