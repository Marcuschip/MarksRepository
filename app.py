import streamlit as st
import pandas as pd
import plotly.express as px

#load parquet into pandas
df = pd.read_parquet("filtered_wagons.pqt")

#ensure timestamp is datetime
df['hub_timestamp'] = pd.to_datetime(df['hub_timestamp'])

#column groupings
speed_columns = ["gps_speed_kph", "egs_kph", "axle_i_speed_kph", "axle_o_speed_kph", "axle_i_rot_speed_rpm", "axle_o_rot_speed_rpm"]  # replace with actual variable names
brake_columns = ["bcp_bar", "bpp_bar", "ip1_bar", "ip2_bar", "lwp_bar"]  # replace with actual variable names
flag_columns = ["mag_1", "mag_2", "wfp_active", "wfp_high", "sustained_wfp", "locked_axle", "persistent_locked_axle", "partial_axle_rotation", "vibration_shock", "load_shock", "shunt_shock", "deep_sleep", "wake_up", "non-stationary", "moving", "stationary", "shutdown", "wfp_inhibit", "undemanded_brake", "pull_away_whilst_braking", "overcharge_function_detected", "interrupted_overcharge", "high_shock"]  # replace with actual variable names

#sidebar filters single selection
wagon_id = st.sidebar.selectbox("Wagon ID", df["wagon"].unique())
date = st.sidebar.date_input("Date")
hour = st.sidebar.selectbox("Hour", df["hub_timestamp"].dt.hour.unique())
mfv = st.sidebar.multiselect("MFV", df["mfv"].unique())

#multi select sidebar filters
selected_speeds = st.sidebar.multiselect("Speed", speed_columns, default = [speed_columns[0]])
selected_brakes = st.sidebar.multiselect("Brake Pressures", brake_columns, default = [brake_columns[0]])
selected_flags = st.sidebar.multiselect("Flags", flag_columns, default = [flag_columns[0]])

#filter dataframe based on sidebar selections
filtered = df.copy()
filtered = filtered[filtered["wagon"] == wagon_id]
if date:
    filtered = filtered[filtered["hub_timestamp"].dt.date == pd.to_datetime(date).date()]
if hour is not None:
    filtered = filtered[filtered["hub_timestamp"].dt.hour == hour]
if mfv:
    filtered = filtered[filtered["mfv"].isin(mfv)]

#speed chart
if selected_speeds:
    melted = filtered.melt(
        id_vars=["hub_timestamp"],
        value_vars = selected_speeds,
        var_name = "SpeedType",
        value_name="SpeedValue"
    )
    fig_speed = px.line(
        melted,
        x = "hub_timestamp",
        y = "SpeedValue",
        color = "SpeedType",
        title = f"Speeds for Wagon {wagon_id} on {date} at {hour}"
    )
    st.plotly_chart(fig_speed, use_container_width=True)

#brake chart
if selected_brakes:
    melted = filtered.melt(
        id_vars=["hub_timestamp"],
        value_vars = selected_brakes,
        var_name = "BrakeType",
        value_name="BrakeValue"
    )
    fig_brake = px.line(
        melted,
        x = "hub_timestamp",
        y = "BrakeValue",
        color = "BrakeType",
        title = f"Brake Pressures for Wagon {wagon_id} on {date} at {hour}"
    )
    st.plotly_chart(fig_brake, use_container_width=True)

#flag chart
if selected_flags:
    melted = filtered.melt(
        id_vars=["hub_timestamp"],
        value_vars = selected_flags,
        var_name = "FlagType",
        value_name="FlagValue"
    )
    fig_flag = px.line(
        melted,
        x = "hub_timestamp",
        y = "FlagValue",
        color = "FlagType",
        title = f"Flags for Wagon {wagon_id} on {date} at {hour}"
    )
    st.plotly_chart(fig_flag, use_container_width=True)

#map
fig_map = px.scatter_mapbox(
    filtered,
    lat = "gps_latitude_deg",
    lon = "gps_longitude_deg",
    color = "hub_timestamp",
    hover_data = ["hub_timestamp"]+selected_speeds+selected_brakes+selected_flags,
    title = f"Location of Wagon {wagon_id}"
)
fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=10
)
st.plotly_chart(fig_map, use_container_width=True)