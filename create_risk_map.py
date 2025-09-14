import folium
from folium.plugins import HeatMap
from load_fire_data import get_wildfire_data # <-- IMPORTANT: We import our NEW function

# --- 1. Load the Historical Wildfire Data ---
# We'll get data for California fires in 2015 for this map.
hotspots = get_wildfire_data(year=2015, state='CA')

# --- 2. Create the Interactive Map ---
def create_map_with_hotspots(hotspot_data):
    """
    Creates a Folium map centered on California and overlays wildfire hotspots.
    """
    if not hotspot_data:
        print("No hotspot data to map. Exiting.")
        return

    # Create a map centered on California
    map_center = [36.7783, -119.4179]
    risk_map = folium.Map(location=map_center, zoom_start=6)

    # --- 3. Add a Heatmap Layer ---
    # Prepare the data for the heatmap: a list of [lat, lon]
    heat_data = [[hotspot['latitude'], hotspot['longitude']] for hotspot in hotspot_data]
    
    HeatMap(heat_data, radius=10).add_to(risk_map)

    # --- 4. Add Individual Hotspot Markers ---
    for hotspot in hotspot_data:
        folium.CircleMarker(
            location=[hotspot['latitude'], hotspot['longitude']],
            radius=1,
            color='red',
            fill=True,
            fill_color='darkred',
            # Add a popup with fire size information
            popup=f"State: {hotspot['state']}<br>Size: {hotspot['fire_size']:.2f} acres"
        ).add_to(risk_map)
        
    # --- 5. Save the Map to an HTML File ---
    output_file = "wildfire_risk_map_local.html"
    risk_map.save(output_file)
    print(f"\nMap saved successfully! Open '{output_file}' in your web browser.")


# --- How to Test This Module Independently ---
if __name__ == "__main__":
    if hotspots:
        create_map_with_hotspots(hotspots)