import folium
import osmnx as ox
import networkx as nx
from load_fire_data import get_wildfire_data # Import our local data loader

# --- Configuration ---
PLACE_NAME = "Berkeley, California, USA"
START_POINT = (37.8716, -122.2727) # Downtown Berkeley
END_POINT = (37.8272, -122.2901)   # A safe point west, near the water
DANGER_RADIUS_METERS = 1000 # 1km risk radius around each fire

# --- 1. Get Wildfire Data to Simulate Danger Zones ---
all_fires = get_wildfire_data(year=2015, state='CA')

DANGER_ZONES = []
if all_fires:
    for fire in all_fires:
        if -122.4 < fire['longitude'] < -122.1 and 37.8 < fire['latitude'] < 37.9:
            DANGER_ZONES.append((fire['longitude'], fire['latitude']))
            if len(DANGER_ZONES) >= 3:
                break
if not DANGER_ZONES:
    print("No historical fires found near Berkeley in the dataset. Using dummy danger zones.")
    DANGER_ZONES = [(-122.26, 37.86), (-122.25, 37.85)]

# --- 2. Download the Street Network ---
print(f"Downloading street network for '{PLACE_NAME}'...")
G = ox.graph_from_place(PLACE_NAME, network_type='drive')
print("Network download complete.")

# --- 3. Identify Nodes to Avoid ---
nodes_to_avoid = set()
for lon, lat in DANGER_ZONES:
    center_node = ox.nearest_nodes(G, X=lon, Y=lat)
    nodes_within_radius = nx.ego_graph(G, center_node, radius=DANGER_RADIUS_METERS, distance='length')
    nodes_to_avoid.update(nodes_within_radius.nodes())
print(f"Identified {len(nodes_to_avoid)} network nodes within {len(DANGER_ZONES)} danger zones.")

# --- 4. Create a Subgraph and Find the Route Start/End Nodes ---
G_safe = G.copy()
G_safe.remove_nodes_from(list(nodes_to_avoid))

start_node = ox.nearest_nodes(G_safe, X=START_POINT[1], Y=START_POINT[0])
end_node = ox.nearest_nodes(G_safe, X=END_POINT[1], Y=END_POINT[0])

# --- 5. Calculate Route and Visualize ---
try:
    print("Calculating safest evacuation route...")
    safe_route_nodes = nx.astar_path(G_safe, start_node, end_node, weight='length')
    print("Route calculated successfully.")

    # --- THIS IS THE NEW, CORRECT VISUALIZATION METHOD ---
    # Create a base map centered on the start point
    route_map = folium.Map(location=START_POINT, zoom_start=14)

    # Get the coordinates for the route nodes
    route_coords = []
    for node_id in safe_route_nodes:
        node = G.nodes[node_id]
        route_coords.append((node['y'], node['x'])) # Append (lat, lon)

    # Add the route as a blue line to the map
    folium.PolyLine(
        locations=route_coords,
        color='blue',
        weight=5,
        opacity=0.8
    ).add_to(route_map)

    # Add markers for start, end, and danger zones
    folium.Marker(location=START_POINT, popup="Evacuation Start", icon=folium.Icon(color='green')).add_to(route_map)
    folium.Marker(location=END_POINT, popup="Safe Destination", icon=folium.Icon(color='blue')).add_to(route_map)

    for zone_lon, zone_lat in DANGER_ZONES:
        folium.Circle(
            location=(zone_lat, zone_lon),
            radius=DANGER_RADIUS_METERS,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.3,
            popup="DANGER ZONE"
        ).add_to(route_map)

    # Save the map
    output_file = "evacuation_route_map_local.html"
    route_map.save(output_file)
    print(f"\nEvacuation map saved! Open '{output_file}' in your web browser.")

except nx.NetworkXNoPath:
    print("Error: Could not find a safe path. The destination may be unreachable or surrounded by danger zones.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")