# backend/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import osmnx as ox
import networkx as nx

# Import custom modules from our backend folder
from load_fire_data import get_wildfire_data
from model_handler import load_wildfire_model, predict_wildfire_from_image

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Climate Sentinel API",
    description="API for providing wildfire data, calculating safe evacuation routes, and AI-powered predictions.",
    version="1.0.0"
)

# --- 2. Configure CORS ---
# This allows our React frontend to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Load AI Models on Server Startup ---
@app.on_event("startup")
async def startup_event():
    """Event handler to load models when the server starts."""
    load_wildfire_model()

# --- 4. Define Data Models for API Request/Response ---
class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

# --- 5. Caching ---
# Simple in-memory caches for data and graphs to improve performance
graph_cache = {}
wildfire_data_cache = {}

# --- 6. Define API Endpoints ---

@app.get("/wildfires/{year}/{state}")
async def get_wildfires(year: int, state: str):
    """
    Endpoint to get historical wildfire data.
    Data is cached in memory after the first request.
    """
    cache_key = f"{year}-{state}"
    if cache_key in wildfire_data_cache:
        return wildfire_data_cache[cache_key]

    hotspots = get_wildfire_data(year=year, state=state)
    if not hotspots:
        raise HTTPException(status_code=404, detail="No wildfire data found for the specified year and state.")
    
    wildfire_data_cache[cache_key] = hotspots
    return hotspots


@app.post("/calculate-route")
async def calculate_route(request: RouteRequest):
    """
    Endpoint to calculate a safe evacuation route, avoiding wildfire locations.
    """
    try:
        start_point = (request.start_lat, request.start_lon)
        end_point = (request.end_lat, request.end_lon)

        # Load wildfire data for the default region (uses cache)
        hotspots = await get_wildfires(year=2015, state='CA')

        # Download or load street network from cache
        north = max(start_point[0], end_point[0]) + 0.1
        south = min(start_point[0], end_point[0]) - 0.1
        east = max(start_point[1], end_point[1]) + 0.1
        west = min(start_point[1], end_point[1]) - 0.1
        
        graph_cache_key = f"{north:.4f}-{south:.4f}-{east:.4f}-{west:.4f}"

        if graph_cache_key in graph_cache:
            G = graph_cache[graph_cache_key]
        else:
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
            graph_cache[graph_cache_key] = G

        # Identify nodes to avoid based on wildfire locations
        danger_zones = [(fire['longitude'], fire['latitude']) for fire in hotspots]
        danger_radius_meters = 1000
        nodes_to_avoid = set()
        for lon, lat in danger_zones:
            if west < lon < east and south < lat < north:
                center_node = ox.nearest_nodes(G, X=lon, Y=lat)
                nodes_within_radius = nx.ego_graph(G, center_node, radius=danger_radius_meters, distance='length')
                nodes_to_avoid.update(nodes_within_radius.nodes())

        # Create a safe subgraph and calculate the shortest path
        G_safe = G.copy()
        G_safe.remove_nodes_from(list(nodes_to_avoid))
        start_node = ox.nearest_nodes(G_safe, X=start_point[1], Y=start_point[0])
        end_node = ox.nearest_nodes(G_safe, X=end_point[1], Y=end_point[0])
        safe_route_nodes = nx.astar_path(G_safe, start_node, end_node, weight='length')
        
        # Prepare the response as a list of [lat, lon] coordinates
        route_coords = []
        for node_id in safe_route_nodes:
            node = G.nodes[node_id]
            route_coords.append([node['y'], node['x']])

        return {"route": route_coords}

    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="Could not find a safe path.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.post("/predict/wildfire")
async def predict_wildfire(file: UploadFile = File(...)):
    """
    Accepts an image file, processes it through the CNN model, and returns a prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    prediction_result = predict_wildfire_from_image(image_bytes)

    if "error" in prediction_result:
        raise HTTPException(status_code=500, detail=prediction_result["error"])
        
    return prediction_result