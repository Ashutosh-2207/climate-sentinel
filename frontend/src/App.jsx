// frontend/src/App.jsx

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';

// --- Main App Component ---
function App() {
  // --- State for Map and Routing ---
  const [wildfires, setWildfires] = useState([]);
  const [route, setRoute] = useState(null);
  const [startPoint, setStartPoint] = useState([37.8716, -122.2727]);
  const [endPoint, setEndPoint] = useState([37.8272, -122.2901]);
  const [loadingRoute, setLoadingRoute] = useState(false);
  const [error, setError] = useState('');

  // --- NEW: State for AI Prediction ---
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);


  // --- API Base URL ---
  const API_URL = 'http://127.0.0.1:8000';

  // --- Data Fetching Effect for Wildfires ---
  useEffect(() => {
    const fetchWildfireData = async () => {
      try {
        const response = await axios.get(`${API_URL}/wildfires/2015/CA`);
        setWildfires(response.data);
      } catch (err) {
        console.error("Error fetching wildfire data:", err);
        setError('Could not load wildfire data. Is the backend running?');
      }
    };
    fetchWildfireData();
  }, []);

  // --- Event Handler for Calculating Route ---
  const handleCalculateRoute = async () => {
    setLoadingRoute(true);
    setError('');
    setRoute(null);
    try {
      const response = await axios.post(`${API_URL}/calculate-route`, {
        start_lat: startPoint[0],
        start_lon: startPoint[1],
        end_lat: endPoint[0],
        end_lon: endPoint[1],
      });
      setRoute(response.data.route);
    } catch (err) {
      console.error("Error calculating route:", err);
      setError(err.response?.data?.detail || 'Failed to calculate route.');
    } finally {
      setLoadingRoute(false);
    }
  };

  // --- NEW: Event Handlers for AI Prediction ---
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionResult(null); // Clear previous results
  };

  const handlePrediction = async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }
    setLoadingPrediction(true);
    setError('');
    
    // FormData is required for sending files
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict/wildfire`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictionResult(response.data);
    } catch (err) {
      console.error("Error predicting wildfire:", err);
      setError(err.response?.data?.detail || 'Failed to get prediction.');
    } finally {
      setLoadingPrediction(false);
    }
  };
  
  return (
    <div style={{ display: 'flex' }}>
      {/* --- Sidebar --- */}
      <div style={{ width: '350px', padding: '20px', backgroundColor: '#f0f0f0', height: '100vh', overflowY: 'auto' }}>
        <h1>🔥 Climate Sentinel</h1>
        <p>Wildfire Risk & Evacuation Planner</p>
        <hr />
        
        {/* --- Evacuation Section --- */}
        <h3>Evacuation Route</h3>
        <div>
          <label>Start Latitude</label>
          <input type="text" value={startPoint[0]} onChange={(e) => setStartPoint([parseFloat(e.target.value), startPoint[1]])} style={{ width: '100%', marginBottom: '10px' }} />
          <label>Start Longitude</label>
          <input type="text" value={startPoint[1]} onChange={(e) => setStartPoint([startPoint[0], parseFloat(e.target.value)])} style={{ width: '100%', marginBottom: '10px' }} />
        </div>
        <div>
          <label>End Latitude</label>
          <input type="text" value={endPoint[0]} onChange={(e) => setEndPoint([parseFloat(e.target.value), endPoint[1]])} style={{ width: '100%', marginBottom: '10px' }} />
          <label>End Longitude</label>
          <input type="text" value={endPoint[1]} onChange={(e) => setEndPoint([endPoint[0], parseFloat(e.target.value)])} style={{ width: '100%', marginBottom: '20px' }} />
        </div>
        <button onClick={handleCalculateRoute} disabled={loadingRoute} style={{ width: '100%', padding: '10px' }}>
          {loadingRoute ? 'Calculating...' : 'Calculate Safe Route'}
        </button>
        
        <hr style={{ marginTop: '30px', marginBottom: '30px' }} />

        {/* --- NEW: AI Prediction Section --- */}
        <h3>Wildfire Image Analysis</h3>
        <p>Upload a satellite or drone image to detect active fires.</p>
        <input type="file" onChange={handleFileChange} accept="image/*" style={{ width: '100%', marginBottom: '10px' }} />
        <button onClick={handlePrediction} disabled={!selectedFile || loadingPrediction} style={{ width: '100%', padding: '10px' }}>
          {loadingPrediction ? 'Analyzing...' : 'Analyze Image'}
        </button>
        {predictionResult && (
          <div style={{ marginTop: '20px', padding: '10px', borderRadius: '5px', backgroundColor: predictionResult.prediction === 'Fire Detected' ? '#ffdddd' : '#ddffdd' }}>
            <strong>Result:</strong> {predictionResult.prediction} <br />
            <strong>Confidence:</strong> {predictionResult.confidence}
          </div>
        )}
        
        {/* --- Universal Error Display --- */}
        {error && <p style={{ color: 'red', marginTop: '10px' }}>Error: {error}</p>}
      </div>

      {/* --- Map --- */}
      <MapContainer center={[36.7783, -119.4179]} zoom={6}>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        {wildfires.map((fire, index) => (
          <Circle key={index} center={[fire.latitude, fire.longitude]} radius={fire.fire_size * 0.5} pathOptions={{ color: 'red', fillColor: 'darkred', fillOpacity: 0.4 }}>
            <Popup>Size: {fire.fire_size.toFixed(2)} acres</Popup>
          </Circle>
        ))}
        <Marker position={startPoint}><Popup>Start Point</Popup></Marker>
        <Marker position={endPoint}><Popup>End Point</Popup></Marker>
        {route && <Polyline positions={route} color="blue" />}
      </MapContainer>
    </div>
  );
}

export default App;