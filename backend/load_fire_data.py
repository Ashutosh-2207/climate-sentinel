# load_fire_data.py - FINAL KAGGLE VERSION
# Reads data from the reliable "1.88 Million US Wildfires" Kaggle dataset

import pandas as pd
import sqlite3

# --- Configuration ---
# THIS FILENAME IS CORRECT and matches the Kaggle download you have.
FIRE_DATABASE_FILE = 'FPA_FOD_20170508.sqlite'

# The name of the table inside the database that contains the fire data.
FIRE_DATA_TABLE = 'Fires'

def get_wildfire_data(year=2015, state='CA'):
    """
    Loads wildfire data from the local USFS SQLite database (from Kaggle) and filters it.
    
    Args:
        year (int): The year to filter the fires for (e.g., 2015).
        state (str): The state to filter the fires for (e.g., 'CA').

    Returns:
        A list of dictionaries, where each dictionary represents a fire.
    """
    try:
        print(f"Connecting to wildfire database: '{FIRE_DATABASE_FILE}'...")
        conn = sqlite3.connect(FIRE_DATABASE_FILE)

        print(f"Querying data for year={year} and state='{state}'...")
        query = f"SELECT FIRE_YEAR, STATE, LATITUDE, LONGITUDE, FIRE_SIZE FROM {FIRE_DATA_TABLE} WHERE FIRE_YEAR = ? AND STATE = ?"
        
        df = pd.read_sql_query(query, conn, params=(year, state))
        
        conn.close()

        df.rename(columns={
            'LATITUDE': 'latitude',
            'LONGITUDE': 'longitude',
            'FIRE_SIZE': 'fire_size',
            'STATE': 'state'
        }, inplace=True)

        df['confidence'] = 90
        df['bright_ti4'] = 330

        hotspots = df.to_dict('records')

        if not hotspots:
            print(f"[Warning] No wildfires found for {state} in {year}. The list will be empty.")
        else:
            print(f"Successfully loaded {len(hotspots)} wildfires for {state} in {year}.")
        
        return hotspots

    except sqlite3.OperationalError as e:
        print(f"--- FATAL DATABASE ERROR ---")
        print(f"An error occurred: {e}")
        print(f"Please ensure that '{FIRE_DATABASE_FILE}' is in your project directory")
        print(f"and that you have installed the necessary library with 'pip install sqlalchemy'.")
        print(f"--------------------------")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- How to Test This Module Independently ---
if __name__ == "__main__":
    wildfire_data = get_wildfire_data(year=2015, state='CA') 
    
    if wildfire_data:
        print("\n--- Sample of First 5 Wildfires Loaded ---")
        for fire in wildfire_data[:5]:
            print(f"Location: ({fire['latitude']:.4f}, {fire['longitude']:.4f}),",
                  f"State: {fire['state']},",
                  f"Acres: {fire['fire_size']:.2f}")