import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime
import os

def calculate_price_trends(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate average price per day until event
    price_trends = df.groupby('days_until_event').agg({
        'price': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    price_trends.columns = ['days_until_event', 'avg_price', 'median_price', 'std_price', 'ticket_count']
    
    # Calculate price volatility (coefficient of variation)
    price_trends['price_volatility'] = price_trends['std_price'] / price_trends['avg_price']
    
    # Sort by days until event
    price_trends = price_trends.sort_values('days_until_event')
    
    # Save to CSV
    price_trends.to_csv(output_file, index=False)
    print(f"Price trends saved to {output_file}")
    return price_trends

def update_visualization_json(price_trends, json_path='streamlit_data/visualization_data.json'):
    # Prepare the event_stats structure for 'All Events'
    event_stats = {
        'All Events': {
            'prices': {int(row['days_until_event']): row['avg_price'] for _, row in price_trends.iterrows()},
            'counts': {int(row['days_until_event']): int(row['ticket_count']) for _, row in price_trends.iterrows()}
        }
    }
    # Load or create the JSON file
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        with open(json_path, 'r') as f:
            viz_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        viz_data = {}
    viz_data['event_stats'] = event_stats
    with open(json_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    print(f"Updated {json_path} with average price trends.")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "data/processed/p17_ga_processed.csv"
    output_file = "data/processed/p17_price_trends.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Calculate price trends
    price_trends = calculate_price_trends(input_file, output_file)
    
    # Update visualization JSON for Streamlit
    update_visualization_json(price_trends) 