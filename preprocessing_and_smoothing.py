import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_and_smooth(input_file="data/dataset.csv", output_file="data/processed_dataset.csv", resample_window="1h"):
    print("Loading data...")
    # Read the dataset
    df = pd.read_csv(input_file, low_memory=False)
    
    print("Initial shape:", df.shape)
    
    # Handle missing values
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df = df.dropna(subset=['start_datetime'])
    
    df['zone'] = df['zone'].fillna('Unknown')
    df['junction'] = df['junction'].fillna('Unknown')
    
    # Create incident counts per 6-hour window per zone and junction
    print("Aggregating data...")
    df.set_index('start_datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Group by zone and junction, then resample to 6H
    # We want a count of incidents
    agg_df = df.groupby(['zone', 'junction']).resample(resample_window).size().reset_index(name='incident_count')
    
    print("Aggregated shape:", agg_df.shape)
    
    # Sort for signal processing
    agg_df.sort_values(by=['zone', 'junction', 'start_datetime'], inplace=True)
    
    # Apply Savitzky-Golay filter to smooth incident frequencies
    print("Applying Savitzky-Golay filter...")
    def smooth_group(group):
        # savgol_filter requires window_length > polyorder. 
        # If a group has too few samples, we just return the original counts.
        window_length = min(15, len(group))
        if window_length % 2 == 0:
            window_length -= 1
            
        if window_length > 3:
            group['smoothed_count'] = savgol_filter(group['incident_count'], window_length, polyorder=3)
        else:
            group['smoothed_count'] = group['incident_count']
        return group

    agg_df = agg_df.groupby(['zone', 'junction']).apply(smooth_group)
    
    # In older pandas, apply might return the grouped columns as index. Reset index if needed.
    if 'zone' in agg_df.index.names:
        agg_df = agg_df.reset_index(drop=True)
    
    # Ensure no negative counts after smoothing
    agg_df['smoothed_count'] = agg_df['smoothed_count'].clip(lower=0)
    
    print("Engineering cyclical time features...")
    # Extract hour and day of week
    agg_df['hour'] = agg_df['start_datetime'].dt.hour
    agg_df['day_of_week'] = agg_df['start_datetime'].dt.dayofweek
    
    # Cyclical encoding
    agg_df['hour_sin'] = np.sin(2 * np.pi * agg_df['hour'] / 24.0)
    agg_df['hour_cos'] = np.cos(2 * np.pi * agg_df['hour'] / 24.0)
    
    agg_df['dow_sin'] = np.sin(2 * np.pi * agg_df['day_of_week'] / 7.0)
    agg_df['dow_cos'] = np.cos(2 * np.pi * agg_df['day_of_week'] / 7.0)
    
    print("Encoding categorical spatial variables...")
    zone_le = LabelEncoder()
    junction_le = LabelEncoder()
    
    agg_df['zone_encoded'] = zone_le.fit_transform(agg_df['zone'])
    agg_df['junction_encoded'] = junction_le.fit_transform(agg_df['junction'])
    
    print("Saving processed dataset to", output_file)
    agg_df.to_csv(output_file, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_and_smooth()
