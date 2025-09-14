import pandas as pd

def smooth_to_six_hour_windows(df: pd.DataFrame, 
                              index_col: str, 
                              value_col: str,
                              method: str = 'mean') -> pd.DataFrame:
    """
    Smooth hourly time series data to 6-hour windows
    
    Args:
        df: DataFrame with time series data
        index_col: Name of the index column (hour_of_year or datetime)
        value_col: Name of the value column to smooth
        method: Aggregation method ('mean', 'median', 'max', 'min')
    
    Returns:
        DataFrame with smoothed data (4 values per day instead of 24)
    """
    df_work = df.copy()
    
    # Ensure index is numeric for grouping
    if not pd.api.types.is_numeric_dtype(df_work[index_col]):
        raise ValueError(f"Index column '{index_col}' must be numeric (hour_of_year)")
    
    # Create 6-hour window groups
    # Hour 1-6 -> Window 0, Hour 7-12 -> Window 1, etc.
    df_work['window_group'] = ((df_work[index_col] - 1) // 6).astype(int)
    
    # Calculate window start hour for new index
    df_work['window_start'] = df_work['window_group'] * 6 + 1
    
    # Remove rows with NaN values before aggregation
    df_clean = df_work.dropna(subset=[value_col])
    
    # Group by window and aggregate
    agg_methods = {
        'mean': 'mean',
        'median': 'median', 
        'max': 'max',
        'min': 'min'
    }
    
    if method not in agg_methods:
        raise ValueError(f"Method must be one of: {list(agg_methods.keys())}")
    
    result = df_clean.groupby('window_start').agg({
        value_col: agg_methods[method],
        'window_group': 'first'  # Keep for reference
    }).reset_index()
    
    # Create sequential index starting from 1
    result[index_col] = range(1, len(result) + 1)
    
    # Clean up result
    result = result.rename(columns={
        value_col: f'{value_col}_{method}_6h'
    })
    
    result = result[[index_col, f'{value_col}_{method}_6h']]
    
    return result


def smooth_multiple_series(df: pd.DataFrame,
                          index_col: str,
                          method: str = 'mean') -> pd.DataFrame:
    """
    Smooth multiple time series columns to 6-hour windows
    
    Args:
        df: DataFrame with multiple time series columns
        index_col: Name of the index column
        method: Aggregation method
    
    Returns:
        DataFrame with all series smoothed
    """
    value_cols = [col for col in df.columns if col != index_col]
    
    if not value_cols:
        raise ValueError("No value columns found")
    
    # Process first column to establish index
    result = smooth_to_six_hour_windows(df, index_col, value_cols[0], method)
    
    # Process remaining columns and merge
    for col in value_cols[1:]:
        col_smoothed = smooth_to_six_hour_windows(df, index_col, col, method)
        result = result.merge(col_smoothed, on=index_col, how='outer')
    
    return result



def analyze_smoothing_effect(original_df: pd.DataFrame,
                           smoothed_df: pd.DataFrame, 
                           index_col: str,
                           value_col: str) -> dict:
    """
    Analyze the effect of smoothing on data characteristics
    
    Args:
        original_df: Original hourly data
        smoothed_df: Smoothed 6-hour data
        index_col: Name of index column
        value_col: Name of value column (use base name without suffix)
    
    Returns:
        Dictionary with smoothing statistics
    """
    # Find smoothed column name
    smoothed_col = None
    for col in smoothed_df.columns:
        if col.startswith(value_col) and col != index_col:
            smoothed_col = col
            break
    
    if smoothed_col is None:
        raise ValueError(f"Could not find smoothed column for {value_col}")
    
    original_values = original_df[value_col].dropna()
    smoothed_values = smoothed_df[smoothed_col].dropna()
    
    stats = {
        'original_points': len(original_values),
        'smoothed_points': len(smoothed_values),
        'compression_ratio': len(original_values) / len(smoothed_values),
        'original_std': original_values.std(),
        'smoothed_std': smoothed_values.std(),
        'noise_reduction': 1 - (smoothed_values.std() / original_values.std()),
        'original_range': original_values.max() - original_values.min(),
        'smoothed_range': smoothed_values.max() - smoothed_values.min()
    }
    
    return stats

# Example usage and demonstration
def quick_smoothing(sample_df: pd.DataFrame, column: str):
    """Demonstrate the smoothing functionality"""
    
    # Apply smoothing
    smoothed_df = smooth_to_six_hour_windows(sample_df, 'hour_of_year', column, 'max')
    
    # Analyze effect
    stats = analyze_smoothing_effect(sample_df, smoothed_df, 'hour_of_year', column)
    
    print("Smoothing Analysis:")
    print(f"Data points: {stats['original_points']} → {stats['smoothed_points']}")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}:1")
    print(f"Standard deviation: {stats['original_std']:.3f} → {stats['smoothed_std']:.3f}")
    print(f"Noise reduction: {stats['noise_reduction']:.1%}")
    print(f"Range: {stats['original_range']:.2f} → {stats['smoothed_range']:.2f}")
    
    return sample_df, smoothed_df, stats