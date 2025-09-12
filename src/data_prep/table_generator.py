"""
Module to generate tables for the Linear Mean Reversion Model.
"""
import pandas as pd
from typing import Optional, Dict

def filter_variable_station(df: pd.DataFrame, variable : str, station : str)-> pd.DataFrame:
    """
    Returns the complete time series of the variable for the specified station
    """
    df_c = df.copy()
    df_c = df_c[df_c['estacion'] == station].reset_index()
    df_c = df_c[["date_index", variable]]
    return df_c

def clean_column_pandas(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Map column removing whitespace and converting to lowercase using pandas"""
    df[column_name] = df[column_name].str.replace(' ', '').str.lower()
    return df

def clean_master_table(df_master: pd.DataFrame)-> pd.DataFrame:
    """
    Estructure station names to group information propperly across years
    """
    df_master_v2 = clean_column_pandas(df = df_master,
                                       column_name = "estacion")
    
    return df_master_v2

def split_timeseries_by_years(df: pd.DataFrame, 
                             date_col: str = "date_index", 
                             value_col: Optional[str] = None) -> Dict[int, pd.DataFrame]:
    """
    Split hourly time series into separate DataFrames by year
    
    Args:
        df: DataFrame with datetime and value columns
        date_col: Name of date column
        value_col: Name of value column (if None, uses second column)
    
    Returns:
        Dictionary with year as key and DataFrame as value
    """
    df_work = df.copy()
    
    if value_col is None:
        value_col = df_work.columns[1]
    
    df_work[date_col] = pd.to_datetime(df_work[date_col])
    df_work['year'] = df_work[date_col].dt.year
    
    year_dfs = {}
    for year in sorted(df_work['year'].unique()):
        year_df = df_work[df_work['year'] == year][[date_col, value_col]].copy()
        year_df = year_df.reset_index(drop=True)
        year_dfs[year] = year_df
    
    return year_dfs

def analyze_year_splits(year_dfs: Dict[int, pd.DataFrame], date_col: str = "date_index") -> pd.DataFrame:
    """
    Analyze the year splits to check record counts and date ranges
    
    Args:
        year_dfs: Dictionary of year DataFrames
        date_col: Name of date column
    
    Returns:
        Summary DataFrame with year statistics
    """
    summary_data = []
    
    for year, df in year_dfs.items():
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        record_count = len(df)
        days_covered = (end_date - start_date).days + 1
        hours_per_day = record_count / days_covered if days_covered > 0 else 0
        
        summary_data.append({
            'year': year,
            'records': record_count,
            'start_date': start_date,
            'end_date': end_date,
            'days_covered': days_covered,
            'avg_hours_per_day': round(hours_per_day, 2)
        })
    
    return pd.DataFrame(summary_data)

def create_aligned_year_comparison(year_dfs: Dict[int, pd.DataFrame], 
                                  date_col: str = "date_index",
                                  value_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create aligned comparison preserving hourly structure
    Limits to 8760 hours and uses hour_of_year as explicit column
    
    Args:
        year_dfs: Dictionary of year DataFrames
        date_col: Name of date column
        value_col: Name of value column
    
    Returns:
        DataFrame with hour_of_year column and years as separate columns
    """
    aligned_data = []
    
    for year, df in year_dfs.items():
        df_work = df.copy()
        if value_col is None:
            value_col = df_work.columns[1]
        
        df_work[date_col] = pd.to_datetime(df_work[date_col])
        
        # Create hour of year (1-8760 max)
        year_start = pd.Timestamp(f'{year}-01-01')
        df_work['hour_of_year'] = ((df_work[date_col] - year_start).dt.total_seconds() / 3600).astype(int) + 1
        
        # Filter to max 8760 hours (exclude Feb 29 for leap years)
        df_work = df_work[df_work['hour_of_year'] <= 8760]
        
        # Select only needed columns and rename value column to year
        year_data = df_work[['hour_of_year', value_col]].rename(columns={value_col: value_col+"_"+str(year)})
        aligned_data.append(year_data)
    
    # Merge all years on hour_of_year
    if aligned_data:
        result_df = aligned_data[0]
        for year_data in aligned_data[1:]:
            result_df = result_df.merge(year_data, on='hour_of_year', how='outer')
        result_df = result_df.sort_values('hour_of_year').reset_index(drop=True)
    else:
        result_df = pd.DataFrame()
    
    return result_df

def orchestrate_contaminant_station_table(df_master: pd.DataFrame, contaminant: str, station: str)-> pd.DataFrame:
    """
    Returns the intrahour year 
    """

    df_master_v2 = clean_master_table(df_master)

    df_cont_station = filter_variable_station(df_master_v2, contaminant, station)
    
    dic_cont_anual  = split_timeseries_by_years(df = df_cont_station,
                                         value_col = contaminant)

    df_aligned_cont = create_aligned_year_comparison(year_dfs = dic_cont_anual,
                                               date_col = 'date_index',
                                               value_col= contaminant)
    
    return df_aligned_cont