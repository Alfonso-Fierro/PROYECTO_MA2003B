import numpy as np
import pandas as pd
import os
from typing import Optional, List
from scipy import interpolate
from sklearn.utils import resample

class TimeSeriesImputer:
    """
    Comprehensive imputation strategies for aligned year comparison DataFrame
    """
    
    def __init__(self, df: pd.DataFrame, hour_col: str = 'hour_of_year'):
        """
        Initialize imputer with aligned DataFrame
        
        Args:
            df: DataFrame from create_aligned_year_comparison
            hour_col: Name of hour column
        """
        self.df = df.copy()
        self.hour_col = hour_col
        self.year_cols = [col for col in df.columns if col != hour_col]
        
    def rolling_mean_imputation(self, column: str, window: int = 24) -> pd.Series:
        """
        Strategy 1: Rolling mean within column (centered window)
        
        Args:
            column: Column name to impute
            window: Window size (default 24 for daily pattern)
        
        Returns:
            Series with imputed values
        """
        series = self.df[column].copy()
        
        # Use centered rolling mean
        rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
        
        # Fill NaN values with rolling mean
        imputed = series.fillna(rolling_mean)
        
        return imputed
    
    def cross_year_median_imputation(self, target_col: str, window: int = 24) -> pd.Series:
        """
        Strategy 2: Median of rolling means across years for same hour
        
        Args:
            target_col: Target column to impute
            window: Window size for rolling means
        
        Returns:
            Series with imputed values
        """
        result = self.df[target_col].copy()
        
        # Calculate rolling means for all years
        rolling_means = {}
        for col in self.year_cols:
            rolling_means[col] = self.df[col].rolling(window=window, center=True, min_periods=1).mean()
        
        # For each missing value, get median of all rolling means at that hour
        missing_mask = result.isna()
        
        for idx in self.df[missing_mask].index:
            hour_means = []
            for col in self.year_cols:
                if not pd.isna(rolling_means[col].iloc[idx]):
                    hour_means.append(rolling_means[col].iloc[idx])
            
            if hour_means:
                result.iloc[idx] = np.median(hour_means)
        
        return result
    
    def rolling_median_imputation(self, column: str, window: int = 24) -> pd.Series:
        """
        Strategy 3: Rolling median within column
        
        Args:
            column: Column name to impute
            window: Window size
        
        Returns:
            Series with imputed values
        """
        series = self.df[column].copy()
        
        # Use centered rolling median
        rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
        
        # Fill NaN values with rolling median
        imputed = series.fillna(rolling_median)
        
        return imputed
    
    def hourly_mean_imputation(self, column: str) -> pd.Series:
        """
        Strategy 4: Mean of same hour across all available years
        
        Args:
            column: Column name to impute
        
        Returns:
            Series with imputed values
        """
        result = self.df[column].copy()
        
        # For each hour, calculate mean across all years
        for hour in self.df[self.hour_col].unique():
            hour_mask = self.df[self.hour_col] == hour
            hour_values = []
            
            # Collect all values for this hour from all years
            for col in self.year_cols:
                values = self.df.loc[hour_mask, col].dropna() # type: ignore
                hour_values.extend(values.tolist())
            
            if hour_values:
                hour_mean = np.mean(hour_values)
                # Fill missing values for this hour in target column
                missing_hour_mask = hour_mask & result.isna()
                result.loc[missing_hour_mask] = hour_mean
        
        return result
    
    def polynomial_interpolation(self, column: str, degree: int = 2) -> pd.Series:
        """
        Strategy 5: Polynomial interpolation using 5 nearest points
        
        Args:
            column: Column name to impute
            degree: Polynomial degree
        
        Returns:
            Series with imputed values
        """
        series = self.df[column].copy()
        missing_indices = series[series.isna()].index
        
        for idx in missing_indices:
            # Find 5 nearest non-NaN values (2 before, 2 after, or adjust if at boundaries)
            nearby_indices = []
            search_radius = 1
            
            while len(nearby_indices) < 5 and search_radius < len(series):
                for offset in range(-search_radius, search_radius + 1):
                    check_idx = idx + offset
                    if (0 <= check_idx < len(series) and 
                        check_idx not in nearby_indices and 
                        not pd.isna(series.iloc[check_idx])):
                        nearby_indices.append(check_idx)
                search_radius += 1
            
            if len(nearby_indices) >= max(2, degree + 1):  # Need at least degree+1 points
                nearby_indices = sorted(nearby_indices[:5])  # Take closest 5
                x_vals = np.array(nearby_indices)
                y_vals = series.iloc[nearby_indices].values # type: ignore
                
                # Fit polynomial
                if len(nearby_indices) > degree:
                    poly_coef = np.polyfit(x_vals, y_vals, degree)
                    interpolated_value = np.polyval(poly_coef, idx)
                    series.iloc[idx] = interpolated_value
        
        return series
    
    def bootstrap_imputation(self, column: str, n_bootstrap: int = 100, 
                           hour_window: int = 2) -> pd.Series:
        """
        Strategy 6: Bootstrap using values from same hour of day across years
        
        Args:
            column: Column name to impute
            n_bootstrap: Number of bootstrap samples
            hour_window: Window around target hour (±hours)
        
        Returns:
            Series with imputed values
        """
        result = self.df[column].copy()
        missing_mask = result.isna()
        
        for idx in self.df[missing_mask].index:
            target_hour = self.df.loc[idx, self.hour_col]
            hour_of_day = target_hour % 24  # type: ignore
            
            # Collect values from similar hours across all years
            bootstrap_pool = []
            
            for col in self.year_cols:
                # Find hours within window
                for h in range(max(1, hour_of_day - hour_window),  # type: ignore
                              min(25, hour_of_day + hour_window + 1)): # type: ignore
                    hour_candidates = self.df[
                        (self.df[self.hour_col] % 24 == h) & 
                        (~self.df[col].isna())
                    ][col]
                    bootstrap_pool.extend(hour_candidates.tolist())
            
            if bootstrap_pool:
                # Perform bootstrap and take mean
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = resample(bootstrap_pool, n_samples=min(len(bootstrap_pool), 10))
                    bootstrap_means.append(np.mean(sample)) # type: ignore
                
                result.iloc[idx] = np.mean(bootstrap_means)
        
        return result
    
    def bootstrap_imputation_df(self, column: str, n_bootstrap: int = 100, 
                           hour_window: int = 2) -> pd.DataFrame:
        """
        Strategy 6: Bootstrap using values from same hour of day across years
        
        Args:
            column: Column name to impute
            n_bootstrap: Number of bootstrap samples
            hour_window: Window around target hour (±hours)
        
        Returns:
            Series with imputed values
        """

        strategy_result = pd.DataFrame({
            'hour_of_year': self.df[self.hour_col],
            column: self.bootstrap_imputation(column)
        })
        
        return strategy_result 

    def apply_all_strategies(self, column: str) -> pd.DataFrame:
        """
        Apply all imputation strategies to a column
        
        Args:
            column: Column name to impute
        
        Returns:
            DataFrame with original and all imputed versions
        """
        strategies_results = pd.DataFrame({
            'hour_of_year': self.df[self.hour_col],
            'original': self.df[column],
            'rolling_mean': self.rolling_mean_imputation(column),
            'cross_year_median': self.cross_year_median_imputation(column),
            'rolling_median': self.rolling_median_imputation(column),
            'hourly_mean': self.hourly_mean_imputation(column),
            'polynomial': self.polynomial_interpolation(column),
            'bootstrap': self.bootstrap_imputation(column)
        })
        
        return strategies_results
