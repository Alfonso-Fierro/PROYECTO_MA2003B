import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from typing import Tuple, Optional
import warnings
import matplotlib.pyplot as plt
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

class OrnsteinUhlenbeckModel:
    """
    Ornstein-Uhlenbeck process modeling and parameter estimation
    
    The OU process follows: dX_t = θ(μ - X_t)dt +  .σdW_t
    where:
    - θ (theta): mean reversion speed
    - μ (mu): long-term mean level
    - σ (sigma): volatility parameter
    """
    
    def __init__(self):
        self.theta: Optional[float] = None
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None
        self.fitted: bool = False
        
    def estimate_parameters_mle(self, data: np.ndarray, dt: float = 1.0) -> Tuple[float, float, float]:
        """
        Estimate OU parameters using Maximum Likelihood Estimation
        
        Args:
            data: Time series data as numpy array
            dt: Time step (default 1.0 for unit intervals)
            
        Returns:
            Tuple of (theta, mu, sigma)
        """
        data = np.array(data)
        n = len(data)
        
        if n < 3:
            raise ValueError("Need at least 3 data points for estimation")
        
        # Differences for MLE
        x = data[:-1]
        y = data[1:]
        
        # MLE estimation using discrete approximation
        def negative_log_likelihood(params):
            theta, mu, sigma = params
            
            # Avoid numerical issues
            if theta <= 0 or sigma <= 0:
                return np.inf
                
            # Expected value and variance for discrete OU
            exp_theta_dt = np.exp(-theta * dt)
            conditional_mean = mu + (x - mu) * exp_theta_dt
            conditional_var = (sigma**2 / (2 * theta)) * (1 - exp_theta_dt**2)
            
            if conditional_var <= 0:
                return np.inf
                
            # Log-likelihood
            residuals = y - conditional_mean
            log_likelihood = -0.5 * n * np.log(2 * np.pi * conditional_var) - \
                           0.5 * np.sum(residuals**2) / conditional_var
            
            return -log_likelihood
        
        # Initial parameter guesses
        sample_mean = np.mean(data)
        sample_var = np.var(data)
        initial_theta = 0.1
        initial_mu = sample_mean
        initial_sigma = np.sqrt(2 * initial_theta * sample_var)
        
        # Optimization bounds
        bounds = [(1e-6, 10), (data.min() - sample_var, data.max() + sample_var), (1e-6, 10)]
        
        try:
            result = optimize.minimize(
                negative_log_likelihood,
                x0=[initial_theta, initial_mu, initial_sigma],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                self.theta, self.mu, self.sigma = result.x
                self.fitted = True
                return self.theta, self.mu, self.sigma # type: ignore
            else:
                raise RuntimeError("Optimization failed")
                
        except Exception as e:
            warnings.warn(f"MLE estimation failed: {e}. Using method of moments.")
            return self.estimate_parameters_moments(data, dt)
    
    def estimate_parameters_moments(self, data: np.ndarray, dt: float = 1.0) -> Tuple[float, float, float]:
        """
        Estimate OU parameters using method of moments
        
        Args:
            data: Time series data
            dt: Time step
            
        Returns:
            Tuple of (theta, mu, sigma)
        """
        data = np.array(data)
        
        # Sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data)
        
        # Lag-1 autocorrelation
        if len(data) > 1:
            x = data[:-1]
            y = data[1:]
            autocorr = np.corrcoef(x, y)[0, 1]
        else:
            autocorr = 0.5
        
        # Method of moments estimators
        if autocorr > 0:
            self.theta = -np.log(max(autocorr, 1e-6)) / dt
        else:
            self.theta = 0.1  # Default fallback
            
        self.mu = sample_mean # type: ignore
        
        if self.theta > 0: # type: ignore
            self.sigma = np.sqrt(2 * self.theta * sample_var) # type: ignore
        else:
            self.sigma = np.sqrt(sample_var)
            
        self.fitted = True
        return self.theta, self.mu, self.sigma # type: ignore
    
    def fit(self, data: np.ndarray, dt: float = 1.0, method: str = 'mle') -> 'OrnsteinUhlenbeckModel':
        """
        Fit the OU model to data
        
        Args:
            data: Time series data
            dt: Time step
            method: 'mle' or 'moments'
            
        Returns:
            Self for chaining
        """
        if method == 'mle':
            self.estimate_parameters_mle(data, dt)
        elif method == 'moments':
            self.estimate_parameters_moments(data, dt)
        else:
            raise ValueError("Method must be 'mle' or 'moments'")
            
        return self
    
    def simulate(self, T: int, dt: float = 1.0, x0: Optional[float] = None) -> np.ndarray:
        """
        Simulate OU process
        
        Args:
            T: Number of time steps
            dt: Time step size
            x0: Initial value (if None, uses mu)
            
        Returns:
            Simulated path
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
            
        if x0 is None:
            x0 = self.mu
            
        # Euler-Maruyama scheme
        path = np.zeros(T)
        path[0] = x0
        
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, T):
            drift = self.theta * (self.mu - path[i-1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.normal()
            path[i] = path[i-1] + drift + diffusion
            
        return path
    
    def log_likelihood(self, data: np.ndarray, dt: float = 1.0) -> float:
        """
        Calculate log-likelihood of data given fitted parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        data = np.array(data)
        n = len(data) - 1
        
        x = data[:-1]
        y = data[1:]
        
        exp_theta_dt = np.exp(-self.theta * dt) # type: ignore
        conditional_mean = self.mu + (x - self.mu) * exp_theta_dt
        conditional_var = (self.sigma**2 / (2 * self.theta)) * (1 - exp_theta_dt**2) # type: ignore
        
        residuals = y - conditional_mean
        log_likelihood = -0.5 * n * np.log(2 * np.pi * conditional_var) - \
                        0.5 * np.sum(residuals**2) / conditional_var
        
        return log_likelihood
    
    def aic(self, data: np.ndarray, dt: float = 1.0) -> float:
        """Calculate Akaike Information Criterion"""
        return 2 * 3 - 2 * self.log_likelihood(data, dt)  # 3 parameters
    
    def bic(self, data: np.ndarray, dt: float = 1.0) -> float:
        """Calculate Bayesian Information Criterion"""
        n = len(data) - 1
        return np.log(n) * 3 - 2 * self.log_likelihood(data, dt)
    
    def half_life(self) -> float:
        """Calculate mean reversion half-life"""
        if not self.fitted or self.theta <= 0: # type: ignore
            return np.inf
        return np.log(2) / self.theta
    
    def summary(self) -> dict:
        """Return summary of fitted parameters"""
        if not self.fitted:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "theta": self.theta,
            "mu": self.mu, 
            "sigma": self.sigma,
            "half_life": self.half_life(),
            "mean_reversion_speed": self.theta,
            "long_term_mean": self.mu,
            "volatility": self.sigma
        }

def analyze_ou_process(data: pd.DataFrame, 
                      index_col: str, 
                      value_col: str,
                      dt: float = 1.0,
                      plot: bool = True) -> dict:
    """
    Complete analysis of time series as OU process
    
    Args:
        data: DataFrame with time series
        index_col: Name of index column
        value_col: Name of value column
        dt: Time step
        plot: Whether to create diagnostic plots
        
    Returns:
        Dictionary with analysis results
    """
    # Extract data
    series_data = data[value_col].dropna().values
    
    # Fit OU model
    ou_model = OrnsteinUhlenbeckModel()
    ou_model.fit(series_data, dt=dt, method='mle') # type: ignore
    
    # Generate statistics
    results = ou_model.summary()
    results['aic'] = ou_model.aic(series_data, dt) # type: ignore
    results['bic'] = ou_model.bic(series_data, dt) # type: ignore
    results['log_likelihood'] = ou_model.log_likelihood(series_data, dt) # type: ignore
    
    # Simulate for comparison
    simulated = ou_model.simulate(len(series_data), dt=dt, x0=series_data[0])
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original vs simulated
        axes[0, 0].plot(series_data, label='Original', alpha=0.7)
        axes[0, 0].plot(simulated, label='Simulated OU', alpha=0.7)
        axes[0, 0].axhline(y=ou_model.mu, color='red', linestyle='--', label=f'Long-term mean: {ou_model.mu:.2f}')
        axes[0, 0].set_title('Original vs Simulated OU Process')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals analysis
        x = series_data[:-1]
        y = series_data[1:]
        exp_theta_dt = np.exp(-ou_model.theta * dt) # type: ignore
        predicted = ou_model.mu + (x - ou_model.mu) * exp_theta_dt # type: ignore
        residuals = y - predicted
        
        axes[0, 1].scatter(predicted, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # QQ plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelation function
        def autocorr(x, max_lag=50):
            n = len(x)
            x = x - np.mean(x)
            autocorrs = np.correlate(x, x, mode='full')
            autocorrs = autocorrs[n-1:]
            autocorrs = autocorrs / autocorrs[0]
            return autocorrs[:min(max_lag, len(autocorrs))]
        
        lags = range(min(50, len(series_data)//4))
        acf_values = autocorr(series_data, max_lag=len(lags))
        theoretical_acf = [np.exp(-ou_model.theta * lag * dt) for lag in lags] # type: ignore
        
        axes[1, 1].plot(lags, acf_values[:len(lags)], 'o-', label='Empirical ACF', markersize=3)
        axes[1, 1].plot(lags, theoretical_acf, '--', label='Theoretical OU ACF')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results