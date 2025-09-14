import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
from typing import Tuple, Optional, Dict
import warnings
import matplotlib.pyplot as plt

class SeasonalOUModel:
    """
    Seasonal Ornstein-Uhlenbeck model for climate data with constraints
    
    Modifications for climate data:
    1. Log-transform to ensure positivity
    2. Seasonal parameter estimation (4 seasons)
    3. Better initialization for climate variables
    """
    
    def __init__(self):
        self.seasonal_params: Dict[int, Dict[str, float]] = {}
        self.fitted: bool = False
        self.transform: str = 'log'  # log, sqrt, or none
        
    def _transform_data(self, data: np.ndarray, method: str = 'log') -> np.ndarray:
        """Transform data to ensure positivity and better behavior"""
        if method == 'log':
            # Add small constant to avoid log(0)
            return np.log(data + 1e-6)
        elif method == 'sqrt':
            return np.sqrt(np.maximum(data, 0))
        else:
            return data
    
    def _inverse_transform(self, data: np.ndarray, method: str = 'log') -> np.ndarray:
        """Inverse transform back to original scale"""
        if method == 'log':
            return np.exp(data)
        elif method == 'sqrt':
            return data ** 2
        else:
            return data
    
    def _split_by_seasons(self, data: np.ndarray, hours_per_season: int = 2190) -> Dict[int, np.ndarray]:
        """
        Split yearly data into 4 seasons
        
        Args:
            data: Annual time series data
            hours_per_season: Hours per season (8760/4 = 2190)
            
        Returns:
            Dictionary with season number as key and data as value
        """
        seasons = {}
        n_data = len(data)
        
        for season in range(4):
            start_idx = season * hours_per_season
            end_idx = min((season + 1) * hours_per_season, n_data)
            
            if start_idx < n_data:
                seasons[season] = data[start_idx:end_idx]
        
        return seasons
    
    def _estimate_ou_parameters_robust(self, data: np.ndarray, dt: float = 1.0) -> Tuple[float, float, float]:
        """
        Robust OU parameter estimation for climate data
        """
        if len(data) < 10:
            # Fallback for small samples
            return 0.1, np.mean(data), np.std(data) # type: ignore
        
        # Transform data
        transformed_data = self._transform_data(data, self.transform)
        
        def negative_log_likelihood(params):
            theta, mu, sigma = params
            
            if theta <= 0 or sigma <= 0:
                return np.inf
            
            x = transformed_data[:-1]
            y = transformed_data[1:]
            
            exp_theta_dt = np.exp(-theta * dt)
            conditional_mean = mu + (x - mu) * exp_theta_dt
            conditional_var = (sigma**2 / (2 * theta)) * (1 - exp_theta_dt**2)
            
            if conditional_var <= 0:
                return np.inf
            
            residuals = y - conditional_mean
            log_likelihood = -0.5 * len(x) * np.log(2 * np.pi * conditional_var) - \
                           0.5 * np.sum(residuals**2) / conditional_var
            
            return -log_likelihood
        
        # Better initial estimates for climate data
        sample_mean = np.mean(transformed_data)
        sample_var = np.var(transformed_data)
        sample_std = np.std(transformed_data)
        
        # Use lag-1 autocorrelation for theta initialization
        if len(transformed_data) > 1:
            autocorr = np.corrcoef(transformed_data[:-1], transformed_data[1:])[0, 1]
            initial_theta = max(-np.log(max(autocorr, 0.01)) / dt, 0.01)
        else:
            initial_theta = 0.1
        
        initial_mu = sample_mean
        initial_sigma = sample_std * np.sqrt(2 * initial_theta)
        
        # Expanded bounds for climate data
        bounds = [
            (0.001, 5.0),  # theta
            (sample_mean - 3 * sample_std, sample_mean + 3 * sample_std),  # mu
            (0.001, 5 * sample_std)  # sigma
        ]
        
        try:
            result = optimize.minimize(
                negative_log_likelihood,
                x0=[initial_theta, initial_mu, initial_sigma],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                return result.x
            else:
                raise RuntimeError("Optimization failed")
                
        except Exception as e:
            warnings.warn(f"MLE failed: {e}. Using robust fallback.")
            # Robust fallback
            theta = initial_theta
            mu = sample_mean
            sigma = sample_std
            return theta, mu, sigma # type: ignore
    
    def fit_seasonal(self, data: np.ndarray, dt: float = 1.0, 
                    hours_per_season: int = 2190) -> 'SeasonalOUModel':
        """
        Fit seasonal OU model
        
        Args:
            data: Full year time series data
            dt: Time step
            hours_per_season: Hours per season (default 8760/4 = 2190)
        """
        # Split into seasons
        seasonal_data = self._split_by_seasons(data, hours_per_season)
        
        # Fit each season
        for season, season_data in seasonal_data.items():
            if len(season_data) > 5:  # Minimum data requirement
                theta, mu, sigma = self._estimate_ou_parameters_robust(season_data, dt)
                
                self.seasonal_params[season] = { # type: ignore
                    'theta': theta,
                    'mu': mu,
                    'sigma': sigma,
                    'n_points': len(season_data),
                    'original_mean': np.mean(season_data),
                    'original_std': np.std(season_data)
                }
        
        self.fitted = True
        return self
    
    def simulate_seasonal_ensemble(self, n_simulations: int = 1000, 
                                  hours_per_season: int = 2190, 
                                  dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate ensemble of seasonal simulations
        
        Args:
            n_simulations: Number of trajectory simulations
            hours_per_season: Hours per season
            dt: Time step
            
        Returns:
            Dictionary with ensemble statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        total_hours = hours_per_season * 4
        ensemble = np.zeros((n_simulations, total_hours))
        
        for sim in range(n_simulations):
            full_simulation = []
            
            for season in range(4):
                if season in self.seasonal_params:
                    params = self.seasonal_params[season]
                    theta, mu, sigma = params['theta'], params['mu'], params['sigma']
                    
                    # Simulate season in transformed space
                    path = np.zeros(hours_per_season)
                    if len(full_simulation) > 0:
                        # Continue from previous season
                        path[0] = self._transform_data(np.array([full_simulation[-1]]), self.transform)[0]
                    else:
                        path[0] = mu + np.random.normal(0, sigma / np.sqrt(2 * theta))
                    
                    sqrt_dt = np.sqrt(dt)
                    for i in range(1, hours_per_season):
                        drift = theta * (mu - path[i-1]) * dt
                        diffusion = sigma * sqrt_dt * np.random.normal()
                        path[i] = path[i-1] + drift + diffusion
                    
                    # Transform back to original scale
                    season_sim = self._inverse_transform(path, self.transform)
                    full_simulation.extend(season_sim.tolist())
                else:
                    # If no parameters for season, use overall mean
                    overall_mean = np.mean([p['original_mean'] for p in self.seasonal_params.values()])
                    season_sim = np.full(hours_per_season, overall_mean)
                    full_simulation.extend(season_sim.tolist())
            
            ensemble[sim, :] = full_simulation[:total_hours]
        
        # Calculate ensemble statistics
        ensemble_stats = {
            'trajectories': ensemble,
            'mean': np.mean(ensemble, axis=0),
            'median': np.median(ensemble, axis=0),
            'std': np.std(ensemble, axis=0),
            'percentile_5': np.percentile(ensemble, 5, axis=0),
            'percentile_25': np.percentile(ensemble, 25, axis=0),
            'percentile_75': np.percentile(ensemble, 75, axis=0),
            'percentile_95': np.percentile(ensemble, 95, axis=0),
            'min': np.min(ensemble, axis=0),
            'max': np.max(ensemble, axis=0)
        }
        
        return ensemble_stats
    
    def predict_with_uncertainty(self, target_hours: np.ndarray, 
                                n_simulations: int = 1000,
                                hours_per_season: int = 2190) -> Dict[str, np.ndarray]:
        """
        Predict values at specific hours with uncertainty quantification
        
        Args:
            target_hours: Array of hour indices to predict
            n_simulations: Number of simulations for ensemble
            hours_per_season: Hours per season
            
        Returns:
            Dictionary with predictions and uncertainty bounds
        """
        ensemble_stats = self.simulate_seasonal_ensemble(n_simulations, hours_per_season)
        
        # Extract predictions for target hours
        target_indices = np.array(target_hours) - 1  # Convert to 0-based indexing
        valid_indices = target_indices[target_indices < len(ensemble_stats['mean'])]
        
        predictions = {
            'hours': target_hours[target_indices < len(ensemble_stats['mean'])],
            'mean_prediction': ensemble_stats['mean'][valid_indices],
            'median_prediction': ensemble_stats['median'][valid_indices],
            'std_prediction': ensemble_stats['std'][valid_indices],
            'lower_5': ensemble_stats['percentile_5'][valid_indices],
            'lower_25': ensemble_stats['percentile_25'][valid_indices],
            'upper_75': ensemble_stats['percentile_75'][valid_indices],
            'upper_95': ensemble_stats['percentile_95'][valid_indices],
            'prediction_interval_90': (ensemble_stats['percentile_5'][valid_indices], 
                                     ensemble_stats['percentile_95'][valid_indices]),
            'prediction_interval_50': (ensemble_stats['percentile_25'][valid_indices], 
                                     ensemble_stats['percentile_75'][valid_indices])
        }
        
        return predictions
    
    def simulate_seasonal(self, hours_per_season: int = 2190, 
                         dt: float = 1.0) -> np.ndarray:
        """
        Simulate full year with seasonal parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        full_simulation = []
        
        for season in range(4):
            if season in self.seasonal_params:
                params = self.seasonal_params[season]
                theta, mu, sigma = params['theta'], params['mu'], params['sigma']
                
                # Simulate season in transformed space
                path = np.zeros(hours_per_season)
                if len(full_simulation) > 0:
                    # Continue from previous season
                    path[0] = self._transform_data(np.array([full_simulation[-1]]), self.transform)[0]
                else:
                    path[0] = mu
                
                sqrt_dt = np.sqrt(dt)
                for i in range(1, hours_per_season):
                    drift = theta * (mu - path[i-1]) * dt
                    diffusion = sigma * sqrt_dt * np.random.normal()
                    path[i] = path[i-1] + drift + diffusion
                
                # Transform back to original scale
                season_sim = self._inverse_transform(path, self.transform)
                full_simulation.extend(season_sim.tolist())
            else:
                # If no parameters for season, use overall mean
                overall_mean = np.mean([p['original_mean'] for p in self.seasonal_params.values()])
                season_sim = np.full(hours_per_season, overall_mean)
                full_simulation.extend(season_sim.tolist())
        
        return np.array(full_simulation)
    
    def summary(self) -> Dict:
        """Return summary of seasonal parameters"""
        if not self.fitted:
            return {"fitted": False}
        
        summary = {"fitted": True, "transform": self.transform, "seasons": {}}
        
        for season, params in self.seasonal_params.items():
            season_names = ["Winter", "Spring", "Summer", "Fall"]
            summary["seasons"][season_names[season]] = {
                "theta": params['theta'],
                "mu": params['mu'],
                "sigma": params['sigma'],
                "original_mean": params['original_mean'],
                "original_std": params['original_std'],
                "half_life": np.log(2) / params['theta'] if params['theta'] > 0 else np.inf,
                "n_points": params['n_points']
            }
        
        return summary

def analyze_seasonal_ou(data: pd.DataFrame,
                       index_col: str,
                       value_col: str,
                       dt: float = 1.0,
                       hours_per_season: int = 2190,
                       n_simulations: int = 1000,
                       plot: bool = True) -> Dict:
    """
    Analyze time series using seasonal OU model with ensemble simulations
    
    Args:
        data: DataFrame with time series
        index_col: Name of index column
        value_col: Name of value column  
        dt: Time step
        hours_per_season: Hours per season
        n_simulations: Number of ensemble simulations
        plot: Whether to create plots
    """
    # Extract data
    series_data = data[value_col].dropna().values
    
    # Ensure positive values for climate data
    if np.any(series_data <= 0): # type: ignore
        print(f"Warning: Found {np.sum(series_data <= 0)} non-positive values. Adding offset.") # type: ignore # type: ignore # type: ignore
        series_data = series_data - np.min(series_data) + 0.1 # type: ignore
    
    # Fit seasonal model
    seasonal_model = SeasonalOUModel()
    seasonal_model.fit_seasonal(series_data, dt=dt, hours_per_season=hours_per_season) # type: ignore
    
    # Generate ensemble predictions
    print(f"Generating {n_simulations} ensemble simulations...")
    ensemble_stats = seasonal_model.simulate_seasonal_ensemble(
        n_simulations=n_simulations, 
        hours_per_season=hours_per_season, 
        dt=dt
    )
    
    # Get model summary
    results = seasonal_model.summary()
    
    # Add ensemble statistics to results
    results['ensemble'] = {
        'n_simulations': n_simulations,
        'mean_trajectory': ensemble_stats['mean'][:len(series_data)],
        'median_trajectory': ensemble_stats['median'][:len(series_data)],
        'uncertainty_bounds': {
            'lower_5': ensemble_stats['percentile_5'][:len(series_data)],
            'upper_95': ensemble_stats['percentile_95'][:len(series_data)],
            'lower_25': ensemble_stats['percentile_25'][:len(series_data)],
            'upper_75': ensemble_stats['percentile_75'][:len(series_data)]
        }
    }
    
    # Calculate model performance metrics
    ensemble_mean = ensemble_stats['mean'][:len(series_data)]
    ensemble_median = ensemble_stats['median'][:len(series_data)]
    
    results['performance'] = {
        'rmse_mean': np.sqrt(np.mean((series_data - ensemble_mean)**2)),
        'rmse_median': np.sqrt(np.mean((series_data - ensemble_median)**2)),
        'mae_mean': np.mean(np.abs(series_data - ensemble_mean)),
        'mae_median': np.mean(np.abs(series_data - ensemble_median)),
        'coverage_90': np.mean((series_data >= ensemble_stats['percentile_5'][:len(series_data)]) & 
                              (series_data <= ensemble_stats['percentile_95'][:len(series_data)])),
        'coverage_50': np.mean((series_data >= ensemble_stats['percentile_25'][:len(series_data)]) & 
                              (series_data <= ensemble_stats['percentile_75'][:len(series_data)]))
    }
    
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # Original vs ensemble mean/median with uncertainty bands
        time_idx = np.arange(len(series_data))
        
        axes[0, 0].fill_between(time_idx, 
                               ensemble_stats['percentile_5'][:len(series_data)],
                               ensemble_stats['percentile_95'][:len(series_data)],
                               alpha=0.2, color='blue', label='90% Prediction Interval')
        axes[0, 0].fill_between(time_idx,
                               ensemble_stats['percentile_25'][:len(series_data)],
                               ensemble_stats['percentile_75'][:len(series_data)],
                               alpha=0.3, color='blue', label='50% Prediction Interval')
        
        axes[0, 0].plot(series_data, label='Original', alpha=0.8, linewidth=1, color='black')
        axes[0, 0].plot(ensemble_mean, label=f'Ensemble Mean ({n_simulations} sims)', 
                       alpha=0.9, linewidth=2, color='red')
        axes[0, 0].plot(ensemble_median, label='Ensemble Median', 
                       alpha=0.9, linewidth=2, color='green', linestyle='--')
        
        # Add seasonal boundaries
        for i in range(1, 4):
            boundary = i * hours_per_season
            if boundary < len(series_data):
                axes[0, 0].axvline(x=boundary, color='orange', linestyle='--', alpha=0.7)
        
        axes[0, 0].set_title(f'Original vs Ensemble Seasonal OU ({n_simulations} trajectories)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sample of individual trajectories
        n_sample_trajectories = min(50, n_simulations)
        sample_indices = np.random.choice(n_simulations, n_sample_trajectories, replace=False)
        
        for i in sample_indices:
            axes[0, 1].plot(ensemble_stats['trajectories'][i, :len(series_data)], 
                           alpha=0.1, color='gray', linewidth=0.5)
        
        axes[0, 1].plot(series_data, label='Original', alpha=0.9, linewidth=2, color='black')
        axes[0, 1].plot(ensemble_mean, label='Ensemble Mean', alpha=0.9, linewidth=2, color='red')
        axes[0, 1].set_title(f'Sample of {n_sample_trajectories} Individual Trajectories')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Seasonal comparison with uncertainty
        season_names = ["Winter", "Spring", "Summer", "Fall"]
        original_means = []
        ensemble_means = []
        ensemble_stds = []
        
        for season in range(4):
            start_idx = season * hours_per_season
            end_idx = min((season + 1) * hours_per_season, len(series_data))
            
            if start_idx < len(series_data):
                original_means.append(np.mean(series_data[start_idx:end_idx])) # type: ignore
                ensemble_means.append(np.mean(ensemble_mean[start_idx:end_idx]))
                ensemble_stds.append(np.std(ensemble_mean[start_idx:end_idx]))
        
        x_pos = np.arange(len(season_names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, original_means, width, label='Original', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, ensemble_means, width, 
                      yerr=ensemble_stds, label='Ensemble Mean ± Std', alpha=0.7, capsize=5)
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].set_title('Seasonal Means: Original vs Ensemble')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(season_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals analysis
        residuals_mean = series_data - ensemble_mean
        residuals_median = series_data - ensemble_median
        
        axes[1, 1].scatter(ensemble_mean, residuals_mean, alpha=0.6, s=1, label='vs Ensemble Mean')
        axes[1, 1].scatter(ensemble_median, residuals_median, alpha=0.6, s=1, label='vs Ensemble Median')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Coverage analysis
        coverage_90 = ((series_data >= ensemble_stats['percentile_5'][:len(series_data)]) & 
                      (series_data <= ensemble_stats['percentile_95'][:len(series_data)]))
        coverage_50 = ((series_data >= ensemble_stats['percentile_25'][:len(series_data)]) & 
                      (series_data <= ensemble_stats['percentile_75'][:len(series_data)]))
        
        axes[2, 0].plot(coverage_90.astype(int), label=f'90% Coverage ({np.mean(coverage_90):.1%})', alpha=0.7)
        axes[2, 0].plot(coverage_50.astype(int), label=f'50% Coverage ({np.mean(coverage_50):.1%})', alpha=0.7)
        axes[2, 0].set_xlabel('Time Index')
        axes[2, 0].set_ylabel('Inside Prediction Interval')
        axes[2, 0].set_title('Prediction Interval Coverage')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Performance metrics visualization
        metrics_names = ['RMSE\n(Mean)', 'RMSE\n(Median)', 'MAE\n(Mean)', 'MAE\n(Median)']
        metrics_values = [results['performance']['rmse_mean'], 
                         results['performance']['rmse_median'],
                         results['performance']['mae_mean'],
                         results['performance']['mae_median']]
        
        bars = axes[2, 1].bar(metrics_names, metrics_values, alpha=0.7, 
                             color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[2, 1].set_ylabel('Error Value')
        axes[2, 1].set_title('Model Performance Metrics')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        print(f"\nModel Performance Summary:")
        print(f"RMSE (Ensemble Mean): {results['performance']['rmse_mean']:.4f}")
        print(f"RMSE (Ensemble Median): {results['performance']['rmse_median']:.4f}")
        print(f"90% Prediction Interval Coverage: {results['performance']['coverage_90']:.1%}")
        print(f"50% Prediction Interval Coverage: {results['performance']['coverage_50']:.1%}")
    
    return results