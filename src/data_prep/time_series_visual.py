import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union
import seaborn as sns

def _prepare_timeseries_data(df: pd.DataFrame, index_col: str) -> tuple[pd.DataFrame, List[str], np.ndarray, List[str], List[str]]:
    """
    Prepare data and styling elements for time series visualization
    
    Args:
        df: DataFrame with time series data
        index_col: Name of the index/time column (can be datetime or numeric)
    
    Returns:
        Tuple of (prepared_df, value_columns, colors, linestyles, markers)
    """
    df_plot = df.copy()
    
    # Check if index is numeric (hour of year) or datetime-like
    if pd.api.types.is_numeric_dtype(df_plot[index_col]):
        # Keep as numeric index for hour-of-year data
        df_plot = df_plot.set_index(index_col)
    else:
        # Try to convert to datetime
        try:
            df_plot[index_col] = pd.to_datetime(df_plot[index_col])
            df_plot = df_plot.set_index(index_col)
        except (ValueError, TypeError):
            # If conversion fails, keep as is but set as index
            df_plot = df_plot.set_index(index_col)
    
    value_cols = [col for col in df_plot.columns]
    n_series = len(value_cols)
    
    if n_series == 0:
        raise ValueError("No value columns found in DataFrame")
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_series)) # type: ignore
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
    
    return df_plot, value_cols, colors, linestyles, markers


def _create_overlapping_plot(df_plot: pd.DataFrame, value_cols: List[str], colors: np.ndarray, 
                           linestyles: List[str], markers: List[str], figsize: tuple, 
                           title: Optional[str], index_col: str) -> plt.Figure: # type: ignore
    """
    Create overlapping time series plot
    
    Args:
        df_plot: Prepared DataFrame with datetime index
        value_cols: List of column names to plot
        colors: Array of colors for series
        linestyles: List of line styles
        markers: List of marker styles
        figsize: Figure size
        title: Plot title
        index_col: Name of original index column for labeling
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, col in enumerate(value_cols):
        ax.plot(df_plot.index, df_plot[col],
               color=colors[i % len(colors)],
               linestyle=linestyles[i % len(linestyles)],
               marker=markers[i % len(markers)],
               markersize=3,
               markevery=max(1, len(df_plot) // 50),
               linewidth=2,
               alpha=0.7,
               label=col)
    
    ax.set_xlabel(f'{index_col}', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def _create_separate_plots(df_plot: pd.DataFrame, value_cols: List[str], colors: np.ndarray,
                         linestyles: List[str], markers: List[str], figsize: tuple,
                         title: Optional[str], index_col: str) -> plt.Figure: # type: ignore
    """
    Create separate subplot for each time series
    
    Args:
        df_plot: Prepared DataFrame with datetime index
        value_cols: List of column names to plot
        colors: Array of colors for series
        linestyles: List of line styles
        markers: List of marker styles
        figsize: Figure size
        title: Main title
        index_col: Name of original index column for labeling
    
    Returns:
        matplotlib Figure object
    """
    n_series = len(value_cols)
    n_rows = int(np.ceil(n_series / 2)) if n_series > 1 else 1
    n_cols = 2 if n_series > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_series == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(value_cols):
        ax = axes[i]
        
        ax.plot(df_plot.index, df_plot[col], 
               color=colors[i % len(colors)],
               linestyle=linestyles[i % len(linestyles)],
               marker=markers[i % len(markers)],
               markersize=2,
               markevery=max(1, len(df_plot) // 100),
               linewidth=1.5,
               alpha=0.8,
               label=col)
        
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(n_series, len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.02, f'{index_col}', ha='center', fontsize=12)
    
    return fig

def visualize_ts_dataframe(df: pd.DataFrame, 
                        index_col: str,
                        separate_plots: bool = False,
                        figsize: tuple = (12, 8),
                        title: Optional[str] = None,
                        style: str = 'seaborn-v0_8') -> plt.Figure: # type: ignore
    """
    Visualize time series data with options for overlapping or separate plots
    
    Args:
        df: DataFrame with time series data
        index_col: Name of the index/time column
        separate_plots: If True, create separate subplot for each series
        figsize: Figure size as (width, height)
        title: Main title for the plot(s)
        style: Matplotlib style to use
    
    Returns:
        matplotlib Figure object
    """
    plt.style.use(style)
    
    df_plot, value_cols, colors, linestyles, markers = _prepare_timeseries_data(df, index_col)
    
    if separate_plots:
        fig = _create_separate_plots(df_plot, value_cols, colors, linestyles, markers, 
                                   figsize, title, index_col)
    else:
        fig = _create_overlapping_plot(df_plot, value_cols, colors, linestyles, markers,
                                     figsize, title, index_col)
    
    plt.tight_layout()
    return fig

def interactive_timeseries_visualizer(df: pd.DataFrame, index_col: str) -> None:
    """
    Interactive wrapper that asks user preferences and visualizes accordingly
    
    Args:
        df: DataFrame with time series data
        index_col: Name of the index/time column
    """
    # Get basic info about the data
    value_cols = [col for col in df.columns if col != index_col]
    n_series = len(value_cols)
    
    print(f"Found {n_series} time series columns: {value_cols}")
    print(f"Time range: {df[index_col].min()} to {df[index_col].max()}")
    print(f"Total data points: {len(df)}")
    
    # Ask user preferences
    separate = input("\nWould you like separate plots for each time series? (y/n): ").lower().strip()
    separate_plots = separate in ['y', 'yes', 'true', '1']
    
    # Optional customizations
    custom_title = input("Enter a title for the plot (or press Enter for default): ").strip()
    title = custom_title if custom_title else f"Time Series Visualization - {', '.join(value_cols)}"
    
    # Create visualization
    fig = visualize_ts_dataframe(
        df=df,
        index_col=index_col,
        separate_plots=separate_plots,
        title=title
    )
    
    plt.show()
    
    return fig # type: ignore
