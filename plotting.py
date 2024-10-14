import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def get_dataframe(path):
    """
    Reads a CSV file into a Pandas DataFrame and converts the index to datetime format.

    Parameters:
        path (str): The file path of the CSV to read.
    Returns:
        DataFrame: A Pandas DataFrame with the index converted to datetime.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)

    return df


def plot_equatorial_pacific(path, cond_name, plot_type='cnt', vmin=None, vmax=None):
    """
    Plots sequential months of Sea Surface Temperature (SST) data for the equatorial Pacific Ocean,
    with each month's data displayed on a separate subplot. The user can choose between a contour map
    or scatter plot for visualization.

    Parameters:
        path (str): Path to the CSV file containing SST data.
        cond_name (str): The name of the event to be plotted (e.g., "El Niño", "La Niña").
        plot_type (str): The type of plot to create, either contour map ['cnt'] (default) or scatter plot ['sct'].
        vmin (float, optional): The minimum scale value for the temperature color range. Defaults to the minimum SST value in the data.
        vmax (float, optional): The maximum scale value for the temperature color range. Defaults to the maximum SST value in the data.
    Returns:
        None: Displays the subplots showing SST data for each month.
    """
    df = get_dataframe(path)

    grouped_df = df.groupby(df.index)

    # Determine the number of subplots (one for each unique date)
    subplot_count = len(grouped_df)

    # Create subplots in a single row
    fig, axs = plt.subplots(1, subplot_count, figsize=(15 * subplot_count, 8))

    # If only one subplot, axs will not be an array, so we convert it to one
    if subplot_count == 1:
        axs = [axs]

    # Determine the common range for the color scale
    all_sst_values = df['sst'].values

    if vmin is None or vmax is None:
        vmin, vmax = np.min(all_sst_values), np.max(all_sst_values)

    for i, (date, data) in enumerate(grouped_df):

        if plot_type == 'sct':
            subplot = axs[i].scatter(data.lon, data.lat, c=data.sst, cmap='jet', alpha=0.7, s=150, vmin=vmin, vmax=vmax)

        else:
            lon_grid = np.linspace(data.lon.min(), data.lon.max(), 100)
            lat_grid = np.linspace(data.lat.min(), data.lat.max(), 100)
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
            sst_grid = griddata((data.lon, data.lat), data.sst, (lon_grid, lat_grid), method='linear')

            subplot = axs[i].contourf(lon_grid, lat_grid, sst_grid, cmap='jet', levels=20, vmin=vmin, vmax=vmax)

        fig.colorbar(subplot, ax=axs[i], label='Temperature')

        x_tick = np.arange(130, 290, 10)
        x_label = [f'{x}°E' if x <= 180 else f'{360 - x}°W' for x in x_tick]

        y_tick = np.arange(-20, 25, 5)
        y_label = [f'{np.abs(x)}°S' if x < 0 else f'{np.abs(x)}°N' for x in y_tick]

        axs[i].add_patch(plt.Rectangle((190, -5), 50, 10, linewidth=2, edgecolor='red', facecolor='none'))

        axs[i].axhline(y=0)

        axs[i].set_xticks(ticks=x_tick)
        axs[i].set_xticklabels(labels=x_label)

        axs[i].set_yticks(ticks=y_tick)
        axs[i].set_yticklabels(labels=y_label)

        axs[i].set_xlim(130, 280)
        axs[i].set_ylim(-20, 20)
        axs[i].set_xlabel('Longitude')
        axs[i].set_ylabel('Latitude')
        axs[i].set_title(f'{cond_name} SST for {pd.to_datetime(date).strftime("%B %Y")}')

    plt.tight_layout()
    plt.show()
