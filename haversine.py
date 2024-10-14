import numpy as np
import pandas as pd
from geopandas.datasets.naturalearth_creation import gdf
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import haversine_distances
from math import radians


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the Earth
    specified in decimal degrees using the Haversine formula.

    Parameters:
        lon1, lat1: Longitude and latitude of the first point.
        lon2, lat2: Longitude and latitude of the second point.

    Returns:
        Distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of Earth in kilometers (mean radius)
    R = 6371.0088
    return R * c


def compute_kde_values(df, phase, method, gridsize=100, bandwidth=0.1):
    """
    Computes the KDE (Kernel Density Estimate) values for Tropical Depressions using Haversine distance.

    Parameters:
        df (DataFrame): The dataset containing longitude, latitude, and ENSO phase information.
        phase (int): The ENSO phase (-1 for La Niña, 0 for Neutral, 1 for El Niño).
        gridsize (int): The number of grid points along each dimension for KDE computation (default: 100).
        bandwidth (float): Bandwidth for KDE (default: 0.1).

    Returns:
        lon_grid (ndarray): The longitude grid used for KDE.
        lat_grid (ndarray): The latitude grid used for KDE.
        kde_values (ndarray): The computed KDE values over the grid.
    """
    # Subset data for the specific ENSO phase
    subset = df[df.enso == phase]

    # Extract longitude and latitude values
    lon_vals = subset['lon'].values
    lat_vals = subset['lat'].values

    # Define grid boundaries in degrees
    lon_min, lon_max = 100, 300
    lat_min, lat_max = -20, 70
    lon_grid, lat_grid = np.linspace(lon_min, lon_max, gridsize), np.linspace(lat_min, lat_max, gridsize)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Initialize KDE grid
    kde_values = np.zeros(lon_grid.shape)

    if method == 'haversine':
        # Compute KDE using Haversine distance for each grid point
        for i in range(gridsize):
            for j in range(gridsize):
                # Calculate the distance from each data point to the grid point
                distances = haversine(lon_vals, lat_vals, lon_grid[i, j], lat_grid[i, j])

                # Apply Gaussian kernel to distances and sum
                kde_values[i, j] = np.sum(np.exp(-0.5 * (distances / bandwidth) ** 2))

        # Normalize KDE values
        kde_values /= (bandwidth * np.sqrt(2 * np.pi)) * len(lon_vals)

    elif method == 'euclidean':
        kde = gaussian_kde(np.vstack([subset['lon'], subset['lat']]))
        kde_values = kde(np.vstack([lon_grid.ravel(), lat_grid.ravel()])).reshape(gridsize, gridsize)

    return lon_grid, lat_grid, kde_values


def plot_td_density_difference(dfs, method):
    """
    Plots the density differences of Tropical Depressions (TDs) for different ENSO phases
    (El Niño, La Niña, Neutral) across the North Pacific Ocean. It computes the differences
    between KDE values for each phase and creates visual comparisons using heatmaps.
    The function shows the results for both NW Pacific and NE/Central Pacific regions.
    """

    kde_values_jma = {phase: compute_kde_values(dfs[0], phase, method)[2] for phase in range(-1, 2)}
    kde_values_nhc = {phase: compute_kde_values(dfs[1], phase, method)[2] for phase in range(-1, 2)}

    # Subtract KDE values to create difference maps for each region
    diff_maps_jma = {
        'El Niño - La Niña (NW Pacific)': kde_values_jma[1] - kde_values_jma[-1],
        'Neutral - La Niña (NW Pacific)': kde_values_jma[0] - kde_values_jma[-1],
        'Neutral - El Niño (NW Pacific)': kde_values_jma[0] - kde_values_jma[1]
    }

    diff_maps_nhc = {
        'El Niño - La Niña (NE/Central Pacific)': kde_values_nhc[1] - kde_values_nhc[-1],
        'Neutral - La Niña (NE/Central Pacific)': kde_values_nhc[0] - kde_values_nhc[-1],
        'Neutral - El Niño (NE/Central Pacific)': kde_values_nhc[0] - kde_values_nhc[1]
    }

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))

    for i, diff_dt in enumerate([diff_maps_jma, diff_maps_nhc]):
        for idx, (title, diff_map) in enumerate(diff_dt.items()):
            im = axs[i, idx].imshow(diff_map, extent=[100, 300, -20, 70], origin='lower', cmap='coolwarm')
            axs[i, idx].set_title(title)
            axs[i, idx].scatter(gdf.lon, gdf.lat, s=0.5, color='black')

            x_tick = np.arange(100, 300, 25)
            x_label = [f'{x}°E' if x <= 180 else f'{360 - x}°W' for x in x_tick]
            y_tick = np.arange(-20, 70, 20)
            y_label = [f'{np.abs(x)}°S' if x < 0 else f'{np.abs(x)}°N' for x in y_tick]
            axs[i, idx].set_xticks(ticks=x_tick)
            axs[i, idx].set_xticklabels(labels=x_label)
            axs[i, idx].set_yticks(ticks=y_tick)
            axs[i, idx].set_yticklabels(labels=y_label)

            axs[i, idx].set_xlabel('Longitude')
            axs[i, idx].set_ylabel('Latitude')

    plt.tight_layout()
    plt.show()


def get_dfs():
    oni_table = pd.read_csv('data/csv_ready/oni_table.csv', index_col=0)
    oni_temp = pd.read_csv('data/csv_ready/oni_temp.csv', index_col=0)
    oni_table.index = pd.to_datetime(oni_table.index)
    enso_phase = oni_table.groupby(oni_table.index.year)['enso'].apply(lambda x: x.unique()[0])

    jma = pd.read_csv('data/csv_ready/jma_td.csv', index_col=0)
    jma.index = pd.to_datetime(jma.index)
    frequency_jma = jma.groupby(jma.index.year)['name'].nunique()
    frequency_jma = pd.merge(frequency_jma, enso_phase, on='date')
    frequency_jma.columns = ['frequency', 'enso']

    nhc = pd.read_csv('data/csv_ready/ne_pacific_td.csv', index_col=0)
    nhc.index = pd.to_datetime(nhc.index)
    frequency_nhc = nhc.groupby(nhc.index.year)['name'].nunique()
    frequency_nhc = pd.merge(frequency_nhc, enso_phase, on='date')
    frequency_nhc.columns = ['frequency', 'enso']

    nhc_cp = nhc[nhc.basin == 'CP']
    frequency_nhc_cp = nhc_cp.groupby(nhc_cp.index.year)['name'].nunique()
    frequency_nhc_cp = pd.merge(frequency_nhc_cp, enso_phase, on='date')
    frequency_nhc_cp.columns = ['frequency', 'enso']

    nhc_ep = nhc[nhc.basin == 'EP']
    frequency_nhc_ep = nhc_ep.groupby(nhc_ep.index.year)['name'].nunique()
    frequency_nhc_ep = pd.merge(frequency_nhc_ep, enso_phase, on='date')
    frequency_nhc_ep.columns = ['frequency', 'enso']

    frequency_tables = [
        ('NW', frequency_jma),
        ('Central', frequency_nhc_cp),
        ('NE', frequency_nhc_ep)]

    dfs_freq = [frequency_jma, frequency_nhc]

    enso_phase_dt = enso_phase.copy()
    enso_phase_dt.index = pd.to_datetime(enso_phase.index.astype(str))

    merged = pd.merge(jma, enso_phase_dt, left_on=jma.index.year, right_on=enso_phase_dt.index.year, how='left')
    merged = merged.set_index(jma.index)
    jma_enso = merged.drop(columns='key_0')

    merged = pd.merge(nhc, enso_phase_dt, left_on=nhc.index.year, right_on=enso_phase_dt.index.year, how='left')
    merged = merged.set_index(nhc.index)
    nhc_enso = merged.drop(columns='key_0')

    gdf = pd.read_csv('data/csv_ready/gdf_pacific.csv')
    return jma_enso, nhc_enso
