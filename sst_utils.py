import numpy as np
import pandas as pd
import xarray as xr
import os

from common import save_data


def round_to_nearest_half(x):
    """
    Rounds the input value to the nearest half.

    Parameters:
        x (float): The input value to be rounded.
    Returns:
        float: The value rounded to the nearest half.
    """
    return np.round(x * 2) / 2


def convert_to_datetime(date):
    """
    Converts a given date to a Pandas Timestamp object and normalizes it to midnight.

    Parameters:
        date (str or datetime-like): The date to be converted and normalized.
    Returns:
        Timestamp: The normalized Pandas Timestamp object.
    """
    date = pd.to_datetime(date)  # convert the string to pd.Timestamp object
    date = date.normalize()  # normalize timestamp

    return date


def extract_date(dataset):
    """
    Extracts the date from a dataset and converts it to a normalized datetime object.

    Parameters:
        dataset (xarray.Dataset): The dataset from which to extract the date.
    Returns:
        Timestamp: The normalized date extracted from the dataset.
    """
    date = dataset.attrs['time_coverage_start']  # extract the date from the dataset

    return convert_to_datetime(date)


def generate_coordinates_filter_query(coordinates):
    """
    Generates a filter query for the given coordinates.

    Parameters:
        coordinates (list): A list of latitude and longitude coordinates.
    Returns:
        None: Placeholder function (currently does not return anything).
    """
    for point in coordinates:
        pass


def filter_coordinates(df):
    """
        Filters a DataFrame based on latitude and longitude bounds for the region of interest.

        Parameters:
            df (DataFrame): The input DataFrame containing 'lat' and 'lon' columns.
        Returns:
            DataFrame: A filtered DataFrame with latitude between -20 and 20, and longitude between 130E and 180 or -80W.
        """
    # filtering lat[20S; 20N]
    df = df[(df['lat'] >= -20) & (df['lat'] <= 20)]

    # filtering long[130E -> 180 -> -80 W]
    df = df[(df.lon >= -180) & (df.lon <= -80) | (df.lon >= 130) & (df.lon <= 180)]

    return df


def dataset_to_csv(filename, array):
    """
    Concatenates multiple datasets into a single CSV file by extracting Sea Surface Temperature (SST)
    and joining the date index. The data is filtered and cleaned before saving.

    Parameters:
        filename (str): The name of the output CSV file.
        array (list of xarray.Datasets): A list of datasets to concatenate and process.
    Returns:
        DataFrame: The combined DataFrame containing SST data for all datasets.
    """

    complete_data = pd.DataFrame([])  # creating empty dataframe

    # extracting to pandas dataframe
    for dataset in array:
        df_sst = dataset['sst'].to_dataframe()
        df_sst.reset_index(inplace=True)  # dropping the multiindex
        df_sst = filter_coordinates(df_sst)  # filtering the dataset basis target coordinates
        df_sst = df_sst.dropna()  # dropping na values

        # rounding lats and long to 0.5 deg
        df_sst.lat = df_sst.lat.apply(round_to_nearest_half)
        df_sst.lon = df_sst.lon.apply(round_to_nearest_half)

        df_sst = df_sst.groupby(['lat', 'lon']).sst.mean()  # grouping by lat and long and averaging the sst
        df_sst = df_sst.reset_index()  # dropping the multiindex

        df_sst.sst = df_sst.sst.apply(lambda x: round(x, 1))  # rounding the sst

        df_sst['lon'] = np.where(df_sst['lon'] < 0, df_sst['lon'] + 360, df_sst['lon'])  # adjusting for negative long
        df_sst.sort_values(['lat', 'lon'])

        ds_date = extract_date(dataset)
        df_sst.index = pd.Index([ds_date] * len(df_sst))
        complete_data = pd.concat([complete_data, df_sst])

    save_data(complete_data, filename)

    return complete_data


def read_files(directory_name, ext):
    """
    Reads all files with a given extension from a directory and returns a list of the datasets.

    Parameters:
        directory_name (str): The directory to search for files.
        ext (str): The file extension ('csv' or 'nc') to filter files by.
    Returns:
        list: A list of datasets (xarray.Dataset for 'nc' files, or DataFrames for 'csv' files).
    """
    result = []

    # Get the full path of the directory
    current_dir = os.getcwd()
    directory_path = os.path.join(current_dir, directory_name)

    if not os.path.isdir(directory_path):  # Check if the directory exists
        print(f'The directory {directory_name} does not exist in the current working directory.')
        return

    for root, dirs, files in os.walk(directory_name):  # Walk through the directory
        for file in files:
            if file.lower().endswith(f'.{ext}'):
                full_path = os.path.join(root, file)
                try:
                    if ext == 'nc':
                        with xr.open_dataset(f'{full_path}', engine='netcdf4') as nc_file:
                            result.append(nc_file)
                    elif ext == 'csv':
                        result.append(pd.read_csv(f'{full_path}', header=None))
                except Exception as e:
                    print(f'Failed to read {full_path}: {e}')
    return result


def clean_csvs(filename, array, start_date):
    """
    Cleans and prepares a list of DataFrames for plotting and analysis, concatenating them into a single CSV file.

    Parameters:
        filename (str): The name of the output CSV file.
        array (list of DataFrames): A list of DataFrames to clean and process.
        start_date (str): The start date for the DataFrame collection in the format 'YYYY-MM'.
    Returns:
        DataFrame: The combined and cleaned DataFrame.
    """

    complete_data = pd.DataFrame([])  # creating empty dataframe

    # creating the range of the latitude and longitude
    lat = np.arange(-90, 90, 0.5)
    lon = np.arange(-180, 180, 0.5)

    start_date = convert_to_datetime(start_date)

    for i, df in enumerate(array):
        # renaming values to math the longitude and adding column matching the latitude
        df.columns = lon
        df['lat'] = lat

        # transposing the longitude to column and extracting the SST to a separate column
        df = df.melt(id_vars='lat', var_name='lon', value_name='sst')

        # filtering the coordinates dataset
        df = filter_coordinates(df)
        df = df.reset_index(drop=True)

        df['lon'] = np.where(df['lon'] < 0, df['lon'] + 360, df['lon'])  # adjusting for negative long
        df.sort_values(['lat', 'lon'])

        df.sst = df.sst.apply(lambda x: None if x > 100 else x)  # cleaning the non-valid data
        df = df.dropna()

        df_date = start_date + pd.DateOffset(months=i)
        df.index = pd.Index([df_date] * len(df))

        complete_data = pd.concat([complete_data, df])

    save_data(complete_data, filename)

    return complete_data


def csv_from_avhrp_csv(directory_name, output_file, start_date):
    """
    Processes and cleans multiple CSV files from a directory, then compiles the data into a single CSV file.

    Parameters:
        directory_name (str): The directory containing the input CSV files.
        output_file (str): The name of the output compiled CSV file.
        start_date (str): The start date for the DataFrame collection in the format 'YYYY-MM'.
    Returns:
        DataFrame: The compiled DataFrame containing data from all input CSV files.
    """

    dataset_array = read_files(directory_name, ext='csv')
    compiled_data = clean_csvs(output_file, dataset_array, start_date)

    return compiled_data


def csv_from_aqua_modis_dataset(directory_name, output_file):
    """
    Processes and cleans multiple netCDF files from a directory, then compiles the data into a single CSV file.

    Parameters:
        directory_name (str): The directory containing the input netCDF files.
        output_file (str): The name of the output compiled CSV file.
    Returns:
        DataFrame: The compiled DataFrame containing data from all input netCDF files.
    """
    dataset_array = read_files(directory_name, ext='nc')
    compiled_data = dataset_to_csv(output_file, dataset_array)

    return compiled_data


def create_urls():
    """
        Generates URLs for downloading AQUA MODIS SST data from the NASA Ocean Data server.

        Returns:
            list: A list of URLs for downloading SST data from January 2002 to December 2022.
        """
    result = []

    base_url = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/AQUA_MODIS.{start_date}_{end_date}.L3m.MO.SST.sst.9km.nc"
    # https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/AQUA_MODIS.20230101_20230131.L3m.MO.SST.sst.9km.nc

    start = pd.to_datetime('20020101')
    offset = pd.DateOffset(months=1)
    day = pd.Timedelta(days=1)

    while True:
        month_start = start.strftime('%Y%m%d')
        start += offset
        month_end = (start - day).strftime('%Y%m%d')

        url = base_url.format(start_date=month_start, end_date=month_end)
        result.append(url)

        with open('urls.txt', 'a') as file:
            file.write(url + '\n')

        if start.year == 2023:
            break

    return result
