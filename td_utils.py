import pandas as pd

from common import save_data


def export_name_basin_count(df):
    """
    Adds 'basin' and 'name' columns to the DataFrame, populates them based on non-numeric time entries,
    and assigns corresponding values to rows where 'time' contains numeric values.

    Parameters:
        df (DataFrame): The input DataFrame containing tropical depression data with 'time', 'date', and 'consecutive_count' columns.
    Returns:
        DataFrame: The modified DataFrame with 'basin', 'name', and 'consecutive_count' columns populated.
    """
    # adding additional cols for the type, name
    df['basin'] = pd.Series()
    df['name'] = pd.Series()
    # getting the rows with names
    names = df[~df.time.str.isdigit()]

    for index in names.index:
        basin = str(names.loc[index, 'date'])[:2]
        name = names.loc[index, 'time']
        count = names.loc[index, 'consecutive_count']

        start_index = index + 1
        while df.loc[start_index, 'time'].isdigit():
            df.loc[start_index, 'basin'] = basin
            df.loc[start_index, 'name'] = name
            df.loc[start_index, 'consecutive_count'] = count
            start_index += 1
            if start_index > df.index[-1]:
                break
        df = df.drop(index)
    return df


def clean_ne_td_data(df):
    """
    Cleans and processes the Northeast Pacific tropical depression data by resetting the index, renaming columns,
    and formatting 'time', 'date', 'lat', and 'lon' fields. Also applies filtering and data corrections.

    Parameters:
        df (DataFrame): The input DataFrame with raw tropical depression data.
    Returns:
        DataFrame: The cleaned and processed DataFrame ready for analysis.
    """
    # reset index
    df = df.reset_index()
    # rename columns from 0 to len
    df.columns = [i for i in range(len(df.columns))]
    # drop columns 8 - end
    df = df.drop(columns=[i for i in df.columns[8:]])
    # name the columns
    df.columns = ['date', 'time', 'consecutive_count', 'type_of_depression', 'lat', 'lon', 'max_wind_kn',
                  'min_pressure_mBar']

    # modify the time
    df.time = df.time.astype(str)
    df.time = df.time.apply(lambda x: x.strip())

    # adding columns, removing non-observation entries
    df = export_name_basin_count(df)

    # continue to modify time
    df.time = df.time.apply(lambda x: x[0:2] + ':' + x[2:] + ':00')
    df.time = pd.to_timedelta(df.time)

    # strip the column, drop non-valid values and convert to int
    df.consecutive_count = df.consecutive_count.apply(lambda x: x.strip())
    df = df.drop(df.index[~df.consecutive_count.str.isdigit()])
    df = df.drop(df.index[~df.date.str.isdigit()])
    df.consecutive_count = df.consecutive_count.astype(int)

    # modify date, concatenate with time and set as index
    df.date = pd.to_datetime(df.date)
    df.date = df.date + df.time
    df.index = pd.Index(df.date)

    # drop the time and date
    df = df.drop(columns=['time', 'date'])

    # set the lat and lon
    df.lat = df.lat.apply(convert_lat_lon_to_number)
    df.lon = df.lon.apply(convert_lat_lon_to_number)

    # rearrange cols
    df = df[['basin', 'name', 'consecutive_count', 'type_of_depression', 'lat', 'lon', 'max_wind_kn',
             'min_pressure_mBar']]

    # removing the non-pressure data

    save_data(df, 'data/csv_ready/ne_pacific_td.csv')

    return df


def convert_lat_lon_to_number(value):
    """
    Converts latitude or longitude string values with directional suffixes (N, S, E, W) to float values.
    'S' and 'W' are converted to negative values, with 'W' values adjusted by adding 360 degrees.

    Parameters:
        value (str): The latitude or longitude string to be converted.

    Returns:
        float: The numeric latitude or longitude value.
    """
    # adds '-' for 'S' and 'W' values and adds 360 deg to negative 'W'
    return float(value[:-1]) if (value[-1] == 'N' or value[-1] == 'E') else float(value[:-1]) * -1 + 360


def modify_jma_date(df):
    """
    Modifies the 'date' column in the JMA (Japan Meteorological Agency) dataset by applying the 'jma_date_time' function
    and setting the resulting values as the index.

    Parameters:
        df (DataFrame): The input DataFrame with a 'date' column.

    Returns:
        DataFrame: The DataFrame with modified 'date' values and the 'date' column set as the index.
    """
    df.date = df.date.apply(jma_date_time)
    df.index = pd.Index(df.date)
    df = df.drop(columns='date')

    return df


def cleaning_jma_columns(df):
    """
    Cleans and processes columns in the JMA dataset. It handles missing or invalid values, converts data types,
    and scales latitude and longitude columns.

    Parameters:
        df (DataFrame): The input DataFrame containing JMA tropical depression data.

    Returns:
        DataFrame: The cleaned DataFrame with properly formatted columns.
    """
    # wind column
    df.max_wind_kn = df.max_wind_kn.apply(lambda x: '000' if pd.isna(x) else x)
    df.max_wind_kn = df.max_wind_kn.astype('int')

    # pressure column
    df.min_pressure_mBar = df.min_pressure_mBar.astype(int)

    # latitude
    df.lat = df.lat.astype(int).apply(lambda x: x / 10)

    # longitude
    df.lon = df.lon.astype(int).apply(lambda x: x / 10)

    return df


def clean_jma_data(data_path='data/jma_data/bst_all.txt'):
    """
    Cleans and processes JMA tropical depression data from the provided file. Extracts names, cleans the data,
    modifies dates, and returns the cleaned dataset.

    Parameters:
        data_path (str): The file path to the JMA dataset.

    Returns:
        DataFrame: The cleaned DataFrame with tropical depression data.
    """
    # read the data and extract the header rows
    header_col_space = [5, 4, 3, 4, 4, 1, 20, 8]
    extract_names = pd.read_fwf(data_path, widths=header_col_space, header=None)
    names = extract_names[extract_names[0] == 66666]

    # read the data and extract the header rows. the column space is manually adjusted
    data_col_space = [8, 4, 2, 4, 5, 5, 13]
    full_data = pd.read_fwf(data_path, widths=data_col_space, header=None, dtype='str')

    # create a column to contain the names
    full_data['name'] = pd.Series()

    for index in names.index:
        name = names.loc[index, 7]

        start_index = index + 1
        while True:
            full_data.loc[start_index, 'name'] = name
            start_index += 1
            if start_index > full_data.index[-1]:
                break
            if start_index in names.index:
                break
        full_data = full_data.drop(index)

    col_names = ['date', 'indicator', 'category', 'lat', 'lon', 'min_pressure_mBar', 'max_wind_kn', 'name']
    full_data.columns = col_names
    full_data = full_data.drop(columns='indicator')

    full_data = cleaning_jma_columns(full_data)
    full_data = modify_jma_date(full_data)

    save_data(full_data, 'jma_td.csv')

    return full_data


def jma_date_time(value):
    """
    Converts a string representing date and time in JMA format to a Pandas Timestamp object.
    The format is 'YYMMDDHH', where 'YY' is the year, 'MM' is the month, 'DD' is the day, and 'HH' is the hour.

    Parameters:
        value (str): The date-time string in JMA format.

    Returns:
        Timestamp: The converted Pandas Timestamp object representing the date and time.
    """

    date = value[:-2]
    time = value[-2:]

    time = pd.Timedelta(hours=int(time))

    date = f'{date[:2]}-{date[2:4]}-{date[-2:]}'

    if int(date[:2]) < 51:
        date = '20' + date
    else:
        date = '19' + date

    date = pd.to_datetime(date)

    date += time

    return date


if __name__ == '__main__':
    print(jma_date_time('500522'))
    # Example usage with different formats
    values = ['99022006', '00031112', '13071212']
    converted_dates = [jma_date_time(value) for value in values]
