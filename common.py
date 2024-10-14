import geopandas as gpd
import pandas as pd


def save_data(data, filename):
    """
    Saves the provided data to a CSV file in the 'data/csv_ready' directory.

    Parameters:
        data (DataFrame or GeoDataFrame): The data to be saved as a CSV file.
        filename (str): The name of the output file (without the '.csv' extension).

    Returns:
        None: Saves the data to a CSV file and prints an error message if the file cannot be saved.
    """
    try:
        data.to_csv(f'data/csv_ready/{filename}.csv')
    except OSError:
        print('Error saving file')


def extract_coords(row):
    """
    Extracts the coordinates from a geometry object (Polygon, MultiPolygon, Point, or LineString) in a GeoDataFrame.

    Parameters:
        row (GeoSeries): A row from a GeoDataFrame containing a 'geometry' column.

    Returns:
        list: A list of coordinate tuples (lon, lat) extracted from the geometry.
    """
    if row.geometry.type == 'Polygon' or row.geometry.type == 'MultiPolygon':
        # Extract exterior coordinates for Polygon and MultiPolygon
        coords = list(row.geometry.exterior.coords)
    elif row.geometry.type == 'Point':
        coords = [(row.geometry.x, row.geometry.y)]
    else:
        # For LineString or other geometry types
        coords = list(row.geometry.coords)
    return coords


def convert_geodata_to_csv():
    """
    Converts geographic data from a shapefile into a CSV file. The function reads a shapefile using GeoPandas,
    extracts the coordinates from the geometry column (handling Polygon, MultiPolygon, Point, and LineString types),
    and saves the resulting data as a CSV file. It also explodes complex geometries into individual coordinates.

    Steps:
        1. Load the shapefile using GeoPandas.
        2. Explode geometries with multiple parts (e.g., MultiPolygon).
        3. Extract coordinates from geometries and create separate rows for each coordinate.
        4. Save the resulting data to 'data/csv_ready/gdf_csv.csv'.

    Returns:
        None: Prints the resulting GeoDataFrame and saves it as a CSV file.
    """
    # Load the shapefile using geopandas
    gdf = gpd.read_file('data/gshhs/GSHHS_shp/l/GSHHS_l_L1.shp')

    # If the geometry is not a single point but polygons or linestrings, you will need to explode them:
    gdf = gdf.explode(index_parts=True)

    # Convert the geometry to x, y coordinates
    # Apply the function to create a new column with coordinates
    gdf['coords'] = gdf.apply(extract_coords, axis=1)

    # Now, we will explode the coordinates list into separate rows to handle multiple points
    gdf_exploded = gdf.explode('coords', ignore_index=True)

    # Split the coordinate tuples into separate columns
    gdf_exploded[['lon', 'lat']] = pd.DataFrame(gdf_exploded['coords'].tolist(), index=gdf_exploded.index)

    # Drop the now-unnecessary 'coords' column
    gdf_exploded = gdf_exploded.drop(columns=['coords'])

    gdf_exploded.to_csv('data/csv_ready/gdf_csv.csv', index=False)

    print(gdf_exploded)


if __name__ == '__main__':
    convert_geodata_to_csv()
