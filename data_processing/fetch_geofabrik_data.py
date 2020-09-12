import pandas as pd
import time 
import typing
import requests

import os 
import sys 
import wget

# Paths
ROOT = "../"
BLOCK_PATH = os.path.join(ROOT, "data", "blocks")    # Africa
GEOJSON_PATH = os.path.join(ROOT, "data", "geojson") #"../data/geojson/Africa"
GADM_GEOJSON_PATH = os.path.join(ROOT, "data", "geojson_gadm") #"../data/geojson_gadm/Africa"
GEOFABRIK_PATH = os.path.join(ROOT, "data", "input")

TRANS_TABLE = pd.read_csv(os.path.join(ROOT, "data_processing", 'country_codes.csv'))


#http://download.geofabrik.de/africa/algeria-latest.osm.pbf
def urlexists_stream(uri: str) -> bool:
	'''
	Tests whether a URL is a valid address 
	'''
    try:
        with requests.get(uri, stream=True) as response:
            try:
                response.raise_for_status()
                return True
            except requests.exceptions.HTTPError:
                return False
    except requests.exceptions.ConnectionError:
        return False

def make_url(geo_name: str, geo_region: str) -> str:
	'''
	Simple helper to construct the geofabrik download URL given the name and region
	'''

    url = "http://download.geofabrik.de/{}/{}-latest.osm.pbf".format(geo_region, geo_name)

    if geo_name == "antarctica":
        url = "http://download.geofabrik.de/antarctica-latest.osm.pbf"

    elif geo_name == "puerto-rico":
        url = "http://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf"

    return url 

def download_data(geofabrik_name: str, geofabrik_region: str):
	'''
	Given a geofabrik country name and the corresponding region, downloads the 
	geofabrik pbf file which contains all OSM data for that country. Checks whether
	the data has already been downloaded
	'''

    outfile = geofabrik_name + "-latest.osm.pbf"


    # Check that we haven't already downloaded it
    output_path = os.path.join(GEOFABRIK_PATH, geofabrik_region.title())
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if os.path.isfile(os.path.join(output_path, outfile)):
        print("\nWe have geofabrik data for {} -- see: {}\n".format(geofabrik_name, output_path))

    else:
        url = make_url(geofabrik_name, geofabrik_region)

        if urlexists_stream(url):
            wget.download(url, os.path.join(output_path, outfile))
            print("\nSuccesfully downloaded geofabrik data for {}".format(geofabrik_name))
        else:
            print("\ngeofabrik_name = {} or geofabrik_region = {} are wrong\n".format(geofabrik_name, geofabrik_region))


if __name__ == "__main__":

    TRANS_TABLE = TRANS_TABLE[TRANS_TABLE['geofabrik_NA'] != 1]

    names = TRANS_TABLE['geofabrik_name']
    regions = TRANS_TABLE['geofabrik_region']

    for geofabrik_name, geofabrik_region in zip(names, regions):

        download_data(geofabrik_name, geofabrik_region)


