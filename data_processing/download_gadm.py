import wget
import zipfile
import os
import shutil
import wget
import pandas as pd 
import sys 
import argparse 

# Paths
ROOT = "../"
BLOCK_PATH = os.path.join(ROOT, "data", "blocks")   
GEOJSON_PATH = os.path.join(ROOT, "data", "geojson") 
GADM_PATH = os.path.join(ROOT, "data", "GADM")
GADM_GEOJSON_PATH = os.path.join(ROOT, "data", "geojson_gadm") 
GEOFABRIK_PATH = os.path.join(ROOT, "data", "input")

TRANS_TABLE = pd.read_csv(os.path.join(ROOT, "data_processing", 'country_codes.csv'))

if not os.path.isdir("./zipfiles"):
    os.mkdir("./zipfiles")

def download_gadm_zip(country_code):
    '''
    Just pulls down the country zip file of GADM boundaries
    '''

    url = "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_{}_shp.zip".format(country_code)
    wget.download(url, "./zipfiles")


def process_zip(country_code, replace=False):
    '''
    Just unpacks the GADM country zip file and stores content

    Inputs:
        - replace: (bool) if True will replace contents, if False will skip if 
                          country code has been processed already
    '''

    p = os.path.join("./zipfiles", "gadm36_{}_shp.zip".format(country_code))

    outpath = os.path.join(GADM_PATH, country_code)

    with zipfile.ZipFile(p) as z:
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        z.extractall(outpath)

def update_gadm_data(replace=False):
    '''
    Downloads all the GADM zip files, then unpacks the files

    Inputs:
        - replace: (bool) if True will replace contents, if False will skip if 
                          country code has been processed already

    '''

    df = pd.read_csv(TRANS_TABLE)
    b = ~ df['gadm_name'].isna()
    codes = df[ b ]['gadm_name'].values
    names = df[ b ]['country'].values

    for country_name, country_code in zip(names, codes):
        print("\nProcessing GADM: ", country_name)
        if replace or not os.path.isdir(country_code):
            print("\tdownloading...")
            download_gadm_zip(country_code)
            process_zip(country_code)

        else:
            print("\tskip, file present")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download GADM administrative boundaries globally')
    parser.add_argument("--replace", action='store_true')

    update_gadm_data(replace_boolean)
