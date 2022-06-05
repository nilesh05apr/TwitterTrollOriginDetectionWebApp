import pandas as pd

cities = pd.read_csv("citydata/cities15000.txt", sep="\t", header=None)
city_df_column_names = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", "feature class", "feature code",
                        "country code", "cc2", "admin1 code", "admin2 code", "admin3 code", "admin4 code", "population", "elevation",
                        "dem", "timezone", "modification date"]
cities.columns = city_df_column_names
cities = cities[["name", "asciiname", "alternatenames", "latitude", "longitude", "country code", "timezone"]].copy()

def check_if_loc_exists(user_location):
    user_location = user_location.split(",")[0]
    #print(user_location)
    if not user_location:
        return 0, 0
    else:
        for index,row in cities.iterrows():
            if (user_location in row["name"]) or (user_location in row["asciiname"]) or (user_location in str(row["alternatenames"]).split(",")):
                return row["latitude"], row["longitude"]
        return 0, 0

