from datetime import datetime
from pathlib import Path
from writers.writer import Writer
from writers.writer_basic import WriterBasic
from pcs_downloader.downloader import Downloader

## Configure the scrapper here
##############################

# From which year to start the scrapping
start_year = 1995

# If to scrap information for this year.
# This will always update the rider's information and also create the start list if the race hasn't happened yet.
#get_this_year = True

# The path to the different data
data_path = Path('scrapped_data/')
races_path = data_path / 'races'
riders_path = data_path / 'riders'
race_metadata_path = data_path / 'race_metadata.csv'
rider_metadata_path = data_path / 'rider_metadata.csv'
top_rider_year_path = data_path / 'top_riders_year'
# Code starts here
##############################
#year_offset = 0
#if get_this_year:
#    year_offset = 1
#past_years = [x for x in range(start_year,datetime.now().year + year_offset)]

past_years = [x for x in range(start_year,datetime.now().year)]

race_result_io = WriterBasic(races_path)
rider_result_io = WriterBasic(riders_path)
race_metadata = Writer(race_metadata_path, 'race_id')
rider_metadata = Writer(rider_metadata_path, 'rider_slug')
top_rider_year_io = WriterBasic(top_rider_year_path)
error_while_getting_start_list = []

downloader = Downloader(race_result_io, rider_result_io, race_metadata, rider_metadata,top_rider_year_io)

#for target_race_slug in target_races:
for target_race_year in past_years:
    print("Downloading top riders: {}".format(target_race_year))
    # The race will read from disk or download
    target_top_riders_info = downloader.get_top_riders(target_race_year)

    # Get all the riders and rider's races for the target race
    #results = target_race_info['{}_{}'.format(target_race_slug, target_race_year)].get('results')
    results = target_top_riders_info.get('results')


    for _, result in results.iterrows():
        if rider_metadata.get_row(result['rider_slug']) is not None and datetime.now().year != target_race_year:
            continue
        downloader.rider_and_races(result['rider_slug'], target_race_year)

