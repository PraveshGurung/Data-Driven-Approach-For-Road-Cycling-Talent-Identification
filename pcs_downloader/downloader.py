import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from writers.writer_basic import WriterBasic
from writers.writer import Writer
from .race_parser import RaceParser
from .rider_metadata_parser import RiderMetaDataParser
from .rider_result_parser import RiderResultPageParser
from .start_list_parser import StartListParser
from .utils import get_page_in_soup
from .top_riders_year_parser import TopRidersYearParser

from pprint import pprint

from copy import deepcopy
import requests


# Stupid error in PCS for this one rider.
def get_alias(rider_slug:str) -> str:
    if rider_slug == 'steve-vermaut':
        return 'stive-vermaut'
    if rider_slug == 'stive-vermaut':
        return 'steve-vermaut'


class Downloader:
    def __init__(self, race_result_io: WriterBasic, rider_result_io: WriterBasic, race_metadata: Writer,
                 rider_metadata: Writer, top_rider_year_io: WriterBasic):
        self.race_results_io = race_result_io
        self.rider_result_io = rider_result_io
        self.race_metadata = race_metadata
        self.rider_metadata = rider_metadata
        self.top_rider_year_io = top_rider_year_io

    def check_races(self):
        errors = []
        length = len(self.race_metadata.data)
        i = 0
        for race_id, row in self.race_metadata.data[self.race_metadata.data['has_results']==True].iterrows():
            i += 1
            if self.race_results_io.read(race_id) is None:
                page_soup = get_page_in_soup(row['url'])

                if page_soup is None:
                    errors.append('Error {}'.format(race_id))
                    continue

                page = RaceParser(page_soup, row['url'], race_id)

                if not page.is_eligible():
                    logging.info("Race is TTT or Hammer - Only getting meta data")
                    continue

                race_results = page.get_results()
                if race_results is None:
                    errors.append("Couldn't get results for {}".format(race_id))
                    continue


                print("{} / {} - {}".format(i, length, i/length))
                self.race_results_io.save(race_id, pd.DataFrame(race_results))

        for error in errors:
            print(error)

    def rider_and_races(self, rider_slug: str, race_year: int, get_this_year=False) -> None:
        print("Getting rider {}".format(rider_slug))
        if rider_slug == 'kaspars-ozers': #pcs being weird
            return
        metadata = self.rider_metadata.get_row(rider_slug)

        if metadata:
            today = datetime.now()
            # if not (metadata['last_update'] <= datetime(today.year, today.month, today.day) and metadata['last_update'] < datetime(race_year, 12, 31)):
            #     return

        url = 'https://www.procyclingstats.com/rider/{}'.format(rider_slug)
        rider_soup = get_page_in_soup(url)
        rider_page = RiderMetaDataParser(rider_soup, url)
        rider_meta_data = rider_page.get_metadata()

        rider_results = pd.DataFrame()
        results_url = 'https://www.procyclingstats.com/rider.php?xseason=&sort=date&race=&km1=&zkm1=&pkm1=equal&limit' \
                      '=200&offset=0&topx=&ztopx=&ptopx=smallerorequal&type=&continent=&pnts=&zpnts=&ppnts=equal' \
                      '&level=&rnk=&zrnk=&prnk=equal&exclude_tt=0&racedate=&zracedate=&pracedate=equal&filter=Filter' \
                      '&id={}&p=results'.format(rider_slug)

        while results_url:
            results_soup = get_page_in_soup(results_url)
            results_page = RiderResultPageParser(results_soup, results_url, rider_slug)
            rider_results = rider_results.append(results_page.get_results(), ignore_index=True)
            results_url = results_page.get_next_url()
        print(rider_meta_data)
        all_race_results = {}
        for i, rider_result in rider_results.iterrows():
            print(rider_result['race_id'])

            # if rider_result['date'].year > 2000:
            #     continue
            if rider_result['has_no_results']:
                continue

            # We keep the all race results in cache (most likely the GC races will come after each other)
            if rider_result['race_id'] not in all_race_results:
                if rider_result['date'].year != 2023: #XXX
                    all_race_results = self.get_race(rider_result['race_slug'], rider_result['date'].year)

            # Didn't get any results for the race_slug / race_year query
            if all_race_results is None:
                logging.warning("Didn't find any race {} from {} for {}".format(rider_result['race_slug'],
                                                                                rider_result['date'].year,
                                                                                rider_meta_data['rider_slug']))
                rider_results.loc[i, 'has_no_results'] = True
                all_race_results = {}
                continue

            # The check if the race if is in the all the results
            if rider_result['race_id'] not in all_race_results:
                logging.warning(
                    "Didn't find race ID {}: {} - {}".format(rider_result['race_id'], rider_result['race_slug'],
                                                             rider_result['date'].year))
                rider_results.loc[i, 'has_no_results'] = True
                pprint(all_race_results)
                pprint(rider_result['race_id'])
                print(self.race_metadata.saved_result)
                print(self.race_metadata.saved_output)
                continue

            # Check if that race in particular has saved results
            if not all_race_results[rider_result['race_id']]['metadata']['has_results']:
                rider_results.loc[i, 'has_no_results'] = True
                continue

            # Get the results for the particular race / stage
            race_results = all_race_results[rider_result['race_id']].get('results')
            # This one should be avoided with has_results..
            if race_results is None:
                logging.warning("No race results for {}.".format(rider_result['race_id']))
                rider_results.loc[i, 'has_no_results'] = True
                continue

            # Find the rider in the results
            race_result = race_results[(race_results['rider_slug'] == rider_meta_data['rider_slug']) | (race_results['rider_slug'] == get_alias(rider_meta_data['rider_slug']))]

            # The rider might not be in the results (thanks PCS)
            if len(race_result) == 0:
                logging.warning(
                    "Rider {} not found in {}".format(rider_meta_data['rider_slug'], rider_result['race_id']))
                continue

            # Save the times e .. ufa.. done!
            rider_results.loc[i, 'time_rel'] = deepcopy(race_result['time_rel'].iloc[0])
            rider_results.loc[i, 'time_abs'] = deepcopy(race_result['time_abs'].iloc[0])

        self.rider_metadata.update_or_write(rider_meta_data)
        self.rider_result_io.save(rider_slug, pd.DataFrame(rider_results))

    def get_race(self, race_slug: str, race_year: int) -> Optional[dict]:
        race_id = '{}_{}'.format(race_slug, race_year)
        metadata = self.race_metadata.get_row(race_id)
        if metadata is None:
            return self.download_race(race_slug, race_year, race_id)

        today = datetime.now()
        if metadata['last_update'] <= datetime(today.year, today.month, today.day) and metadata['last_update'] < datetime(race_year, 12, 31):
            return self.download_race(race_slug, race_year, race_id)

        results = {metadata['race_id']: {
            'metadata': metadata}
        }

        if metadata['has_results']:
            results[metadata['race_id']]['results'] = self.race_results_io.read(metadata['race_id'])

        if metadata['is_only_start_list']:
            start_list = self.race_results_io.read('start_list_' + metadata['race_id'])
            results[metadata['race_id']]['start_list'] = start_list

        for stage_metadata in self.race_metadata.get_rows_by_column({'year': race_year, 'race_slug': race_slug}):
            if stage_metadata['race_type'] == 'gc':
                continue
            if not stage_metadata['has_results']:
                continue
            stage_results = self.race_results_io.read(stage_metadata['race_id'])
            results[stage_metadata['race_id']] = {
                'metadata': stage_metadata,
                'results': stage_results
            }
        return results


    def get_top_riders(self, year: int):
        return self.download_top_riders(year)


    def download_top_riders(self, year: int):
        range=[0,100,200,300,400]
        rider_ranks = list()
        for offset in range:
            url = "https://www.procyclingstats.com/rankings.php?date={}-12-31&offset={}&s=season-individual".format(year,offset)

            page_soup = get_page_in_soup(url)
            if page_soup is None:
                return None

            page = TopRidersYearParser(page_soup, url, year)

            rider_ranks += page.get_ranks()
            if rider_ranks is None:
                print("Couldn't get results for {}".format(url))
                return None

        self.top_rider_year_io.save(str(year), pd.DataFrame(rider_ranks))

        return {'results': pd.DataFrame(rider_ranks)}

    def get_specific_top_riders(self, year: int, age: int,country: str):
        url = "https://www.procyclingstats.com/rankings.php?date={}-12-31&age={}&nation={}&zage=&page=equal&team=&offset=0&filter=Filter&s=season-individual".format(year,age,country)
        page_soup = get_page_in_soup(url)
        if page_soup is None:
            return None

        page = TopRidersYearParser(page_soup, url, year)

        rider_ranks = page.get_ranks()
        if rider_ranks is None:
            print("Couldn't get results for {}".format(url))
            return None

        return {'results': pd.DataFrame(rider_ranks)}

    """
    def download_top_riders(self, year: int):
        url = "https://www.procyclingstats.com/rankings.php?date={}-12-31&offset={}".format(year,0)
        results = self.download_single_year_top_riders(url, year)

        return results

    def download_single_year_top_riders(self, url: str, year: int):
        page_soup = get_page_in_soup(url)
        #todo: probably for next pages
        if page_soup is None:
            return None

        page = TopRidersYearParser(page_soup, url, year)

        rider_ranks = page.get_ranks()
        if rider_ranks is None:
            print("Couldn't get results for {}".format(url))
            return None

        self.top_rider_year_io.save(str(year), pd.DataFrame(rider_ranks))

        return {'results': pd.DataFrame(rider_ranks)}
    """
    def download_race(self, race_slug: str, race_year: int, race_id: str) -> Optional[dict]:
        # URL of the HTML page
        url = "https://www.procyclingstats.com/race/{}/{}/gc".format(race_slug, race_year)  # XXX

        # Make a GET request to retrieve the HTML content
        response = requests.get(url)

        ## Check if the response status is successful (200)
        if response.status_code == 200:
            # Check if the desired sentence is present in the HTML content
            if 'Page not found' in response.text:
                url = "https://www.procyclingstats.com/race/{}/{}/result".format(race_slug, race_year)

        results = {}

        race_info, stages = self.download_single_race(url, race_id)
        if race_info is None:
            return None

        results[race_info['metadata']['race_id']] = race_info

        for stage_slug in stages:
            url = "https://www.procyclingstats.com/race/{}/{}".format(race_slug, race_year) #XXX
            stage_url = url + '/{}'.format(stage_slug)
            stage_id = "{}_{}".format(race_id, stage_slug)

            stage_info, _ = self.download_single_race(stage_url, stage_id)
            if stage_info is None:
                logging.error("Couldn't get the following stage: {}".format(stage_url))
                continue

            results[stage_info['metadata']['race_id']] = stage_info

        return results

    def download_single_race(self, url: str, race_id: str) -> Optional[tuple]:
        page_soup = get_page_in_soup(url)

        if page_soup is None:
            return None, None

        page = RaceParser(page_soup, url, race_id)

        if not page.is_eligible():
            logging.info("Race is TTT or Hammer - Only getting meta data")
            return None, None

        metadata = page.get_metadata()

        # Check if it is only the start list and save / return it
        if metadata['is_only_start_list']:
            if 'stage' in metadata['race_type']:
                return {'metadata': metadata, 'start_list': None}, page.get_stages()
            start_list = self.get_start_list(metadata)
            if start_list is None:
                print("Couldn't get a future race: {}".format(url))
                return None, None
            self.race_metadata.update_or_write(metadata)
            self.race_results_io.save('start_list_' + metadata['race_id'], pd.DataFrame(start_list))
            return {'metadata': metadata, 'start_list': pd.DataFrame(start_list)}, page.get_stages()

        race_results = page.get_results()
        if race_results is None:
            print("Couldn't get results for {}".format(url))
            return None, None

        self.race_metadata.update_or_write(metadata)
        self.race_results_io.save(metadata['race_id'], pd.DataFrame(race_results))

        return {'metadata': metadata, 'results': pd.DataFrame(race_results)}, page.get_stages()

    def get_start_list(self, metadata: dict) -> Optional[pd.DataFrame]:

        page_soup = get_page_in_soup('{}/startlist'.format(metadata['url']))

        if page_soup is None:
            return None

        page = StartListParser(page_soup)
        return page.get_start_list()
