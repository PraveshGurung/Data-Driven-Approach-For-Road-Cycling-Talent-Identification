import logging
from typing import Optional

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .utils import seconds_to_str, str_to_seconds, to_float, to_int


class TopRidersYearParser:
    def __init__(self, page: BeautifulSoup, url: str, year: int):
        self.page = page
        self.url = url
        self.year = year
        self.rank_results = None

    def __repr__(self):
        return str(self.page)

    def get_ranks(self) -> Optional[list]:
        if self.rank_results is None:
            self.build_results()
        return self.rank_results

    def build_results(self) -> None:
        self.rank_results = []
        table = None
        print("Getting results from {}".format(self.url))
        table = self.page.find('table', {'class':'basic'})

        if table is None:
            print("Error getting: ")
            print(self.url)
            return None


        idx = self.__build_result_index(table)
        if not idx:
            return

        if 'Rider' not in idx:
            return

        table_content = table.find('tbody')
        for row in table_content.find_all('tr'):
            try:
                result = self.__build_row(row, idx)
            except IndexError:
                logging.error("Error indexing on {}, row:\n {}".format(self.url, row))
                continue
            if result is None:
                logging.error("Error getting results on {}, row:\n {}".format(self.url, row))
                continue
            self.rank_results.append(result)

    def __build_result_index(self, table:BeautifulSoup) -> dict:
        results_header = {}
        try:
            table_header = table.find('thead').find_all('th')
        except:
            return results_header
        for i, theader in enumerate(table_header):
            results_header[theader.string] = i
        return results_header

    def __build_row(self, row:BeautifulSoup, index:dict) -> (dict, int):
        result_raw = {}
        items = row.find_all('td')

        for name, inx in index.items():
            result_raw[name] = items[inx]

        pcs_points = None

        if 'Points' in result_raw:
            pcs_points = result_raw.get('Points').text.strip()

        result = {'rider_name': result_raw['Rider'].find('a').text.strip(),
                  'rider_slug': result_raw['Rider'].find('a')['href'].split('/')[1],
                  'rider_url': result_raw['Rider'].find('a')['href'],
                  'team': result_raw['Team'].text.strip(),
                  'pcs_points': pcs_points}

        if result['rider_slug'] == 'steve-vermaut':
            result['rider_slug'] = 'stive-vermaut'

        if result['rider_url'] == 'rider/steve-vermaut':
            result['rider_url'] = 'rider/stive-vermaut'


        try:
            result['team_slug'] = result_raw['Team'].find('a')['href'].split('/')[1]
            result['team_url'] = result_raw['Team'].find('a')['href']
        except:
            result['team_slug'] = None
            result['team_url'] = None

        rank = result_raw['#'].string.strip()
        if result_raw['#'].find('s'):
            rank = "CANCELLED"

        # Racer finished the race
        result['rank'] = rank

        return result

    def __get_abs_time(self, rel_time, first_finish_time):
        if first_finish_time == 'error':
            return 'error'
        abs_time = first_finish_time + rel_time

        return seconds_to_str(abs_time)

