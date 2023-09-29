from datetime import datetime
from typing import Optional
import urllib.parse
import pandas as pd

from bs4 import BeautifulSoup

from pcs_downloader.utils import to_float, to_int


class RiderResultPageParser:
    def __init__(self, page: BeautifulSoup, url:str, rider_slug:str, include_this_year=False):
        self.url = url
        self.rider_slug = rider_slug
        self.page = page
        self.results = None
        self.current_offset = None
        self.limit = None
        self.include_this_year = include_this_year

    def get_results(self) -> list:
        if self.results is None:
            self.build_results()
        return self.results

    def build_results(self):
        self.results = pd.DataFrame()
        results_list = []
        idx = self.__build_index()
        table = self.page.find('tbody')

        for row in table.find_all('tr'):
            try:
                item = self.__build_row(row, idx)
            except IndexError as e:
                print("Error indexing at {}".format(self.url))
                print("Index: {}".format(idx))
                print("Row: {}".format(row))
                raise e
            if item is None:
                continue
            results_list.append(item)
        self.results = pd.DataFrame(results_list)

    def get_next_url(self) -> Optional[str]:
        if self.results is None:
            self.build_results()
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.url).query))
        if len(self.results) == 0:
            return None
        if int(params['limit']) > len(self.results):
            return None

        new_offset = int(params['offset']) + int(params['limit'])

        next_url = 'https://www.procyclingstats.com/rider.php?xseason=&sort=date&race=&km1=&zkm1=&pkm1=equal&limit' \
                    '=200&offset={offset}&topx=&ztopx=&ptopx=smallerorequal&type=&continent=&pnts=&zpnts=&ppnts=equal' \
                    '&level=&rnk=&zrnk=&prnk=equal&exclude_tt=0&racedate=&zracedate=&pracedate=equal&filter=Filter' \
                    '&id={rider_slug}&p=results'.format(rider_slug = self.rider_slug, offset = new_offset)

        return next_url

    def get_history(self) -> Optional[list]:
        if self.history is not None:
            return self.history

    def __build_index(self):
        try:
            table_header = self.page.find('thead').find_all('th')
        except Exception as e:
            print("URL: {}".format(self.url))
            raise(e)
        headers = {}
        for i, theader in enumerate(table_header):
            headers[theader.text.strip()] = i
        return headers

    def __build_row(self, row, index):
        result_raw = {}
        items = row.find_all('td')

        for name, inx in index.items():
            result_raw[name] = items[inx]
        if not result_raw['#'].text.strip():
            return None
        url_split = result_raw['Race'].find('a')['href'].split('/')

        race_type = 'unknown'
        race_name = result_raw['Race'].text.strip()
        race_no_results = False
        stage_slug = None
        if 'hammer' in race_name.lower():
            race_type = 'hammer'
            race_no_results = True
        elif url_split[3] == 'result':
            if 'ttt' in race_name.lower():
                race_type = 'ttt'
                race_no_results = True
            else:
                race_type = 'one_day'
        elif url_split[3] == 'gc':
            race_type = 'gc'
        elif url_split[3] == 'points':
            race_type = 'gc_points'
            race_no_results = True
            stage_slug = 'gc_points'
        elif url_split[3] == 'kom':
            race_type = 'gc_kom'
            race_no_results = True
            stage_slug = 'gc_kom'
        elif url_split[3] == 'youth':
            race_type = 'gc_youth'
            race_no_results = True
            stage_slug = 'gc_youth'
        elif 'stage' in url_split[3]:
            stage_slug = url_split[3]
            if 'itt' in race_name.lower():
                race_type = 'gc_stage_itt'
            elif 'ttt' in race_name.lower():
                race_type = 'gc_stage_ttt'
                race_no_results = True
            else:
                race_type = 'gc_stage'
        elif 'prologue' in url_split[3]:
            stage_slug = url_split[3]
            race_type = 'gc_stage'
        else:
            race_no_results = True
            stage_slug = url_split[3]

        try:
            race_date = datetime.strptime(result_raw['Date'].text.strip(), '%Y-%m-%d')
        except ValueError:
            split_date = result_raw['Date'].text.strip().split('-')
            split_date[2] = '01'
            try:
                race_date = datetime.strptime('-'.join(split_date), '%Y-%m-%d')
            except ValueError:
                split_date = result_raw['Date'].text.strip().split('-')
                split_date[1] = '01'
                split_date[2] = '01'
                race_date = datetime.strptime('-'.join(split_date), '%Y-%m-%d')

        race_id = "{}_{}".format(url_split[1], race_date.year)
        if stage_slug is not None:
            race_id += '_' + stage_slug

        result = {
                  'race_id': race_id,
                  'race_slug': url_split[1],
                  'stage_slug': stage_slug,
                  'date': race_date,
                  'year': race_date.year,
                  'race_type': race_type,
                  'class': result_raw['Class'].text.strip(),
                  'race_name': race_name,
                  'race_url': result_raw['Race'].find('a')['href'],
                  'rank':result_raw['Result'].text,
                  'distance': to_float(result_raw['KMs'].text),
                  'pcs_points': to_float(result_raw['PCS points'].text),
                  'uci_points': to_float(result_raw['UCI points'].text),
                  'time_abs': None,
                  'time_rel': None,
                  'has_no_results': race_no_results}
        return result
