import logging
from typing import Optional

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .utils import seconds_to_str, str_to_seconds, to_float, to_int


class RaceParser:
    def __init__(self, page: BeautifulSoup, url: str, race_id: str):
        self.page = page
        self.url = url
        self.race_id = race_id
        split = self.race_id.split('_')
        self.race_slug = split[0]
        self.race_year = split[1]
        try:
            self.stage_slug = split[2]
        except IndexError:
            self.stage_slug = None
        self.parsed = False
        self.metadata = None
        self.race_info = None
        self.race_results = None
        self.stage_list = None
        self.title = None

    def __repr__(self):
        return str(self.page)

    def parse(self) -> None:
        if self.race_info is None:
            self.parse_race_info()

        self.metadata = {'race_id': self.race_id,
                         'race_slug': self.race_slug,
                         'url': self.url,
                         'race_type': self.get_race_type(),
                         'race_name': self.get_race_name(),
                         'date_start': self.__fillin_start_date(),
                         'date_finish': self.race_info.get('race_date'),
                         'year': self.race_info.get('race_date').year,
                         'distance': self.race_info.get('distance'),
                         'category': self.race_info.get('category'),
                         'class': self.get_uci_point_scale(),
                         'parcours_type': self.race_info.get('parcours_type'),
                         'profile_score': self.race_info.get('profile_score'),
                         'departure': self.__exclude_type('gc', self.race_info.get('departure')),
                         'arrival': self.__exclude_type('gc', self.race_info.get('arrival')),
                         'avg_speed_winner': self.race_info.get('avg_speed_winner'),
                         'race_ranking': self.race_info.get('race_ranking'),
                         'has_uci_points': self.field_exists('uci_points'),
                         'has_time': self.field_exists('time_abs'),
                         'is_only_start_list': self.is_in_future(),
                         'last_update': datetime.now(),
                         'nr_stages': len(self.get_stages()),
                         'stage_name': self.get_stage_name(),
                         'stage_slug': self.stage_slug,
                         'has_results': len(self.get_results()) > 0}
        #print(self.metadata)

        self.parsed = True

    def parse_race_info(self) -> None:
        sections = {
            'Date:': 'race_date',
            'Avg. speed winner:': 'avg_speed_winner',
            'Race category:': 'category',
            'ProfileScore:': 'profile_score',
            'Departure:': 'departure',
            'Arrival:': 'arrival',
            'Distance:': 'distance',
            'Race ranking:': 'race_ranking',
            'Parcours type:': 'parcours_type',
            'PCS point scale:': 'pcs_point_scale'
        }

        self.race_info = {}

        try:
            info_section = self.page.select('.infolist')[0]
        except Exception as e:
            print("Couldn't find the info list section in {}".format(self.url))
            raise e
        for li_item in info_section.findAll('li'):
            divs = li_item.findAll('div')
            new_section_name = divs[0].text.strip()
            if new_section_name == 'Parcours type:':
                self.race_info[sections[new_section_name]] = li_item.find('span')['class'][2][1]
            elif new_section_name in sections:
                self.race_info[sections[new_section_name]] = divs[1].text.strip()

        if 'race_date' in self.race_info:
            try:
                self.race_info['race_date'] = datetime.strptime(self.race_info['race_date'].split(',')[0], '%d %B %Y')
            except ValueError as e:
                date_split = self.race_info['race_date'].split(' ')
                date_split[0] = '1'
                if len(date_split) == 1:
                    self.race_info['race_date'] = datetime(year=int(self.race_year), day=1, month=1)
                else:
                    try:
                        self.race_info['race_date'] = datetime.strptime(' '.join(date_split), '%d %B %Y')
                    except:
                        raise ValueError("Couldn't parse date in {}".format(self.url))

        self.race_info['avg_speed_winner'] = to_float(self.race_info.get('avg_speed_winner'), zero_is_none=True)
        self.race_info['parcours_type'] = to_int(self.race_info.get('parcours_type'), zero_is_none=True)
        self.race_info['profile_score'] = to_int(self.race_info.get('profile_score'), zero_is_none=True)
        self.race_info['distance'] = to_float(self.race_info.get('distance'), zero_is_none=True)
        self.race_info['race_ranking'] = to_float(self.race_info.get('race_ranking'), zero_is_none=True)

    def get_metadata(self) -> dict:
        if not self.parsed:
            self.parse()
        return self.metadata

    def is_eligible(self) -> bool:
        if 'results' == self.get_title():
            logging.warning('FOUND EMPTY PAGE')
            return False
        if 'test-event' in self.race_slug:
            return False
        if 'hammer' in self.get_title().lower():
            return False
        if 'TTT' in self.get_title():
            return False
        # if 'Prologue' in self.get_title():
        #     return False
        return True

    def get_title(self) -> str:
        if self.title:
            return self.title
        title = self.page.find('title').text.replace('/n', '').replace('"', '')
        self.title = title.replace("| ", "").replace("|", "").replace("Results", "").strip()
        return self.title

    def get_race_type(self) -> str:
        if self.stage_slug is not None:
            if 'itt' in self.get_title():
                return 'gc_stage_itt'
            if 'ttt' in self.get_title():
                return 'gc_stage_ttt'
            if len(self.race_id.split('_')) == 3:
                return 'gc_stage'
        if len(self.page.find_all('ul', {'class': 'restabs'})) == 0:
            return 'one_day'
        if len(self.race_id.split('_')) == 2:
            return 'gc'

        raise ValueError("Unknown type for {}".format(self.url))

    def get_uci_point_scale(self) -> Optional[str]:
        race_title = self.page.select('.main')[0]
        if race_title:
            uci_point_scale = re.findall('\(([^\)]+)\)', race_title.text)
            if uci_point_scale:
                return uci_point_scale[0]
        return None

    def get_race_name(self) -> Optional[str]:
        race_title = self.page.select('.main')[0]
        if race_title:
            return race_title.find('h1').text.strip()
        return None

    def get_stage_name(self) -> Optional[str]:
        race_type = self.get_race_type()
        if 'stage' not in race_type:
            return None

        current_stage = self.page.find_all('div', {'class': 'pageSelectNav'})[1].find('option', selected=True).text
        return current_stage.replace('/n', '').replace('"', '').strip()

    def get_stages(self) -> list:
        if self.stage_list is not None:
            return self.stage_list

        stage_list = []
        top_option_nav = self.page.find_all('div', {'class': 'pageSelectNav'})
        if len(top_option_nav) < 2:
            return stage_list

        stage_selector = self.page.find_all('div', {'class': 'pageSelectNav'})[1].find_all('option')
        for item in stage_selector:
            stage_name = item.text
            if '|' not in stage_name:
                continue
            stage_id = item.get('value')
            stage_id = stage_id.split('/')[-1]
            stage_list.append(stage_id)
        self.stage_list = stage_list
        return self.stage_list

    def is_in_future(self):
        return self.race_info['race_date'] >= datetime.today()

    def field_exists(self, header_key):
        if self.race_results is None:
            self.build_results()
        if not self.race_results:
            return False
        return self.race_results[0][header_key] is not None

    def get_results(self) -> Optional[list]:
        if self.race_results is None:
            self.build_results()
        return self.race_results

    def build_results(self) -> None:
        self.race_results = []
        table = None
        print("Getting results from {}".format(self.url))
        for div in self.page.select("div.result-cont"):
            # There is tab selector on the page, we want the selected one
            if "hide" in div["class"]:
                continue
            table = div
            break
        if table is None:
            print("Error getting: ")
            print(self.url)
            return None

        if len(table.find_all('table')) == 0:
            return

        idx = self.__build_result_index(table)
        if not idx:
            return

        if 'Rider' not in idx:
            return

        first_finish_time = None
        table_content = table.find('tbody')
        for row in table_content.find_all('tr'):
            try:
                result, first_finish_time = self.__build_row(row, first_finish_time, idx)
            except IndexError:
                logging.error("Error indexing on {}, row:\n {}".format(self.url, row))
                continue
            if result is None:
                logging.error("Error getting results on {}, row:\n {}".format(self.url, row))
                continue
            self.race_results.append(result)

    def __build_result_index(self, table:BeautifulSoup) -> dict:
        results_header = {}
        try:
            table_header = table.find('thead').find_all('th')
        except:
            return results_header
        for i, theader in enumerate(table_header):
            results_header[theader.string] = i
        return results_header

    def __build_row(self, row:BeautifulSoup, first_finish_time:Optional[int], index:dict) -> (dict, int):
        result_raw = {}
        items = row.find_all('td')

        for name, inx in index.items():
            result_raw[name] = items[inx]

        uci_points = None
        pcs_points = None
        if 'UCI' in result_raw:
            uci_points = result_raw.get('UCI').text.strip()

        if 'Pnt' in result_raw:
            pcs_points = result_raw.get('Pnt').text.strip()

        result = {'rider_name': result_raw['Rider'].find('a').text.strip(),
                  'rider_slug': result_raw['Rider'].find('a')['href'].split('/')[1],
                  'rider_url': result_raw['Rider'].find('a')['href'],
                  'team': result_raw['Team'].text.strip(),
                  'uci_points': uci_points,
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

        rank = result_raw['Rnk'].string.strip()
        if result_raw['Rnk'].find('s'):
            rank = "CANCELLED"

        # Racer finished the race
        int_rank = True
        try:
            rank = int(rank)
            if rank > 700:
                int_rank = False
        except:
            int_rank = False

        """
        if int_rank:
            # The raw relative finish time is relative EXCEPT for the first rider
            if first_finish_time is None:
                try:
                    first_finish_time = str_to_seconds(result_raw['Time'].find('div').text)
                except AttributeError:
                    first_finish_time = str_to_seconds(result_raw['Time'].text)
                rel_time = 0
                abs_time = result_raw['Time'].text

            else:
                try:
                    raw_relative_finish_time = result_raw['Time'].find('div').text
                except AttributeError:
                    logging.warning("Error parsing time from {}, used backup method".format(self.url))
                    raw_relative_finish_time = result_raw['Time'].text
                try:
                    rel_time = str_to_seconds(raw_relative_finish_time)
                    abs_time = self.__get_abs_time(rel_time, first_finish_time)
                except ValueError as e:
                    err_str = "Found Value Error in {} with value {}".format(self.url, raw_relative_finish_time)
                    logging.critical(err_str)
                    logging.critical(e)
                    print("!!! " + err_str)
                    return None, first_finish_time

            result['rank'] = rank
            result['time_rel'] = rel_time
            result['time_abs'] = abs_time

        else:
            if isinstance(rank, int):
                rank = "OTHER_" + str(rank)
            elif (rank.isdigit()):
                rank = "OTHER_" + rank
            result['rank'] = rank
            result['time_rel'] = None
            result['time_abs'] = None
        """
        #todo my temp changes
        result['rank'] = rank
        result['time_rel'] = None
        result['time_abs'] = None
        return result, first_finish_time

    def __get_abs_time(self, rel_time, first_finish_time):
        if first_finish_time == 'error':
            return 'error'
        abs_time = first_finish_time + rel_time

        return seconds_to_str(abs_time)

    def __exclude_type(self, race_type:str, in_str:Optional[str]) -> Optional[str]:
        if race_type == self.get_race_type():
            return None
        return in_str

    def __fillin_start_date(self) -> datetime:
        if self.get_race_type() != 'gc':
            return self.race_info['race_date']
