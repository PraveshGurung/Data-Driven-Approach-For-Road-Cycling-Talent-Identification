import logging
from typing import Optional

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .utils import seconds_to_str, str_to_seconds, to_float, to_int


# Parses the start list

class StartListParser:
    def __init__(self, page: BeautifulSoup):
        self.page = page
        self.start_list = None

    def __repr__(self):
        return str(self.page)

    def get_start_list(self) -> Optional[list]:
        if self.start_list is None:
            self.build_start_list()
        return self.start_list

    def build_start_list(self) -> None:
        self.start_list = []
        start_list_container = self.page.find('ul', {'class':'startlist_v3'})
        for team_container in start_list_container.find_all('li', {'class':'team'}):
            team_slug = team_container.find('a')['href'].split('/')[1]
            team_url = team_container.find('a')['href']
            team_name = team_container.find('a').text
            team_members_container = team_container.find('ul')
            for team_member in team_members_container.find_all('li'):
                rider_slug = team_member.find('a')['href'].split('/')[1]
                rider_url = team_member.find('a')['href']
                rider_name = team_member.find('a').text
                rider = {
                    'team_slug': team_slug,
                    'team_url': team_url,
                    'team_name': team_name,
                    'rider_slug': rider_slug,
                    'rider_url': rider_url,
                    'rider_name': rider_name
                }
                self.start_list.append(rider)
