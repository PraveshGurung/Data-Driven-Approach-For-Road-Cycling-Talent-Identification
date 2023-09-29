import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup

from pcs_downloader.utils import process_string_date, to_int, to_float


class RiderMetaDataParser:
    def __init__(self, page: BeautifulSoup, url: str):
        self.page = page
        self.metadata = None
        self.rider_info = None
        self.url = url

    def get_metadata(self) -> dict:
        if self.metadata is None:
            self.build_metadata()
        return self.metadata

    def build_metadata(self) -> None:
        # rider_slug,rider_name,birthday,nationality,
        # weight,height,place_of_birth,instagram,strava,facebook,twitter,pcs,website,pcs_photo_url,last_update

        if self.rider_info is None:
            self.parse_info()

        self.metadata = {'rider_slug': self.url.split('/')[-1],
                         'rider_name': self.get_rider_name(),
                         'birthday': self.rider_info.get('birthday'),
                         'nationality': self.rider_info.get('nationality'),
                         'weight': self.rider_info.get('weight'),
                         'height': self.rider_info.get('height'),
                         'place_of_birth': self.rider_info.get('place_of_birth'),
                         'instagram': self.rider_info.get('instagram'),
                         'strava': self.rider_info.get('strava'),
                         'facebook': self.rider_info.get('facebook'),
                         'twitter': self.rider_info.get('twitter'),
                         'pcs': self.url,
                         'website': self.rider_info.get('website'),
                         'pcs_photo_url': self.get_image_url(),
                         'last_update': datetime.now()}

    def parse_info(self):
        sections = {
            'Date of birth:': 'birthday',
            'Nationality:': 'nationality',
            'Weight:': 'weight',
            'Height:': 'height',
            'Place of birth:': 'place_of_birth',
        }

        urls = ['twitter', 'instagram', 'facebook', 'website', 'strava']

        extracted_sections = {}
        self.rider_info = {}

        # Find the meta data section
        try:
            info_section = self.page.select('.rdr-info-cont')[0]
        except Exception as e:
            print("URL: {}".format(self.url))
            raise e

        # Get all the metadata
        available_sections = info_section.findAll('b')

        for available_section in available_sections:
            html_answer = re.findall("(?<=<b>{}<\/b>)(.*?)(?=<b>)".format(available_section.text), str(info_section))
            if len(html_answer)>0:
                bs_answer = BeautifulSoup(html_answer[0],features="lxml")
                extracted_sections[available_section.text] = bs_answer.text
                continue
            html_answer = re.findall("(?<=<b>{}<\/b>)(.*?)(?=<div class)".format(available_section.text), str(info_section))
            bs_answer = BeautifulSoup(html_answer[0],features="lxml")
            extracted_sections[available_section.text] = bs_answer.text

        for section_name, section_id in sections.items():
            self.rider_info[section_id] = extracted_sections.get(section_name)
        self.rider_info['birthday'] = process_string_date(self.rider_info.get('birthday'))
        self.rider_info['weight'] = to_int(self.rider_info.get('weight', '---'), end_position=3)
        self.rider_info['height'] = to_float(self.rider_info.get('height', '---'))

        # Get all the URLs
        links_ul = info_section.select('ul.list.horizontal.sites')
        if links_ul:
            for links_li in links_ul[0].find_all('li'):
                if links_li.text in urls:
                    try:
                        self.rider_info[links_li.text] = links_li.find('a')['href']
                    except TypeError as e:
                        logging.warning("URL not found in social for {}".format((self.url)))

    def get_image_url(self):
        image_section = self.page.select('.rdr-img-cont')
        if len(image_section) < 1:
            return None
        image = image_section[0].find('img')
        if image is None:
            return None
        return image['src']

    def get_rider_name(self) -> str:
        name = self.page.find('h1').text
        return ' '.join(name.split())
