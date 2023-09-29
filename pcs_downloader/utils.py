import logging
import re
from datetime import datetime
from typing import Optional, Any
import requests
from bs4 import BeautifulSoup, BeautifulStoneSoup

# This is a special case of a rider that has been renamed. 
redirects = {
    'https://www.procyclingstats.com/rider/steve-vermaut': 'https://www.procyclingstats.com/rider/stive-vermaut'
}

# Download a page and create a BS object
def get_page_in_soup(url: str) -> BeautifulSoup:
    if url in redirects:
        url = redirects[url]

    user_agent = {'User-agent': 'Mozilla/5.0'}
    logging.debug(">>> " + url)
    response = requests.get(url, headers=user_agent)
    logging.debug("<<< " + str(response.status_code) + " " + url)

    if response.status_code != 200:
        logging.error('unsuccessful request!')
        return None

    if "The page could not be found." in response.text:
        logging.warning("page doesn't exist!")
        return None

    if "the page you are trying to visit doesn't exists" in response.text:
        logging.warning("page doesn't exist!")
        return None

    if "requested page could not be found" in response.text:
        logging.warning("page doesn't exist!")
        return None

    return BeautifulSoup(response.text, 'lxml')


# Find a URL in a BS object and return it as a string
def get_url(html: BeautifulSoup) -> str:
    url_tag = html.find('a')
    return url_tag.get('href', None)


# Convert a time in format HH:MM:SS to seconds
# TODO: This most likely can be done with a datetime / regex
def str_to_seconds(time_str: str) -> int:
    if time_str == '-':
        return -1
    splits = time_str.split(":")
    if len(splits) == 1:
        return int(splits[0])
    if len(splits) == 2:
        return int(splits[1]) + (int(splits[0]) * 60)
    if len(splits) == 3:
        return int(splits[2]) + (int(splits[1]) * 60) + (int(splits[0]) * 60 * 60)
    return -1


# Convert seconds to HH:MM:SS in string
# TODO: This most likely can be done with a datetime / regex
def seconds_to_str(time_sec: int) -> str:
    hours = time_sec // 3600
    minutes = (time_sec - (hours * 3600)) // 60
    seconds = time_sec - (hours * 3600) - (minutes * 60)

    if minutes < 10:
        minutes = '0' + str(minutes)
    if seconds < 10:
        seconds = '0' + str(seconds)

    return "{}:{}:{}".format(hours, minutes, seconds)


# Takes a string date used at PCS and converts into datetime
def process_string_date(date_str: str) -> Optional[datetime]:
    if date_str is None:
        return None

    data_str = date_str. \
        replace('th ', ' '). \
        replace('nd ', ' '). \
        replace('st ', ' '). \
        replace('rd ', ' '). \
        replace('Augu', 'August')

    data_str_split = data_str.split('(')

    try:
        race_date = datetime.strptime(data_str_split[0].strip(), '%d %B %Y')
    except ValueError as e:
        print(e)
        return None
    return race_date


# Sets to none if a condition is met
def set_to_none(in_any: Optional[Any], condition: Any) -> Optional[Any]:
    if in_any is None:
        return None
    if in_any == condition:
        return None
    return in_any


# Finds a number in the string and returns as int
def to_int(in_str: Optional[str], zero_is_none:bool=False, end_position:int=0) -> Optional[int]:
    output = find_numbers(in_str, end_position)
    if output is None:
        return output
    output = int(output)
    if zero_is_none and output == 0:
        return None
    return output


# Finds a float in the string and returns as float
def to_float(in_str: Optional[str], zero_is_none:bool=False, end_position:int=0) -> Optional[float]:
    output = find_numbers(in_str, end_position)
    if output is None:
        return output
    output = float(output)
    if zero_is_none and output == 0:
        return None
    return output


# Finds number a string
def find_numbers(in_str: str, end_position:int=0) -> Optional[str]:
    if in_str is None:
        return None

    if end_position > 0:
        in_str = in_str[:end_position]

    matches = re.findall(r"[-+]?\d*\.\d+|\d+", in_str)

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError("More than one match found")

    return matches[0]
