from os import makedirs
from datetime import datetime

def today():
    date = datetime.today().strftime('%Y-%m-%d')
    return date

def make_dir_if_not_exists(path):
    if path != '':
        try:
            makedirs(path)
        except FileExistsError:
            pass