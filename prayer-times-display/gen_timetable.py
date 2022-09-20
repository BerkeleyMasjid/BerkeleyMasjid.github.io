import requests
from datetime import datetime
import pandas as pd

TIMETABLE_LEN_MONTHS = 360

PRAYER_NAMES = ["Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha"]
IQAMA_NAMES = [p+"_iqama" for p in PRAYER_NAMES]

class PrayerTimeApiWrapper:
    def __init__(self):
        self.api_url = 'http://api.aladhan.com/v1/'

    def _get(self, resource, params=None, **kwargs):
        request = requests.get(
            url=f"{self.api_url}{resource}",
            params=params,
            **kwargs
        )
        request.raise_for_status()
        return request.json()

    def get_data(self, params):
        return self._get(
            resource="calendarByCity",
            params=params
        )

api = PrayerTimeApiWrapper()

def time_to_float(timestring):
    # expect time with "HH:MM" format
    hours, minutes = [int(part) for part in timestring.split(":")]
    timefloat = hours + minutes/60
    return timefloat

def format_timings(data):
    timings = dict((k.lower(), time_to_float(data["timings"][k][:5])) for k in PRAYER_NAMES)
    date = data["date"]["gregorian"]["date"]
    date_YYYYMMDD = date[-4:] + date[3:5] + date[:2]
    return date_YYYYMMDD, timings


empty_iqama_timings = dict([(k.lower(), "") for k in IQAMA_NAMES])
all_timings = {}

i = 0
j = 0
while TIMETABLE_LEN_MONTHS > 0:
    if datetime.now().month + i > 12:
        i = 1 - datetime.now().month
        j += 1

    params = dict(
        city="Berkeley", 
        state="California", 
        country="US",
        month=datetime.now().month + i,
        year=datetime.now().year - 1 + j, #start a year before today to equalize iqama trend calculation in calc_iqama_times.py
        method=2, #ISNA
        school=0, #shafii
        latitudeAdjustmentMethod=3, #angle based
        )

    response = api.get_data(params)["data"]

    for day_data in response:
        date, timings = format_timings(day_data)
        timings.update(empty_iqama_timings)
        all_timings[date] = timings

    i += 1
    TIMETABLE_LEN_MONTHS -= 1

df = pd.DataFrame.from_dict(all_timings, orient="index")
df.to_parquet("timetable.parquet")
