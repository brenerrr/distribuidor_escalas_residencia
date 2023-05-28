from typing import Literal
from types import MappingProxyType

MORNING = "M"
AFTERNOON = "A"
EVENING = "E"
START_TIMES = MappingProxyType({MORNING: "07:00", AFTERNOON: "14:00", EVENING: "19:00"})
LARGE_NUMBER = 1e3

PERIODS = (MORNING, AFTERNOON, EVENING)
DAYS_OF_WEEK = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")

PERIODS_TYPE = Literal["M", "A", "E"]
