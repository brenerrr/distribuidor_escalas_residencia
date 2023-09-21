from typing import Literal
from types import MappingProxyType

MORNING = "M"
AFTERNOON = "T"
EVENING = "N"
START_TIMES_STR = MappingProxyType(
    {MORNING: "07:00", AFTERNOON: "14:00", EVENING: "19:00"}
)
DURATION_TIMES = MappingProxyType({MORNING: 5, AFTERNOON: 5, EVENING: 12})

START_TIMES = MappingProxyType({MORNING: 7, AFTERNOON: 14, EVENING: 19})


LARGE_NUMBER = 1e5

PERIODS = (MORNING, AFTERNOON, EVENING)
DAYS_OF_WEEK = ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "DOM")
DAYS_PORT2ENG = MappingProxyType(
    dict(zip(DAYS_OF_WEEK, ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]))
)
WEEKEND_START_DAY = 4
WEEKEND_START_TIME = START_TIMES[EVENING]

PERIODS_TYPE = Literal["M", "A", "E"]
