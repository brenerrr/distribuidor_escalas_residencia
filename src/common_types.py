from typing import Literal
from types import MappingProxyType

MORNING = "M"
AFTERNOON = "A"
EVENING = "E"
START_TIMES_STR = MappingProxyType(
    {MORNING: "07:00", AFTERNOON: "14:00", EVENING: "19:00"}
)
DURATION_TIMES = MappingProxyType({MORNING: 5, AFTERNOON: 5, EVENING: 12})

START_TIMES = MappingProxyType({MORNING: 7, AFTERNOON: 14, EVENING: 19})

LARGE_NUMBER = 1e5

PERIODS = (MORNING, AFTERNOON, EVENING)

WEEKEND_START_DAY = 4
WEEKEND_START_TIME = START_TIMES[EVENING]

PERIODS_TYPE = Literal["M", "A", "E"]

MONTHS_STR2NUM = MappingProxyType(
    {
        "Jan": 1,
        "Fev": 2,
        "Mar": 3,
        "Abr": 4,
        "Mai": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Set": 9,
        "Out": 10,
        "Nov": 11,
        "Dez": 12,
    }
)

# Translations
PERIOD = "Período"
NAME = "Nome"
DATE = "Data"
AREA = "Área"
PERIOD_TRANSLATE = MappingProxyType(
    {"Manhã": MORNING, "Tarde": AFTERNOON, "Noite": EVENING}
)
DAYS = "Dias"
DAYS_OF_WEEK = ("Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom")
DAYS_PORT2ENG = MappingProxyType(
    dict(zip(DAYS_OF_WEEK, ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]))
)
N_EMPLOYEES = "Qtd. Pessoas"
DURATION = "Duração"
BALANCE = "Balancear"
YES = "Sim"
MANDATORY = "Obrigatório"
EXCEPTIONS = "Exceções"
MAX_SHIFTS = "Qtd. Máxima de Turnos"
OUTPUT_SHIFTS_NAME = "escala"
OUTPUT_STATS_NAME = "estatisticas"
