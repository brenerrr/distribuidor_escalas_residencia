# %%

from common_types import *


class Employee:
    def __init__(self, name: str, areas: list) -> None:
        self.__id = -1
        self.__name = name
        self.__restrictions = []
        self.__activities = []
        self.__soft_restrictions = []
        self.__areas = areas

    @property
    def name(self) -> str:
        return self.__name

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, x: int) -> None:
        self.__id = x

    @property
    def restrictions(self) -> dict:
        return self.__restrictions

    @property
    def activities(self) -> list:
        return self.__activities

    @property
    def areas(self) -> list:
        return self.__areas

    def __repr__(self) -> str:
        string = (
            f"Employee: {self.__id}, "
            + f"Name {self.__name}, "
            + f"Areas: {self.__areas}, "
            + f"Restrictions {self.__restrictions}, "
            + f"Soft Restrictions:{self.__soft_restrictions}"
        )
        return string

    def add_restriction(
        self, day: int, period: PERIODS_TYPE, soft: bool = False
    ) -> None:
        assert period in [MORNING, AFTERNOON, EVENING]
        if soft:
            self.__soft_restrictions.append(dict(day=day, period=period))
        else:
            self.__restrictions.append(dict(day=day, period=period))
