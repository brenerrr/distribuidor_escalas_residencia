# %%

from collections import defaultdict
from typing import Union

from matplotlib import pyplot as plt
from common_types import *
import pprint
import datetime
import calendar
from employee import Employee
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import numpy as np
from random import choices, seed
import time
from matplotlib.colors import rgb2hex

np.random.seed(1)
seed(1)


class Manager:
    def __init__(self, year: int, month: int, GA_params: dict) -> None:
        self.__start_date = f"{year}-{month}"
        # self.__employees = list()
        self.__employees = pd.DataFrame([])
        self.__shifts_params = list()
        self.__GA_params = GA_params
        self.__shifts = pd.DataFrame(
            [], columns=["name", "area", "date", "duration", "period"]
        )

    @property
    def shifts_params(self):
        return self.__shifts_params

    @property
    def employees(self):
        return self.__employees

    @property
    def n_employees(self):
        return self.__employees.shape[0]

    @property
    def shifts(self):
        return self.__shifts

    def add_shift_params(
        self,
        name: str,
        area: str,
        duration: int,
        period: PERIODS_TYPE,
        n_employees: int,
        days_of_week: list,
        should_be_balanced: bool = False,
        must_be_filled: bool = True,
        exclude_dates: list = None,
    ) -> None:
        assert all(day in DAYS_OF_WEEK for day in days_of_week)
        assert period in PERIODS

        # Get dates of shifts in this area
        dates = pd.DatetimeIndex([])
        start_date = pd.to_datetime(f"{self.__start_date}-1 {START_TIMES_STR[period]}")
        end_date = start_date + pd.offsets.MonthEnd()
        for day in days_of_week:
            dates_ = pd.date_range(
                start_date,
                end_date,
                freq=f"W-{day}",
            )
            dates = dates.union(dates_)
        if exclude_dates is not None:
            for date in exclude_dates:
                date = pd.to_datetime(f"{date} {START_TIMES_STR[period]}")
                dates = dates[~(dates == date)]

        self.__shifts_params.append(
            dict(
                name=name,
                area=area,
                duration=duration,
                period=period,
                dates=dates,
                should_be_balanced=should_be_balanced == 1,
                must_be_filled=must_be_filled == 1,
                n_employees=n_employees,
            )
        )

    # def add_employee(self, employee: Employee) -> None:
    def add_employee(self, name: str, working_areas: list) -> None:
        areas = [params["area"] for params in self.__shifts_params]
        # assert all(area in areas for area in employee.areas)
        message = (
            "Employee area not registered on manager. \n\n"
            + f"Available areas: {[area['name'] for area in self.shifts_params]} \n\n"
            + f"Employee areas: {working_areas}"
        )
        assert all(area in areas for area in working_areas), message
        # employee.id = len(self.__employees)
        employee = dict(name=[name], hours_worked=[0])
        employee.update({area: [area in working_areas] for area in areas})

        self.__employees = pd.concat(
            [self.__employees, pd.DataFrame(employee)]
        ).reset_index(drop=True)

    def create_shifts(self) -> pd.DataFrame:
        start = pd.to_datetime(f"{self.__start_date}")
        end = start + pd.offsets.MonthEnd()
        dates = pd.date_range(start, end)
        weeks_i = (dates.shift(-1).weekday // 6).values.cumsum()
        # weeks = dates.groupby(weeks_i)
        shifts = []

        for date, week in zip(dates, weeks_i):
            for params in self.__shifts_params:
                i = params["dates"].date == date.date()

                if any(i):
                    is_weekend = (date.weekday() > WEEKEND_START_DAY) | (
                        (date.weekday() == WEEKEND_START_DAY)
                        & (params["dates"][0].hour >= WEEKEND_START_TIME)
                    )

                    for n in range(params["n_employees"]):
                        shifts.append(
                            dict(
                                name=f"{date.date()}_{params['name']}_{n}",
                                area=params["area"],
                                duration=params["duration"],
                                period=params["period"],
                                week=week,
                                is_weekend=is_weekend,
                                should_be_balanced=params["should_be_balanced"],
                                must_be_filled=params["must_be_filled"],
                                date=params["dates"][i][0],
                            )
                        )

        #  Sort shifts chronollogically
        shifts.sort(key=lambda x: (x["date"].day, x["date"].hour, -x["duration"]))

        shifts = pd.DataFrame(shifts)
        shifts["employee"] = ""
        shifts["employee_id"] = ""
        self.__shifts = shifts

    def create_schedule(self) -> None:
        # Create dates

        # Create shifts
        # self.create_shifts(all_dates)

        best = self.perform_GA()

        # Assign people from solution to shifts
        solution = best["solution"][best["solution"] >= 0]
        self.__shifts = self.shifts[best["solution"] >= 0]
        self.__shifts["employee"] = self.employees.iloc[solution]["name"].values
        self.__shifts["employee_id"] = solution

        # Add statistics to employees
        hours_worked = self.shifts.groupby(solution)["duration"].sum()
        hours_worked = hours_worked.rename("hours_worked")
        n_shifts = (
            self.shifts.groupby(["employee_id", "area"])
            .size()
            .reset_index(level=1)
            .pivot(values=0, columns="area")
        )
        n_shifts = n_shifts.fillna(0)
        n_shifts = n_shifts.astype(int)

        employees = pd.merge(
            self.employees[["name"]], n_shifts, left_index=True, right_index=True
        )
        employees = pd.merge(employees, hours_worked, left_index=True, right_index=True)
        self.__employees = employees
        pass

    def perform_GA(self):
        self.create_shifts()

        # Initialize solutions and scores
        solutions = self.create_solutions(self.__GA_params["n_solutions"])
        scores = self.calculate_scores(solutions)

        best = {}
        i = scores.argmax()
        best["score"] = scores[i]
        best["solution"] = solutions.loc[:, i]

        print(scores)
        # print(best)
        # plt.plot(max_scores)
        # plt.plot(mean_scores)
        # plt.show(block=True)
        return best

    def pick_best_solutions(self, scores: np.array):
        # message = "Not enough valid scores. Try increasing population."
        # assert (scores > 0).sum() >= 2, message
        # prob = scores / scores.sum()
        # best_solutions_i = np.random.choice(prob.size, 2, p=prob, replace=False)
        best_solutions_i = scores.argsort()[-2:]
        return best_solutions_i

    def mutate(self, solutions: np.array, shifts: pd.DataFrame) -> np.array:
        mutation_rate = self.__GA_params["mutation_rate"]
        mask = np.random.choice(
            [True, False], solutions.shape, p=[mutation_rate, 1 - mutation_rate]
        )

        random_values = np.random.randint(0, self.n_employees, solutions.shape)
        new_solutions = solutions * (1 - mask) + random_values * mask
        return new_solutions

    def crossover(self, solutions: pd.DataFrame) -> pd.DataFrame:
        solution1, solution2 = solutions.T.values
        ci = np.random.randint(0, solution1.size)
        offspring1 = np.concatenate([solution1[:ci], solution2[ci:]])
        offspring2 = np.concatenate([solution2[:ci], solution1[ci:]])
        # return np.stack((offspring1, offspring2)).T
        return pd.DataFrame(offspring1.reshape(-1, 1))

    def create_solutions(
        self, n_solutions: int, guide: pd.DataFrame = None
    ) -> pd.DataFrame:
        shifts = self.shifts
        solutions = np.full((shifts.shape[0], n_solutions), np.nan)
        solutions = pd.DataFrame(solutions)

        c = 0
        for j in range(n_solutions):
            solutions.loc[:, j] = np.nan

            # Loop through shifts
            i = 0
            while i <= shifts.index[-1]:
                shift = shifts.iloc[i]

                next_shifts = shifts.iloc[i + 1 :]

                # Do not consider shifts without solutions
                ii = solutions.loc[: i - 1, j] >= 0
                previous_shifts = shifts.loc[: i - 1]
                solution = solutions.loc[: i - 1, j]
                previous_shifts = previous_shifts[ii].reset_index(drop=True)
                solution = solution[ii].reset_index(drop=True)

                cost_matrix = self.calculate_costs(
                    shift,
                    solution.values,
                    previous_shifts,
                    next_shifts,
                )

                # If there are no eligible employees, start again
                if (cost_matrix >= LARGE_NUMBER).all():
                    if shift["must_be_filled"]:
                        print(f"Trying generating another initial solution {c} {i} {j}")
                        c += 1
                        i = previous_shifts[
                            previous_shifts["date"].dt.day == (shift["date"].day - 10)
                        ]
                        i = 0 if i.empty else i.index[0]
                        continue
                    else:
                        solutions.loc[i, j] = -1
                else:
                    # The eligible employees with the most amount of hours worked
                    # will not be considered
                    mask = cost_matrix < LARGE_NUMBER
                    if (cost_matrix[mask].min() == cost_matrix[mask]).sum() > 1:
                        solutions.loc[i, j] = choices(
                            cost_matrix[cost_matrix == cost_matrix.min()].index
                        )[0]
                    else:
                        solutions.loc[i, j] = np.argmin(cost_matrix)

                i += 1
        return solutions

    def calculate_costs(
        self,
        shift: pd.DataFrame,
        solution: np.array = None,
        previous_shifts: pd.DataFrame = None,
        next_shifts: pd.DataFrame = None,
    ) -> pd.Series:
        cost = np.zeros(self.n_employees)

        # Only work on right areas
        cost += (~self.employees[shift["area"]]) * LARGE_NUMBER

        if previous_shifts.empty or solution is None:
            return cost

        # Add hours worked
        hours_worked = previous_shifts.groupby(solution)["duration"].sum()
        cost[hours_worked.index] += hours_worked

        # Only one shift per employee
        same_time_shifts = previous_shifts[previous_shifts["date"] == shift["date"]]
        cost[solution[same_time_shifts.index]] = LARGE_NUMBER

        # If worked yesterday evening, can't work this afternoon or evening
        # Also not able to work next morning if work goes until afternoon
        if shift["period"] in (EVENING, AFTERNOON) or (
            (shift["duration"] > 5) and (shift["period"] == MORNING)
        ):
            yesterday_evening = shift["date"] - pd.offsets.Day(1)
            yesterday_evening = yesterday_evening.replace(hour=START_TIMES[EVENING])
            yesterday_evening_shifts = previous_shifts[
                (previous_shifts["date"] == yesterday_evening)
            ]
            cost[solution[yesterday_evening_shifts.index]] = LARGE_NUMBER

        # On weekends, can not work evening shift if worked long morning shift
        if shift["period"] == EVENING and shift["is_weekend"]:
            this_weekend = previous_shifts[
                (previous_shifts["is_weekend"])
                & (previous_shifts["week"] == shift["week"])
                & (previous_shifts["period"] == MORNING)
                & (previous_shifts["duration"] >= 12)
            ]
            cost[solution[this_weekend.index]] = LARGE_NUMBER

        # If someone is working on a shift that starts in the morning but
        # goes until the afternoon, than this person can't get another shift
        # in the afternoon
        if shift["period"] == AFTERNOON:
            morning_shifts = previous_shifts[
                (previous_shifts["date"].dt.date == shift["date"].date())
                & (previous_shifts["period"] == MORNING)
                & (previous_shifts["duration"] > 5)
            ]
            cost[solution[morning_shifts.index]] = LARGE_NUMBER

        # Dont assign evening shifts if an area tomorrow does not have
        # more employees available than necessary
        if shift["period"] == EVENING:
            tomorrow = shift["date"] + pd.offsets.Day(1)
            tomorrow_shifts = next_shifts[
                (next_shifts["date"].dt.date == tomorrow.date())
                & (
                    (
                        (next_shifts["period"].isin((AFTERNOON, EVENING)))
                        & next_shifts["must_be_filled"]
                    )
                    | (
                        (next_shifts["period"] == MORNING)
                        & (next_shifts["duration"] > 5)
                    )
                )
            ]
            n_necessary_employees = tomorrow_shifts.groupby("area").size()
            available_employees = self.employees[n_necessary_employees.index].sum()
            unavailable_employees = []

            # Account for employees that are working on evening shifts today
            today_shifts = previous_shifts[previous_shifts["date"] == shift["date"]]
            available_employees = (
                available_employees
                - self.employees.iloc[solution[today_shifts.index]]
                .select_dtypes(bool)
                .sum()
            ).dropna()
            unavailable_employees.extend(solution[today_shifts.index])

            previous_sum = -1
            updated_available_employees = available_employees
            while updated_available_employees.sum() != previous_sum:
                previous_sum = updated_available_employees.sum()

                # Account for employees that will be assigned to another area for sure
                for area in n_necessary_employees[
                    n_necessary_employees == updated_available_employees
                ].index:
                    employees = self.employees[self.employees[area]].index
                    unavailable_employees = np.concatenate(
                        (unavailable_employees, employees)
                    )

                unavailable_employees = np.unique(unavailable_employees)
                updated_available_employees = (
                    available_employees
                    - self.employees.iloc[unavailable_employees]
                    .select_dtypes(bool)
                    .sum()
                )
                updated_available_employees = updated_available_employees.dropna()

            cost[unavailable_employees] = LARGE_NUMBER

        # If this shift is part of the should be balanced category,
        # then people that had a lot of shifts in this area should not
        # be picked
        if shift["should_be_balanced"]:
            available = pd.Series(
                index=self.employees[self.employees[shift["area"]]].index
            )
            available = available.fillna(0)
            unique, counts = np.unique(
                solution[previous_shifts["area"] == shift["area"]], return_counts=True
            )
            available[unique] = counts
            available = available[(cost < LARGE_NUMBER)]
            cost[available.index] = available

        # # Employees should have (hopefully) at least one weekend free
        # # Therefore, the cost of choosing someone for a weekend shift who
        # # doesn't have a free weekend yet should be higher
        # if shift["is_weekend"]:
        #     # Find everyone who didn't have a free weekend
        #     weekend_shifts = previous_shifts[previous_shifts["is_weekend"]]

        #     if not weekend_shifts.empty:
        #         mask = cost < LARGE_NUMBER
        #         # mask = cost < LARGE_NUMBER - 1
        #         changeable = cost[mask].index

        #         # If someone is already working this weekend, they should be more likely
        #         # to work more on the weekend so others can have a free wekend
        #         this_weekend = weekend_shifts[weekend_shifts["week"] == shift["week"]]
        #         working_this_weekend = np.unique(solution[this_weekend.index])
        #         cost += LARGE_NUMBER / 3
        #         cost[np.intersect1d(changeable, working_this_weekend)] -= (
        #             LARGE_NUMBER / 3
        #         )

        #         # # Assign people from long night shifts to shifts in the next morning
        #         # if shift["period"] == MORNING:
        #         #     previous_night_shift = weekend_shifts[
        #         #         (weekend_shifts["week"] == shift["week"])
        #         #         & (weekend_shifts["date"].dt.day == (shift["date"].day - 1))
        #         #         & (weekend_shifts["period"] == EVENING)
        #         #     ]
        #         #     previous_night_shift = solution[previous_night_shift.index]
        #         #     not_in_previous_night_shift = [
        #         #         e for e in self.employees.index if e not in previous_night_shift
        #         #     ]
        #         #     cost[np.intersect1d(changeable, not_in_previous_night_shift)] = (
        #         #         LARGE_NUMBER - 1
        #         #     )
        #         #     cost[np.intersect1d(changeable, previous_night_shift)] = 0

        #         # Those who already had a free weekend should be more likely to be chosen
        #         weekend_shifts = weekend_shifts[weekend_shifts["week"] != shift["week"]]
        #         had_free_weekend = []
        #         for _, weekend_shift in weekend_shifts.groupby("week"):
        #             worked_weekend = np.unique(solution[weekend_shift.index])
        #             mask = [
        #                 employee not in worked_weekend
        #                 for employee in range(self.n_employees)
        #             ]
        #             had_free_weekend.append(self.employees[mask].index)
        #         if had_free_weekend:
        #             had_free_weekend = np.unique(np.concatenate(had_free_weekend))
        #             not_had_free_weekend = [
        #                 e for e in self.employees.index if e not in had_free_weekend
        #             ]
        #             cost[np.intersect1d(changeable, not_had_free_weekend)] += (
        #                 LARGE_NUMBER / 3
        #             )

        return cost

    def is_solution_valid(self, solutions: pd.DataFrame) -> bool:
        for j in solutions.columns:
            for i in self.shifts.index:
                shift = self.shifts.iloc[i]

                # Account for shifts without anyone assigned
                ii = solutions.loc[: i - 1, j] >= 0

                previous_shifts = self.shifts.loc[: i - 1]
                previous_shifts = previous_shifts[ii].reset_index(drop=True)

                solution = solutions.loc[: i - 1, j]
                solution = solution[ii].reset_index(drop=True)

                iii = solutions.loc[i + 1 :, j] >= 0
                next_shifts = self.shifts.iloc[i + 1 :]
                next_shifts = next_shifts[iii].reset_index(drop=True)

                cost = self.calculate_costs(
                    shift, solution, previous_shifts, next_shifts
                )

                current_solution = solutions.loc[i, j]

                if current_solution >= 0:
                    # The pick was not valid
                    if cost[solutions.loc[i, j]] >= LARGE_NUMBER:
                        return False
                else:
                    # Someone could work on this shift but wasnt assigned
                    if (cost < LARGE_NUMBER).any():
                        return False
        return True

    def calculate_scores(self, solutions: pd.DataFrame) -> np.array:
        scores = []
        for i in solutions.columns:
            solution = solutions[i]
            hours_worked = self.shifts.groupby(solution[solution >= 0])[
                "duration"
            ].sum()
            # scores.append(-hours_worked.max())
            scores.append(-np.var(hours_worked))
        return np.array(scores)


GA_params = dict(n_solutions=30)
manager = Manager(2023, 7, GA_params)

inputs_areas = pd.read_excel(
    r"C:\Users\brene\Dropbox\shift_manager\areas.xlsx", keep_default_na=False
)
for _, row in inputs_areas.iterrows():
    # _, row = next(areas.iterrows())
    row[5] = row[5].replace(" ", "").split(",")
    manager.add_shift_params(*row)
manager.shifts_params

employees = pd.read_excel(
    r"C:\Users\brene\Dropbox\shift_manager\employees.xlsx",
)
employees = employees.fillna(False)
for col in employees.columns[1:]:
    employees[col] = employees[col].astype(bool)
for _, row in employees.iterrows():
    name = row[0]
    manager.add_employee(name, employees.columns[1:][row[1:]].values)
inputs_areas

# %%

manager.create_schedule()

# %%


def find_conflicts(shift, shifts):
    conflicts = []

    # Can not work on two shifts at the same time
    conflicts.extend(shifts[(shift["date"] == shifts["date"])].index)

    # If evening shift, check if employee i is working
    # - yesterday on an evening shift
    # - tomorrow on an afternoon/evening shift
    # - tomorrow on a long morning shift if it is a weekend
    if shift["period"] == EVENING:
        conflicts.extend(
            shifts[
                (shift["date"] - pd.DateOffset(days=1) == (shifts["date"]))
                | (shift["date"] + pd.DateOffset(days=1) == (shifts["date"]))
                | (
                    (shift["date"].day + 1 == shifts["date"].dt.day)
                    & (shifts["period"] == AFTERNOON)
                )
                | (
                    (shift["date"].day + 1 == shifts["date"].dt.day)
                    & (shifts["period"] == MORNING)
                    & (shifts["duration"] >= 12)
                    & (shifts["is_weekend"])
                )
            ].index
        )

    # If long morning shift on weekend check if employee i is working
    # - yesterday on an evening shift
    # - today's evening shift
    if shift["period"] == MORNING and shift["duration"] >= 12:
        conflicts.extend(
            shifts[
                (
                    (shift["date"].day - 1 == shifts["date"].dt.day)
                    & (shifts["period"] == EVENING)
                )
                | (shift["date"].day == shifts["date"].dt.day)
            ].index
        )

    return conflicts


# %%
# if True:
seed(1)
shifts = manager.shifts.copy()
employees = manager.employees.copy()
employees["free_weekend"] = ""
weekends_shifts = shifts[shifts["is_weekend"]]
weeks = shifts[shifts["is_weekend"]]["week"].unique()
# Loop through employees

for i in employees.index:
    weekends_shifts = shifts[(shifts["is_weekend"]) & (shifts["employee_id"] == i)]
    if weekends_shifts.empty:
        continue

    sucessful_swaps = False
    while not sucessful_swaps:
        week = choices(weeks)[0]
        free_weekend_shifts = shifts[
            (shifts["is_weekend"])
            & (shifts["employee_id"] == i)
            & (shifts["week"] == week)
        ]

        # Loop through shifts this weekend and replace them
        for index in free_weekend_shifts.index:
            # Choose people that are already working the most this weekend
            shifts_this_weekend = shifts[
                (shifts["is_weekend"])
                & (shifts["employee_id"] != i)
                & (shifts["week"] == week)
            ]
            # If empty then try another weekend
            if shifts_this_weekend.empty:
                continue

            shift = free_weekend_shifts.loc[index]

            candidates = shifts[(shifts["area"] == shift["area"])][
                "employee_id"
            ].unique()
            candidates = candidates[employees.loc[candidates]["free_weekend"] != week]

            # If evening shift, dont consider people that are working on
            # tomorrow's or yesterday's evening shift
            exclude = []
            if shift["period"] == EVENING:
                exclude.extend(
                    shifts_this_weekend[
                        (
                            (
                                shifts_this_weekend["date"]
                                == (shift["date"] + pd.DateOffset(days=1))
                            )
                            | (
                                shifts_this_weekend["date"].dt.day
                                == (shift["date"] - pd.DateOffset(days=1))
                            )
                        )
                    ]["employee_id"].astype(int)
                )

            # If evening shift, don't consider people that worked on long morning
            # shifts today or tomorrow
            if shift["period"] == EVENING:
                exclude.extend(
                    shifts_this_weekend[
                        (
                            (shifts_this_weekend["date"].dt.day == shift["date"].day)
                            | (
                                shifts_this_weekend["date"].dt.day
                                == (shift["date"].day + 1)
                            )
                        )
                        & (shifts_this_weekend["period"] == MORNING)
                        & (shifts_this_weekend["duration"] >= 12)
                    ]["employee_id"].astype(int)
                )

            # If long morning shift, dont consider people that are working
            # on today's and yesterday's evening shift
            if shift["period"] == MORNING and shift["duration"] >= 12:
                exclude.extend(
                    shifts_this_weekend[
                        (shifts_this_weekend["date"].dt.day == (shift["date"].day - 1))
                        & (shifts_this_weekend["date"].dt.day == (shift["date"].day))
                        & (shifts_this_weekend["period"] == EVENING)
                    ]["employee_id"].astype(int)
                )

            exclude.extend(
                shifts_this_weekend[shifts_this_weekend["date"] == shift["date"]][
                    "employee_id"
                ].astype(int)
            )
            candidates = [c for c in candidates if c not in exclude]

            # Try to switch shifts with candidates
            swap = []
            for j in candidates:
                # List all shifts of the same kind of this candidate that are on other weekends
                shifts_swap = shifts[
                    (shifts["employee_id"] == j)
                    & (shifts["area"] == shift["area"])
                    & (shifts["duration"] == shift["duration"])
                    & ~(shifts["is_weekend"] & (shifts["week"] == shift["week"]))
                    & ~(
                        shifts["is_weekend"]
                        & (shifts["week"] == employees.loc[j, "free_weekend"])
                    )
                    & (shifts["date"] != shift["date"])
                ]

                shifts_i = shifts[(shifts["employee_id"] == i)]
                shifts_j = shifts[(shifts["employee_id"] == j)]

                for _, shift_ in shifts_swap.iterrows():
                    conflicts = find_conflicts(shift_, shifts_i)
                    conflicts.extend(find_conflicts(shift, shifts_j))
                    if conflicts:
                        continue
                    else:
                        swap.append(shift_)

            if swap:
                shift_to_swap = choices(swap)[0]
                index_swap = shifts[shifts["name"] == shift_to_swap["name"]].index[0]
                print("Swapping:")
                print(
                    f'{shifts.loc[index, ["name", "employee"]]} with \n{shifts.loc[index_swap, ["name", "employee"]]}\n\n'
                )
                (
                    shifts.loc[index, ["employee_id", "employee"]],
                    shifts.loc[index_swap, ["employee_id", "employee"]],
                ) = (
                    shifts.loc[index_swap, ["employee_id", "employee"]],
                    shifts.loc[index, ["employee_id", "employee"]],
                )
                sucessful_swaps = True
            else:
                sucessful_swaps = False
                print("\n\nDidnt work, trying next weekend\n\n")
                break

    employees.loc[i, "free_weekend"] = week

weekends = shifts[shifts["is_weekend"]].groupby("week")["employee"].unique()

l = []
for employee in employees["name"]:
    for weekend in weekends:
        if employee not in weekend:
            l.append(employee)

print([e for e in employees["name"] if e not in l])

employees.sort_values("hours_worked")
# shifts[shifts["employee"] == "Maria Guerra"]

# %%
# shifts = manager.shifts.copy()
shifts["day"] = shifts["date"].dt.day

shifts[(shifts["period"] == MORNING) & (shifts["duration"] > 5)]
shifts["sort_key"] = shifts["period"].map({"M": 0, "E": 1, "A": 2})
areas = shifts.sort_values("sort_key")["area"].unique()
areas

max_employees = inputs_areas["N_Residentes"].max()

data_all = []

for week_i in shifts.week.unique():
    shifts_week = shifts[shifts["week"] == week_i]
    for period_i, period in enumerate([MORNING, AFTERNOON, EVENING]):
        shifts_time = shifts_week[shifts_week["period"] == period]
        for area_i, area in enumerate(areas):
            shifts_area = shifts_time[shifts_time["area"] == area]
            if shifts_area.empty:
                continue
            for weekday_i in [0, 1, 2, 3, 4, 5, 6]:
                shifts_weekday = shifts_area[
                    shifts_area["date"].dt.weekday == weekday_i
                ]
                texts = ["-"]
                if not shifts_weekday.empty:
                    e = shifts_weekday["employee"].tolist()
                    until_afternoon = (shifts_weekday["duration"] > 5) & (
                        shifts_weekday["period"] == MORNING
                    )
                    texts = []
                    for name, flag in zip(e, until_afternoon):
                        if flag:
                            text = name + " (MANHÃ E TARDE)"
                        else:
                            text = name
                        texts.append(text)
                for i, text in enumerate(texts):
                    data_all.append([week_i, period_i, area, weekday_i, i, text])

df = pd.DataFrame(
    data_all,
    columns=["SEMANA", "PERIODO", "AREA", "DIA DA SEMANA", "i", "RESIDENTE"],
)

df.head(10)
df = df.pivot(
    index=["SEMANA", "PERIODO", "AREA", "i"],
    columns="DIA DA SEMANA",
    values="RESIDENTE",
)
df.columns = df.columns.map(
    {0: "SEG", 1: "TER", 2: "QUA", 3: "QUI", 4: "SEX", 5: "SAB", 6: "DOM"}
)
df.index = df.index.set_levels(["MANHA", "TARDE", "NOITE"], level=1)
df = df.fillna("-")
df

# People that are working in only one area
one_area_person = employees[employees.select_dtypes(bool).sum(axis=1) == 1]
one_area_person = one_area_person.select_dtypes(bool).sum() > 0
one_area_person = one_area_person[one_area_person].index
df = df.drop(level="AREA", index=one_area_person)

# Areas that have only one person and must be filled
one_person_area = employees.select_dtypes(bool).sum()
one_person_area = one_person_area[one_person_area == 1].index
must_not_be_filled = inputs_areas[inputs_areas["Obrigatorio"] != 1]["Area"]
one_person_area = one_person_area[~one_person_area.isin(must_not_be_filled)]
df = df.drop(level="AREA", index=one_person_area)

df.head(30)


# df.transform(lambda row: 'Leticia' in row.tolist() )
def filter_employee(name_to_filter, name):
    print(name_to_filter)
    if name == name_to_filter:
        return name
    else:
        return "-"


def add_to_format(existing_format, dict_of_properties, workbook):
    """Give a format you want to extend and a dict of the properties you want to
    extend it with, and you get them returned in a single format"""
    new_dict = {}
    for key, value in existing_format.__dict__.iteritems():
        if (value != 0) and (value != {}) and (value != None):
            new_dict[key] = value
    del new_dict["escapes"]

    return workbook.add_format(dict(new_dict.items() + dict_of_properties.items()))


def box(workbook, sheet_name, row_start, col_start, row_stop, col_stop):
    """Makes an RxC box. Use integers, not the 'A1' format"""

    rows = row_stop - row_start + 1
    cols = col_stop - col_start + 1

    for x in xrange((rows) * (cols)):  # Total number of cells in the rectangle
        box_form = workbook.add_format()  # The format resets each loop
        row = row_start + (x // cols)
        column = col_start + (x % cols)

        if x < (cols):  # If it's on the top row
            box_form = add_to_format(box_form, {"top": 1}, workbook)
        if x >= ((rows * cols) - cols):  # If it's on the bottom row
            box_form = add_to_format(box_form, {"bottom": 1}, workbook)
        if x % cols == 0:  # If it's on the left column
            box_form = add_to_format(box_form, {"left": 1}, workbook)
        if x % cols == (cols - 1):  # If it's on the right column
            box_form = add_to_format(box_form, {"right": 1}, workbook)

        sheet_name.write(row, column, "", box_form)


def format_sheet(sheet, df):
    sheet.autofit()
    for column in df:
        column_length = 30
        col_idx = df.columns.get_loc(column) + 4
        sheet.set_column(col_idx, col_idx, column_length)

    sheet.freeze_panes(1, 3)

    # Color by area
    colors = iter([plt.cm.tab20(i) for i in range(20)])
    color = defaultdict(lambda: next(colors))
    for i, (indexes, row) in enumerate(df.iterrows()):
        week, period, area, ii = indexes
        sheet.set_row(
            i + 1, 20, writer.book.add_format({"bg_color": rgb2hex(color[area])})
        )

    # Hide column of index i
    sheet.set_column("D:D", None, None, {"hidden": True})
    sheet.set_column("L:XFD", None, None, {"hidden": True})


with pd.ExcelWriter(r"../test.xlsx") as writer:
    df.to_excel(writer, sheet_name="Escala", index=True)
    format_sheet(writer.sheets["Escala"], df)

    for name in employees["name"]:
        df_ = df.apply(lambda row: row == name).apply(
            lambda col: col.map({False: "", True: name}), axis=1
        )
        df_2 = df.apply(lambda row: row == name + " (MANHÃ E TARDE)").apply(
            lambda col: col.map({False: "", True: name + " (MANHÃ E TARDE)"}),
            axis=1,
        )
        df_ = df_ + df_2
        df_ = df_[(~(df_ == ["", "", "", "", "", "", ""])).any(axis=1)]

        df_.to_excel(writer, sheet_name=name, index=True)
        format_sheet(writer.sheets[name], df_)


shifts[(shifts["area"] == "Puerperio") & (shifts["duration"] > 5)].groupby(
    "employee"
).size()
