# %%
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from random import shuffle
from collections import defaultdict
from matplotlib import pyplot as plt
from common_types import *
import pandas as pd
import numpy as np
from random import choices, seed, shuffle
from matplotlib.colors import rgb2hex

np.random.seed(1)
seed(1)


class Manager:
    def __init__(self, filepath: str) -> None:
        self.read_inputs(filepath)
        self.initialize_variables()
        self.add_shifts_params()
        self.add_employees()
        self.add_constraints()
        self.add_timeoff()
        self.create_shifts()

    def initialize_variables(self):
        self.__start_date = f"{self.i['year']}-{MONTHS_STR2NUM[self.i['month']]}"
        self.__employees = pd.DataFrame([])
        self.__shifts_params = list()
        self.__n_solutions = self.i["n_solutions"]
        self.__max_tries = 20
        self.__constraints = pd.DataFrame([])
        self.__shifts = pd.DataFrame(
            [], columns=["name", "area", "date", "duration", "period"]
        )
        self.__areas = []
        self.__timeoff = pd.Series([])

    def read_inputs(self, filepath: str):
        with open(filepath, "r") as f:
            self.i = json.load(f)

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

    @property
    def timeoff(self):
        return self.__timeoff

    @property
    def areas(self):
        return self.__areas

    def add_timeoff(self):
        df = pd.DataFrame(self.i["timeoff"])
        df["Period_Time"] = df[PERIOD].map(
            lambda x: START_TIMES_STR[PERIOD_TRANSLATE[x]]
        )
        df["start"] = pd.to_datetime(
            df[DATE] + " " + df["Period_Time"], format=r"%d/%m/%Y %H:%M"
        )
        df["duration"] = df[PERIOD].map(
            lambda x: pd.offsets.Hour(DURATION_TIMES[PERIOD_TRANSLATE[x]])
        )
        df["end"] = df["start"] + df["duration"]
        intervals = df.apply(
            lambda row: pd.Interval(row["start"], row["end"], closed="left"), axis=1
        )
        index = df[NAME].map(lambda x: self.i["employees"].index(x))
        self.__timeoff = pd.Series(intervals.values, index=index)

    def add_shifts_params(self) -> None:
        shift = self.i["shifts"][0]
        for shift in self.i["shifts"]:
            name = shift[NAME]
            area = shift[AREA]
            duration = float(shift[DURATION])
            period = PERIOD_TRANSLATE[shift[PERIOD]]
            n_employees = int(shift[N_EMPLOYEES])
            days_of_week = shift[DAYS].split(", ")
            should_be_balanced = True if shift[BALANCE] == YES else False
            must_be_filled = True if shift[MANDATORY] == YES else False
            exclude_dates = (
                shift[EXCEPTIONS] if isinstance(shift[EXCEPTIONS], list) else [""]
            )

            assert all(day in DAYS_OF_WEEK for day in days_of_week)
            assert period in PERIODS

            # Get dates of shifts in this area
            dates = pd.DatetimeIndex([])
            start_date = pd.to_datetime(
                f"{self.__start_date}-1 {START_TIMES_STR[period]}"
            )
            end_date = start_date + pd.offsets.MonthEnd()
            for day in days_of_week:
                dates_ = pd.date_range(
                    start_date,
                    end_date,
                    freq=f"W-{DAYS_PORT2ENG[day]}",
                )
                dates = dates.union(dates_)
            for date in exclude_dates:
                if date == "":
                    continue
                print(f"Excluding {date} for shift {name}")
                date = pd.to_datetime(f"{date} {START_TIMES_STR[period]}")
                dates = dates[~(dates == date)]

            self.__shifts_params.append(
                dict(
                    name=name,
                    area=area,
                    duration=duration,
                    period=period,
                    dates=dates,
                    should_be_balanced=should_be_balanced,
                    must_be_filled=must_be_filled,
                    n_employees=n_employees,
                )
            )
            if area not in self.__areas:
                self.__areas.append(area)

    # def add_employee(self, employee: Employee) -> None:
    def add_employees(self) -> None:
        for name in self.i["employees"]:
            emp_areas = [
                area for area, flag in self.i["employees_areas"][name].items() if flag
            ]
            # Also add sub areas
            emp_areas = [area for area in self.areas if area.split("_")[0] in emp_areas]
            message = (
                "Employee area not registered on manager. \n\n"
                + f"Available areas: {self.areas} \n\n"
                + f"Employee areas: {emp_areas}"
            )
            assert all(area in self.areas for area in emp_areas), message
            employee = dict(name=[name], hours_worked=[0])
            employee.update({area: [area in emp_areas] for area in self.areas})

            self.__employees = pd.concat(
                [self.__employees, pd.DataFrame(employee)]
            ).reset_index(drop=True)

    def add_constraints(self):
        df = pd.DataFrame(self.i["restrictions"])
        df[MAX_SHIFTS] = df[MAX_SHIFTS].astype(int)
        self.__constraints = df

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
        shifts.sort(key=lambda x: (x["date"].day, x["date"].hour, x["duration"]))

        shifts = pd.DataFrame(shifts)
        shifts["employee"] = ""
        shifts["employee_id"] = ""
        self.__shifts = shifts

    def find_conflicts(self, shift, shifts, timeoff):
        conflicts = []

        # Can not work on two shifts at the same time
        conflicts.extend(shifts[(shift["date"] == shifts["date"])].index)

        # Can not work on shifts that are on timeoff
        if not timeoff.empty:
            shift_start = shift["date"]
            shift_end = (
                shift["date"]
                + pd.offsets.Hour(shift["duration"] // 1)
                + pd.offsets.Minute((shift["duration"] % 1) * 60)
            )
            shift_interval = pd.Interval(shift_start, shift_end)
            i_timeoff = [
                i
                for i, interval in timeoff.items()
                if shift_interval.overlaps(interval)
            ]
            if len(i_timeoff) > 0:
                conflicts.extend([0])

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
                        & (shifts["duration"] > 5)
                        # & (shifts["is_weekend"])
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

    def create_schedule(self, best: dict = None) -> None:
        if best is None:
            best = self.find_solution()

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

        employees_stats = pd.merge(
            self.employees[["name"]], n_shifts, left_index=True, right_index=True
        )
        employees_stats = pd.merge(
            employees_stats, hours_worked, left_index=True, right_index=True
        )
        self.stats = employees_stats

        self.drop_extra_shifts()

        self.create_free_weekends()

    def create_free_weekends(self) -> None:
        # Create free weekends
        seed(1)
        shifts = self.shifts.copy()
        stats = self.stats
        stats["free_weekend"] = ""
        weekends_shifts = shifts.loc[shifts["is_weekend"]]

        # Only consider a weekend if there are at least 2 days
        weeks_all = shifts.loc[shifts["is_weekend"]]["week"].unique().tolist()
        weeks = []
        for week in weeks_all:
            days_in_weekend = weekends_shifts[weekends_shifts["week"] == week][
                "date"
            ].dt.day.unique()
            if days_in_weekend.size > 1:
                weeks.append(week)

        # Loop through employees
        pool = stats.index[:-1]
        counter = defaultdict(lambda: 0)
        while not pool.empty:
            i = choices(pool)[0]
            weekends_shifts = shifts[
                (shifts["is_weekend"]) & (shifts["employee_id"] == i)
            ]

            if weekends_shifts.empty:
                pool = pool.drop(i)
                continue

            if counter[stats.loc[i, "name"]] > 50:
                print(f"Could not find a free weekend for {stats.loc[i, 'name']}")
                pool = pool.drop(i)

            sucessful_swaps = False
            week_pool = weeks.copy()
            shuffle(week_pool)
            while not sucessful_swaps:
                # Try other employee if week pool is empty
                if len(week_pool) == 0:
                    counter[stats.loc[i, "name"]] += 1
                    break

                week = week_pool.pop()
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

                    candidates = shifts[
                        (shifts["area"] == shift["area"]) & (shifts["employee_id"] != i)
                    ]["employee_id"].unique()
                    candidates = candidates[
                        stats.loc[candidates]["free_weekend"] != week
                    ]

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
                                    (
                                        shifts_this_weekend["date"].dt.day
                                        == shift["date"].day
                                    )
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
                                (
                                    shifts_this_weekend["date"].dt.day
                                    == (shift["date"].day - 1)
                                )
                                & (
                                    shifts_this_weekend["date"].dt.day
                                    == (shift["date"].day)
                                )
                                & (shifts_this_weekend["period"] == EVENING)
                            ]["employee_id"].astype(int)
                        )

                    exclude.extend(
                        shifts_this_weekend[
                            shifts_this_weekend["date"] == shift["date"]
                        ]["employee_id"].astype(int)
                    )
                    candidates = [c for c in candidates if c not in exclude]

                    # Try to switch shifts with candidates
                    swap = []
                    for j in candidates:
                        # List all shifts of the same kind for this candidate that are not on this weekend or on their free weekend
                        shifts_swap = shifts[
                            (shifts["employee_id"] == j)
                            & (shifts["area"] == shift["area"])
                            & (shifts["duration"] == shift["duration"])
                            & ~(
                                shifts["is_weekend"] & (shifts["week"] == shift["week"])
                            )
                            & ~(
                                shifts["is_weekend"]
                                & (shifts["week"] == stats.loc[j, "free_weekend"])
                            )
                            & (shifts["date"] != shift["date"])
                        ]

                        shifts_i = shifts[(shifts["employee_id"] == i)]
                        shifts_j = shifts[(shifts["employee_id"] == j)]
                        timeoff_i = pd.Series(self.timeoff.get(i))
                        timeoff_j = pd.Series(self.timeoff.get(j))

                        for _, shift_ in shifts_swap.iterrows():
                            conflicts = self.find_conflicts(shift_, shifts_i, timeoff_i)
                            conflicts.extend(
                                self.find_conflicts(shift, shifts_j, timeoff_j)
                            )
                            if conflicts:
                                continue
                            else:
                                swap.append(shift_)

                    if swap:
                        shift_to_swap = choices(swap)[0]
                        index_swap = shifts[
                            shifts["name"] == shift_to_swap["name"]
                        ].index[0]
                        # print("Swapping:")
                        # print(
                        #     f'{shifts.loc[index, ["name", "employee"]]} with \n{shifts.loc[index_swap, ["name", "employee"]]}\n\n'
                        # )
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
                        # print("\n\nDidnt work, trying next weekend\n\n")
                        break

                if free_weekend_shifts.empty:
                    sucessful_swaps = True

            if sucessful_swaps:
                stats.loc[i, "free_weekend"] = week
                print(f"Found free weekend for {stats.loc[i,'name']}")
                pool = pool.drop(i)

        weekends = shifts[shifts["is_weekend"]].groupby("week")["employee"].unique()

        stats["free_weekend"] = ""
        for i, employee in stats["name"].items():
            for j, weekend in weekends.items():
                if employee not in weekend:
                    stats.loc[i, "free_weekend"] += f"{j} "

        self.stats = stats
        self.__shifts = shifts

    def find_solution(self):
        # Initialize solutions and scores
        solutions = self.create_solutions(self.__n_solutions)
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

    def create_solution(self):
        shifts = self.shifts.loc[::-1].reset_index()
        solution = np.full((shifts.shape[0]), np.nan)
        solution = pd.Series(solution)

        i = 0
        c = 0
        while i < shifts.shape[0]:
            shift = shifts.iloc[i]

            # # Do not consider shifts without solutions
            ii = solution.loc[: i - 1] >= 0
            next_shifts = shifts.loc[: i - 1]
            current_solution = solution.loc[: i - 1]
            next_shifts = next_shifts[ii].reset_index(drop=True)
            current_solution = current_solution[ii].reset_index(drop=True)

            cost_matrix = self.calculate_costs(
                shift,
                current_solution,
                next_shifts,
            )

            # If there are no eligible employees, start again
            if (cost_matrix >= LARGE_NUMBER).all():
                print(f"Trying generating another initial solution at step {i}")
                print(shift["name"])
                c += 1
                if c > self.__max_tries:
                    raise RuntimeError(
                        "Could not find solution that respected all rules"
                    )
                i = next_shifts[next_shifts["date"].dt.day == (shift["date"].day - 7)]
                i = 0 if i.empty else i.index[0]
                continue
            else:
                # The eligible employees with the most amount of hours worked
                # will not be considered
                mask = cost_matrix < LARGE_NUMBER
                if (cost_matrix[mask].min() == cost_matrix[mask]).sum() > 1:
                    solution.loc[i] = choices(
                        cost_matrix[cost_matrix == cost_matrix.min()].index
                    )[0]
                else:
                    solution.loc[i] = np.argmin(cost_matrix)

            # Find optional shifts that should not be worked
            next_shifts_all = shifts.iloc[:i] if i > 0 else pd.DataFrame()
            if shift["period"] == EVENING and not next_shifts_all.empty:
                optional = next_shifts_all[
                    (next_shifts_all["date"].dt.day == (shift["date"].day + 1))
                    & (
                        (next_shifts_all["period"] == AFTERNOON)
                        | (next_shifts_all["period"] == EVENING)
                    )
                    & (~next_shifts_all["must_be_filled"])
                ]
                if not optional.empty:
                    employee = solution.loc[i]
                    for ii, _ in optional.iterrows():
                        optional_employee = solution.loc[ii]
                        if optional_employee == employee:
                            solution.loc[ii] = -1

            i += 1
        return solution.loc[::-1].reset_index(drop=True)

    def create_solutions(self, n_solutions: int) -> pd.DataFrame:
        shifts = self.shifts.loc[::-1].reset_index()
        solutions = np.full((shifts.shape[0], n_solutions), np.nan)
        solutions = pd.DataFrame(solutions)

        for j in range(n_solutions):
            solutions.loc[:, j] = self.create_solution()
        return solutions

    def calculate_costs(
        self,
        shift: pd.DataFrame,
        solution: np.array = None,
        next_shifts: pd.DataFrame = None,
    ) -> pd.Series:
        cost = np.zeros(self.n_employees)

        # Only work on right areas
        cost += (~self.employees[shift["area"]]) * LARGE_NUMBER

        if next_shifts.empty or solution is None:
            return cost

        # Try to not assign someone who is in timeoff
        shift_start = shift["date"]
        shift_end = (
            shift["date"]
            + pd.offsets.Hour(shift["duration"] // 1)
            + pd.offsets.Minute((shift["duration"] % 1) * 60)
        )
        shift_interval = pd.Interval(shift_start, shift_end)
        i_timeoff = [
            i
            for i, interval in self.timeoff.items()
            if shift_interval.overlaps(interval)
        ]
        i_timeoff = np.array(i_timeoff)
        if i_timeoff.size > 0:
            cost[i_timeoff] = LARGE_NUMBER - 1

        # Add hours worked
        hours_worked = next_shifts.groupby(solution)["duration"].sum()
        cost[hours_worked.index] += hours_worked

        # Only one shift per employee
        same_time_shifts = next_shifts[next_shifts["date"] == shift["date"]]
        cost[solution[same_time_shifts.index]] = LARGE_NUMBER

        # Do not assign employees from areas that need all of them
        # in shifts at the same time
        same_time_previous = self.shifts[
            (self.shifts.date == shift.date) & (self.shifts.index < shift["index"])
        ]
        if shift["period"] == AFTERNOON:
            long_morning_shifts = self.shifts[
                (self.shifts["date"].dt.day == shift["date"].day)
                & (self.shifts["period"] == MORNING)
                & (self.shifts["duration"] > 5)
            ]

            same_time_previous = pd.concat((same_time_previous, long_morning_shifts))
            same_time_previous = same_time_previous.sort_index()

        areas = self.employees[self.employees[shift["area"]]][
            self._Manager__areas
        ].sum()
        areas = areas[areas > 0].index
        same_time_previous = same_time_previous[same_time_previous["area"].isin(areas)]
        if not same_time_previous.empty:
            # Create pool with employees that can work on same_time_previous
            i_areas = same_time_previous["area"].str.split("_").str[0].unique()
            pool = self.employees[self.employees[i_areas].sum(axis=1) > 0]

            # Do not consider the ones who are already working now
            pool = pool.drop(index=solution[same_time_shifts.index], errors="ignore")

            # Loop through shifts and take employees out of the pool
            for _, shift_ in same_time_previous.iterrows():
                i = pool[pool[shift_["area"]]]
                pool = pool.drop(index=i.index[0], errors="ignore")

            # Get employees from areas that are no in the pool anymore
            # and make them unselectable
            areas_exclude = pool[self._Manager__areas].sum()
            areas_exclude = areas_exclude[areas_exclude <= 0].index
            areas_exclude = areas_exclude[areas_exclude.isin(i_areas)].values

            i_exclude = self.employees[
                self.employees[areas_exclude].sum(axis=1) > 0
            ].index
            cost[i_exclude] = LARGE_NUMBER

        # If working tomorrow evening, afternoon or long morning,
        # can not work this evening
        if shift["period"] in (EVENING):
            tomorrow_evening = shift["date"] + pd.offsets.Day(1)
            tomorrow_morning = tomorrow_evening.replace(hour=START_TIMES[MORNING])
            tomorrow_afternoon = tomorrow_evening.replace(hour=START_TIMES[AFTERNOON])
            tomorrow_shifts = next_shifts[
                (
                    (next_shifts["date"] == tomorrow_evening)
                    & (next_shifts["must_be_filled"])
                )
                | (
                    (next_shifts["date"] == tomorrow_afternoon)
                    & next_shifts["must_be_filled"]
                )
                | (
                    (next_shifts["date"] == tomorrow_morning)
                    & (next_shifts["duration"] > 5)
                    & (next_shifts["must_be_filled"])
                )
            ]
            cost[solution[tomorrow_shifts.index]] = LARGE_NUMBER

        # On weekends, can not work evening shift if worked long morning shift
        if (
            shift["period"] == MORNING
            and shift["is_weekend"]
            and shift["duration"] >= 12
        ):
            evening_shifts = next_shifts[
                (next_shifts["date"].dt.day == shift["date"].day)
                & (next_shifts["period"] == EVENING)
            ]
            cost[solution[evening_shifts.index]] = LARGE_NUMBER

        # If working in the afternoon, can not work on long morning shifts
        if shift["period"] == MORNING and shift["duration"] > 5:
            afternoon_shifts = next_shifts[
                (next_shifts["date"].dt.date == shift["date"].date())
                & (next_shifts["period"] == AFTERNOON)
            ]
            cost[solution[afternoon_shifts.index]] = LARGE_NUMBER

        # If this shift is part of the should be balanced category,
        # then people that had a lot of shifts in this area should not
        # be picked
        if shift["should_be_balanced"]:
            available = pd.Series(
                index=self.employees[self.employees[shift["area"]]].index
            )
            available = available.fillna(0)
            unique, counts = np.unique(
                solution[next_shifts["area"] == shift["area"]], return_counts=True
            )
            available[unique] = counts
            available = available[(cost < LARGE_NUMBER - 1)]
            cost[available.index] = available

        return cost

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

    def export_results(self, path: str) -> None:
        shifts = self.shifts.copy()
        employees = self.employees
        shifts["sort_key"] = shifts["period"].map({"M": 0, "E": 1, "A": 2})
        areas = shifts.sort_values("sort_key")["area"].unique()

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
                            data_all.append(
                                [week_i, period_i, area, weekday_i, i, text]
                            )

        df = pd.DataFrame(
            data_all,
            columns=["SEMANA", "PERIODO", "AREA", "DIA DA SEMANA", "i", "RESIDENTE"],
        )

        df = df.pivot(
            index=["SEMANA", "PERIODO", "AREA", "i"],
            columns="DIA DA SEMANA",
            values="RESIDENTE",
        )
        df.columns = df.columns.map(
            dict(zip(range(7), DAYS_OF_WEEK))
            # {0: "SEG", 1: "TER", 2: "QUA", 3: "QUI", 4: "SEX", 5: "SAB", 6: "DOM"}
        )
        df.index = df.index.set_levels(["MANHA", "TARDE", "NOITE"], level=1)
        df = df.fillna("-")

        # People that are working in only one area
        one_area_person = employees[employees.select_dtypes(bool).sum(axis=1) == 1]
        one_area_person = one_area_person.select_dtypes(bool).sum() > 0
        one_area_person = one_area_person[one_area_person].index
        df = df.drop(level="AREA", index=one_area_person)

        # Areas that have only one person and must not be filled
        one_person_area = employees.select_dtypes(bool).sum()
        one_person_area = one_person_area[one_person_area == 1].index
        not_mandatory = [p[AREA] for p in self.i["shifts"] if p[MANDATORY] != YES]
        one_person_area = one_person_area[~one_person_area.isin(not_mandatory)]
        df = df.drop(level="AREA", index=one_person_area)

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
                    i + 1,
                    20,
                    writer.book.add_format({"bg_color": rgb2hex(color[area])}),
                )

            # Hide column of index i
            sheet.set_column("D:D", None, None, {"hidden": True})
            sheet.set_column("L:XFD", None, None, {"hidden": True})

        file = f"{OUTPUT_SHIFTS_NAME}_{self.i['month']}.xlsx"
        with pd.ExcelWriter(os.path.join(path, file)) as writer:
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

        file = f"{OUTPUT_STATS_NAME}_{self.i['month']}.csv"
        self.stats.sort_values("hours_worked").to_csv(os.path.join(path, file))

    def drop_extra_shifts(self):
        constraints = self.__constraints
        self.stats.loc[self.n_employees] = 0
        self.stats.loc[self.n_employees, "name"] = "R+"

        for _, constraint in constraints.iterrows():
            area = constraint["Área"]
            max_shifts = constraint[MAX_SHIFTS]
            employees = self.stats[self.stats[area] > max_shifts]
            n_extra = employees[area] - max_shifts
            for i, n in n_extra.items():
                shifts = self.shifts[
                    (self.shifts["employee_id"] == i) & (self.shifts["area"] == area)
                ]

                # Drop shifts that have only one employee
                single_e_dates = (
                    self.shifts[self.shifts["area"] == area].groupby("date").size()
                )
                single_e_dates = single_e_dates[single_e_dates < 2].index
                shifts = shifts[~(shifts["date"].isin(single_e_dates))]

                indexes = shifts.index.values.copy()
                shuffle(indexes)
                i_exclude = indexes[: int(n)]
                self.__shifts.loc[i_exclude, ["employee", "employee_id"]] = [
                    "R+",
                    self.n_employees,
                ]

                self.stats.loc[i, "hours_worked"] -= shifts.loc[i_exclude][
                    "duration"
                ].sum()
                self.stats.loc[i, area] -= n
                self.stats.loc[self.n_employees, area] += n
                self.stats.loc[self.n_employees, "hours_worked"] += shifts.loc[
                    i_exclude
                ]["duration"].sum()
