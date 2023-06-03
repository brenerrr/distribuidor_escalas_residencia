# %%

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
        must_be_filled: bool = True,
        exclude_dates: list = None,
    ) -> None:
        assert all(day in DAYS_OF_WEEK for day in days_of_week)
        assert period in PERIODS

        # Get dates of shifts in this area
        dates = pd.DatetimeIndex([])
        start_date = pd.to_datetime(f"{self.__start_date} {START_TIMES_STR[period]}")
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
                must_be_filled=(not must_be_filled == ""),
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
        self.__shifts["employee"] = self.employees.iloc[best["solution"]]["name"].values
        self.__shifts["employee_id"] = best["solution"]

        self.__employees["hours_worked"] = self.shifts.groupby(best["solution"])[
            "duration"
        ].sum()

    def perform_GA(self):
        self.create_shifts()

        # Initialize solutions and scores
        solutions = self.create_solutions(self.__GA_params["n_solutions"])
        scores = self.calculate_scores(solutions)

        max_scores = []
        mean_scores = []
        best = {"score": -np.inf, "solution": None}
        max_mating = 1000
        counter = 0
        early_stop_n = 15
        previous_mean = 0
        # Loop through n iterations
        # for n in range(self.__GA_params["n_iterations"]):
        #     print(n, scores)

        #     if n / self.__GA_params["n_iterations"] * 100 % 10 == 0:
        #         print(n)
        #     best_solutions_i = self.pick_best_solutions(scores)
        #     best_solutions = solutions[best_solutions_i]

        #     # Crossover and Mutation are represented by creating a new
        #     # solution with the best solutions as a guide. Doing it this way
        #     # guarantees that the new solution will be valid. The solution
        #     # will likely have new traits because sometimes a different gene
        #     # will be chosen in order to make sure the solution is valid
        #     new_solutions = self.create_solutions(1, best_solutions)

        #     # Replace worst solutions
        #     worst_i = np.argsort(scores)[: new_solutions.shape[1]]
        #     solutions[worst_i] = new_solutions

        #     # Calculate score
        #     scores = self.calculate_scores(solutions)
        #     # scores = self.calculate_scores(solutions, shifts)

        #     # Early stop
        #     if abs(scores.max() - previous_mean) < 1e-3:
        #         counter += 1
        #     else:
        #         counter = 0

        #     if counter > early_stop_n:
        #         break

        #     previous_mean = scores.mean()

        #     # Logging statistics
        #     mean_scores.append(scores.mean())
        #     max_scores.append(scores.max())

        #     if scores.max() > best["score"]:
        #         i = scores.argmax()
        #         best["score"] = scores[i]
        #         best["solution"] = solutions.loc[:, i]

        if scores.max() > best["score"]:
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
            while (solutions.loc[:, j].isnull().values).any():
                solutions.loc[:, j] = np.nan

                # Loop through shifts
                for i in shifts.index:
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

                    # Choose person with least amount of work so far
                    weights = cost_matrix.copy()

                    # Guides are solutions used as guides when deciding which employee
                    # should be chosen. In case it is possible to choose an employee
                    # that is on guides, they will be chosen.
                    if guide is not None:
                        desired_solution = guide.loc[i]
                        desired_solution = desired_solution[desired_solution >= 0]
                        if (weights[desired_solution] < LARGE_NUMBER).all():
                            weights[desired_solution] = weights.min()

                    # If there are no eligible employees, start again
                    if (cost_matrix >= LARGE_NUMBER).all():
                        if shift["must_be_filled"]:
                            c += 1
                            print(f"Trying generating another initial solution {c} {i}")
                            break
                        else:
                            solutions.loc[i, j] = -1
                    else:
                        # The eligible employees with the most amount of hours worked
                        # will not be considered
                        mask = weights < LARGE_NUMBER
                        if (cost_matrix[mask].max() == cost_matrix[mask]).sum() > 1:
                            weights[mask] -= weights[mask].max()
                            weights[mask] *= -1
                            weights[mask] += 1e-3  # Avoid dividing by zero
                            weights *= mask
                            weights = weights / weights.sum()
                            solutions.loc[i, j] = choices(
                                self.employees.index, weights
                            )[0]
                        else:
                            solutions.loc[i, j] = np.argmin(weights)

                    pass
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

        # # Only two evening shifts are allowed per week
        # if shift["period"] == EVENING:
        #     week_start = shift["date"] - pd.offsets.Day(shift["date"].weekday())
        #     week_start = max(week_start, previous_shifts["date"].min())
        #     evening_shifts = previous_shifts[
        #         (previous_shifts["week"] == shift["week"])
        #         & (previous_shifts["period"] == EVENING)
        #     ]
        #     if not evening_shifts.empty:
        #         n_shifts = evening_shifts.groupby(solution[evening_shifts.index]).size()
        #         n_shifts = n_shifts[n_shifts >= 2]
        #         cost[n_shifts.index] = LARGE_NUMBER

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

            # Account for employees that are working on evening shifts today
            today_shifts = previous_shifts[previous_shifts["date"] == shift["date"]]
            employees_today = self.employees.iloc[solution[today_shifts.index]]
            available_employees = (
                available_employees - employees_today[n_necessary_employees.index].sum()
            )

            areas = n_necessary_employees.index[
                available_employees == n_necessary_employees
            ]
            for area in areas:
                cost[self.employees[area]] = LARGE_NUMBER

        # Employees should have (hopefully) at least one weekend free
        # Therefore, the cost of choosing someone for a weekend shift who
        # doesn't have a free weekend yet should be higher
        if shift["is_weekend"]:
            # Find everyone who didn't have a free weekend
            weekend_shifts = previous_shifts[previous_shifts["is_weekend"]]

            if not weekend_shifts.empty:
                mask = cost < LARGE_NUMBER
                changeable = cost[mask].index

                # If someone is already working this weekend, they should be more likely
                # to work more on the weekend so others can have a free wekend
                this_weekend = weekend_shifts[weekend_shifts["week"] == shift["week"]]
                working_this_weekend = np.unique(solution[this_weekend.index])
                cost[np.intersect1d(changeable, working_this_weekend)] = 0

                # Those who already had a free weekend should be more likely to me chosen
                weekend_shifts = weekend_shifts[weekend_shifts["week"] != shift["week"]]
                had_free_weekend = []
                for _, weekend_shift in weekend_shifts.groupby("week"):
                    worked_weekend = np.unique(solution[weekend_shift.index])
                    mask = [
                        employee not in worked_weekend
                        for employee in range(self.n_employees)
                    ]
                    had_free_weekend.append(self.employees[mask].index)
                if had_free_weekend:
                    had_free_weekend = np.unique(np.concatenate(had_free_weekend))
                    cost[np.intersect1d(changeable, had_free_weekend)] = 0

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
            scores.append(-hours_worked.max())
            # scores.append(-np.var(hours_worked))
        return np.array(scores)


if __name__ == "__main__":
    GA_params = dict(n_solutions=1, n_iterations=100, mutation_rate=0.001)
    manager = Manager(2023, 7, GA_params)
    # manager.add_area("Plantao_Noturno", 12, "E", 2, DAYS_OF_WEEK)
    # manager.add_area("Plantao_Diurno", 12, "M", 2, ["SAT", "SUN"])
    # manager.add_area("Puerperio_Manha1", 5, "M", 4, ["MON", "WED"])

    # manager.add_area("Puerperio_Manha2", 5, "M", 3, ["TUE", "THU", "FRI"])
    # manager.add_area("Puerperio_Ambulatorio", 5 + 3, "M", 1, ["TUE", "THU", "FRI"])

    inputs_areas = pd.read_csv(
        r"C:\Users\brene\Dropbox\shift_manager\areas.csv", keep_default_na=False
    )
    for _, row in inputs_areas.iterrows():
        # _, row = next(areas.iterrows())
        row[5] = row[5].replace(" ", "").split(",")
        manager.add_shift_params(*row)
    manager.shifts_params

    employees = pd.read_csv(
        r"C:\Users\brene\Dropbox\shift_manager\employees.csv",
    )
    employees = employees.fillna(False)
    for col in employees.columns[1:]:
        employees[col] = employees[col].astype(bool)
    for _, row in employees.iterrows():
        name = row[0]
        manager.add_employee(name, employees.columns[1:][row[1:]].values)
    manager.employees

    manager.create_schedule()

    shifts = manager.shifts
    shifts[shifts["employee"] == "Iara"]
    manager.employees.hours_worked.plot(kind="hist")

    manager.employees.sort_values("hours_worked")

    shifts[shifts["employee"] == "Maria Guerra"]
    shifts[shifts["employee"] == "Vitoria"]

    shifts["day"] = shifts["date"].dt.day
    shifts.to_csv("escala.csv")
    shifts[shifts["employee"] == "Maria Guerra"].to_csv("escala_Maria.csv")

    shifts[shifts["employee"] == "Isabelle Amorim"]

    def color_per_day(row):
        value = (
            "background-color: AliceBlue;"
            if ((row.day) % 2 == 0)
            else "background-color: Beige;"
        )
        return np.repeat(value, row.size)

    shifts[shifts["employee"] == "Isabelle Amorim"].style.apply(
        color_per_day, axis=1
    ).set_properties(color="black")

    # %%

    shifts[(shifts["period"] == MORNING) & (shifts["duration"] > 5)]
    shifts["sort_key"] = shifts["period"].map({"M": 0, "E": 1, "A": 2})
    areas = shifts.sort_values("sort_key")["area"].unique()
    areas

    data_all = []
    shifts.week.unique()

    for week_i in shifts.week.unique():
        shifts_week = shifts[shifts["week"] == week_i]
        for period_i, period in enumerate([MORNING, AFTERNOON, EVENING]):
            shifts_time = shifts_week[shifts_week["period"] == period]
            for area_i, area in enumerate(areas):
                shifts_area = shifts_time[shifts_time["area"] == area]
                if shifts_area.empty:
                    continue
                data = ["-", "-", "-", "-", "-", "-", "-"]
                for weekday_i in [0, 1, 2, 3, 4, 5, 6]:
                    shifts_weekday = shifts_area[
                        shifts_area["date"].dt.weekday == weekday_i
                    ]
                    if not shifts_weekday.empty:
                        employees = shifts_weekday["employee"].tolist()
                        until_afternoon = (shifts_weekday["duration"] > 5) & (
                            shifts_weekday["period"] == MORNING
                        )
                        texts = []
                        for employee, flag in zip(employees, until_afternoon):
                            if flag:
                                text = employee + "(TARDE)"
                            else:
                                text = employee
                            texts.append(text)
                        data[weekday_i] = ", ".join(texts)
                data.extend([area, period, week_i])
                data_all.append(data)

    df = pd.DataFrame(
        data_all,
        columns=[
            "SEG",
            "TER",
            "QUAR",
            "QUI",
            "SEX",
            "SAB",
            "DOM",
            "AREA",
            "PERIODO",
            "SEMANA",
        ],
    )
    df = df.set_index(["SEMANA", "PERIODO", "AREA"])

    # People that is working in only one area
    one_area_person = manager.employees[
        manager.employees.select_dtypes(bool).sum(axis=1) == 1
    ]
    one_area_person = one_area_person.select_dtypes(bool).sum() > 0
    one_area_person = one_area_person[one_area_person].index
    df = df.drop(level="AREA", index=one_area_person)

    # Areas that have only one person and must be filled
    one_person_area = manager.employees.select_dtypes(bool).sum()
    one_person_area = one_person_area[one_person_area == 1].index
    must_not_be_filled = inputs_areas[inputs_areas["Obrigatorio"] == ""]["Area"]
    one_person_area = one_person_area[~one_person_area.isin(must_not_be_filled)]
    df = df.drop(level="AREA", index=one_person_area)

    df.head(30)
    df.to_excel("test.xlsx")
