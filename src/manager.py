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

np.random.seed(1)


class Manager:
    def __init__(self, year: int, month: int, GA_params: dict) -> None:
        self.__start_date = f"{year}-{month}"
        # self.__employees = list()
        self.__employees = pd.DataFrame([])
        self.__areas = list()
        self.__GA_params = GA_params
        self.__shifts = pd.DataFrame(
            [], columns=["name", "area", "date", "duration", "period"]
        )

    @property
    def areas(self):
        return self.__areas

    @property
    def employees(self):
        return self.__employees

    @property
    def n_employees(self):
        return self.__employees.shape[0]

    @property
    def shifts(self):
        return self.__shifts

    def add_area(
        self,
        name: str,
        duration: int,
        period: PERIODS_TYPE,
        n_employees: int,
        days_of_week: list,
        exclude_dates: list = None,
    ) -> None:
        assert all(day in DAYS_OF_WEEK for day in days_of_week)
        assert period in PERIODS

        # Get dates of shifts in this area
        dates = pd.DatetimeIndex([])
        start_date = pd.to_datetime(f"{self.__start_date} {START_TIMES[period]}")
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
                date = pd.to_datetime(f"{date} {START_TIMES[period]}")
                dates = dates[~(dates == date)]

        self.__areas.append(
            dict(
                name=name,
                duration=duration,
                period=period,
                dates=dates,
                n_employees=n_employees,
            )
        )

    # def add_employee(self, employee: Employee) -> None:
    def add_employee(self, name: str, working_areas: list) -> None:
        areas = [area["name"] for area in self.__areas]
        # assert all(area in areas for area in employee.areas)
        assert all(area in areas for area in working_areas)
        # employee.id = len(self.__employees)
        employee = dict(name=[name], hours_worked=[0])
        employee.update({area: [area in working_areas] for area in areas})

        self.__employees = pd.concat(
            [pd.DataFrame(employee), self.__employees]
        ).reset_index(drop=True)

    def create_shifts(self, dates: DatetimeIndex) -> list:
        shifts = []
        for date in dates:
            for area in self.__areas:
                i = area["dates"].date == date.date()
                if any(i):
                    for n in range(area["n_employees"]):
                        shifts.append(
                            dict(
                                name=f"{date.date()}_{area['name']}_{n}",
                                area=area["name"],
                                date=area["dates"][i][0],
                                duration=area["duration"],
                                period=area["period"],
                            )
                        )

        #  Sort shifts chronollogically
        shifts.sort(key=lambda x: (x["date"].day, x["date"].hour))

        shifts = pd.DataFrame(shifts)
        return shifts

    def create_schedule(self) -> None:
        # Create dates
        start = pd.to_datetime(f"{self.__start_date}")
        end = start + pd.offsets.MonthEnd()
        all_dates = pd.date_range(start, end)
        weeks_i = (all_dates.shift(-1).weekday // 6).values.cumsum()
        weeks = all_dates.groupby(weeks_i)

        # Loop through weeks
        for i, dates in weeks.items():
            print(f"Week {i}")
            # Create shifts
            shifts = self.create_shifts(dates)
            if shifts.empty:
                continue

            # Find a solution
            best = self.perform_GA(shifts)

            # Assign people from solution to shifts
            shifts["employee"] = self.employees.iloc[best["solution"]]["name"].values
            shifts["employee_id"] = best["solution"]
            self.__shifts = pd.concat([self.__shifts, shifts])

            # Add hours worked to employees
            for i in range(self.n_employees):
                name = self.employees.iloc[i]["name"]
                hours_worked = shifts[shifts["employee"] == name]["duration"].sum()
                self.__employees.loc[i, ["hours_worked"]] += hours_worked

    def perform_GA(self, shifts: pd.DataFrame):
        # Initialize solutions and scores
        solutions = self.create_random_solutions(shifts)
        scores = self.calculate_scores(solutions, shifts)

        max_scores = []
        mean_scores = []
        best = {"score": 0, "solution": None}
        max_mating = 1000
        counter = 0
        early_stop_n = 15
        previous_mean = 0
        # Loop through n iterations
        for n in range(self.__GA_params["n_iterations"]):
            best_solutions_i = self.pick_best_solutions(solutions, scores)
            best_solutions = solutions[:, best_solutions_i]

            for k in range(max_mating):
                # Crossover
                new_solutions = self.crossover(best_solutions)

                # Mutation
                new_solutions = self.mutate(new_solutions, shifts)
                new_scores = self.calculate_scores(new_solutions, shifts)
                if (new_scores > 0).all():
                    break
            if k == max_mating - 1:
                raise RuntimeError("Could not create valid new solutions")

            # Replace worst solutions
            worst_i = np.argsort(scores)[:2]
            solutions[:, worst_i] = new_solutions

            # Calculate score
            scores = self.calculate_scores(solutions, shifts)

            # Early stop
            if abs(scores.mean() - previous_mean) < 1e-3:
                counter += 1
            else:
                counter = 0

            if counter > early_stop_n:
                break

            previous_mean = scores.mean()

            # Logging statistics
            if scores.mean() > 1:
                mean_scores.append(scores.mean())
                max_scores.append(scores.max())

            if scores.max() > best["score"]:
                i = scores.argmax()
                best["score"] = scores[i]
                best["solution"] = solutions[:, i]

        # print(best)
        # plt.plot(max_scores)
        # plt.plot(mean_scores)
        # plt.show(block=True)
        return best

    def pick_best_solutions(self, solutions: np.array, scores: np.array):
        assert (scores > 0).sum() >= 2
        prob = scores / scores.sum()
        best_solutions_i = np.random.choice(prob.size, 2, p=prob, replace=False)
        return best_solutions_i

    def mutate(self, solutions: np.array, shifts: pd.DataFrame) -> np.array:
        mutation_rate = self.__GA_params["mutation_rate"]
        mask = np.random.choice(
            [True, False], solutions.shape, p=[mutation_rate, 1 - mutation_rate]
        )

        # TODO Make sure new mutated values are valid
        random_values = np.random.randint(0, self.n_employees, solutions.shape)
        new_solutions = solutions * (1 - mask) + random_values * mask
        return new_solutions

    def crossover(self, solutions: np.array) -> np.array:
        solution1, solution2 = solutions.T
        ci = np.random.randint(0, solution1.size)
        offspring1 = np.concatenate([solution1[:ci], solution2[ci:]])
        offspring2 = np.concatenate([solution2[:ci], solution1[ci:]])
        return np.stack((offspring1, offspring2)).T

    def create_random_solutions(self, shifts: pd.DataFrame) -> np.array:
        solutions = -np.ones((shifts.shape[0], self.__GA_params["n_solutions"]))
        eligible_employees = [
            self.employees[self.employees[area]].index.values for area in shifts["area"]
        ]
        # Loop through shifts grouped by periods
        shifts_groups = shifts.groupby("date")

        # Find a valid combination/solution for shifts in this time
        for j in range(self.__GA_params["n_solutions"]):
            c = 0
            while (solutions[:, j] < 0).any():
                c += 1
                if c > self.__GA_params["n_solutions"] * 1000:
                    raise RuntimeError(
                        "Could not create initial solution. Iterations exceeded"
                    )
                for _, shifts_per_date in shifts_groups:
                    k = slice(shifts_per_date.index[0], shifts_per_date.index[-1] + 1)

                    for i, employees in zip(
                        shifts_per_date.index, eligible_employees[k]
                    ):
                        employees_ = [
                            employee
                            for employee in employees
                            if employee not in solutions[k, j]
                        ]

                        # If there are no eligible employees try again from scratch
                        if not employees_:
                            break
                        solutions[i, j] = np.random.choice(employees_)

        return np.array(solutions, dtype=np.int32)

    def calculate_cost_matrices(
        self, shifts: pd.DataFrame, solutions: np.array
    ) -> np.array:
        cost_matrix = np.zeros((shifts.shape[0], self.n_employees))

        # Add constraints to cost matrix
        for j, employee in self.employees.iterrows():
            # Add hours worked from previous shifts
            if not self.shifts.empty:
                df = self.shifts[self.shifts["employee"] == employee["name"]]
                if not df.empty:
                    cost_matrix[:, j] += df["duration"].sum()

            for i, shift in shifts.iterrows():
                # Areas outside of employee scope
                cost_matrix[i, j] += (not employee[shift["area"]]) * LARGE_NUMBER

                if not self.shifts.empty:
                    # Employee worked yesterday evening
                    previous_shifts_dates = self.shifts["date"].dt.date
                    yesterday = (shift["date"] - pd.offsets.Day(1)).date()
                    df = self.shifts[
                        (self.shifts["employee"] == employee["name"])
                        & (self.shifts["period"] == EVENING)
                        & (previous_shifts_dates == yesterday)
                    ]
                    if not df.empty:
                        cost_matrix[i, j] = LARGE_NUMBER

                # TODO add soft constraints here

        # Create a cost matrix for each solution
        n_solutions = solutions.shape[1]
        cost_matrices = np.array([cost_matrix for _ in range(n_solutions)])

        # Update cost matrices according to solutions
        for i in shifts.index:
            shift = shifts.iloc[i]
            shift_solutions = solutions[i]

            # TODO make constraint more readable

            # If people work in the evening they don't work the next afternoon and evening
            if shift["period"] == EVENING:
                # Find shifts in the afternoon the next date
                next_day = (shift["date"] + pd.offsets.Day()).date()

                next_day_shifts = shifts[shifts["date"].dt.date == next_day]

                for j in next_day_shifts.index:
                    cost_matrices[range(n_solutions), j, shift_solutions] = LARGE_NUMBER

            # Work only one shift on a given period
            same_time_shifts_i = shifts[shifts["date"] == shift["date"]].index
            same_time_shifts_i = same_time_shifts_i[same_time_shifts_i > i]
            for j in same_time_shifts_i:
                cost_matrices[range(n_solutions), j, shift_solutions] = LARGE_NUMBER

            # Update cost matrices after people worked in a shift
            remaining_shifts = shifts[shifts["date"] > shift["date"]]
            for j in remaining_shifts.index:
                cost_matrices[range(n_solutions), j, shift_solutions] += shift[
                    "duration"
                ]

        # Working more than two night shifts is not allowed
        evening_shifts = shifts[shifts["period"] == EVENING]
        for employee in range(self.n_employees):
            n_evening_shifts = (solutions[evening_shifts.index, :] == employee).sum(
                axis=0
            )
            mask = n_evening_shifts > 2
            for j in evening_shifts.index[2:]:
                cost_matrices[mask, j, employee] = LARGE_NUMBER

        np.histogram(solutions[evening_shifts.index, 0])
        evening_shifts
        solutions

        return cost_matrices

    def calculate_scores(self, solutions: np.array, shifts: pd.DataFrame) -> np.array:
        cost_matrices = self.calculate_cost_matrices(shifts, solutions)
        scores = np.array(
            [
                cost_matrices[range(solution.size), j, solution]
                for j, solution in zip(shifts.index, solutions)
            ]
        )
        scores = scores.sum(axis=0)
        scores = np.clip(scores, 0, LARGE_NUMBER)
        return -(scores - LARGE_NUMBER)


if __name__ == "__main__":
    GA_params = dict(n_solutions=50, n_iterations=100, mutation_rate=0.05)
    manager = Manager(2023, 7, GA_params)
    manager.add_area("Plantao_Noturno", 12, "E", 2, DAYS_OF_WEEK)

    # manager.add_area("Permanencia", 4, "E", 1, ["SAT", "SUN"])

    employees = [
        ["Camila Neves", ["Plantao_Noturno"]],
        ["Henrique Costa", ["Plantao_Noturno"]],
        ["Maria Guerra", ["Plantao_Noturno"]],
        ["Jessika Cristina", ["Plantao_Noturno"]],
        ["Isabelle Amorim", ["Plantao_Noturno"]],
        ["Rafaela Ferraz", ["Plantao_Noturno"]],
        ["Giovanna Macedo", ["Plantao_Noturno"]],
        ["Ana Luiza", ["Plantao_Noturno"]],
        ["Joyce Fragoso", ["Plantao_Noturno"]],
        ["Iara", ["Plantao_Noturno"]],
        ["Lilian", ["Plantao_Noturno"]],
        ["Eduarda Vital", ["Plantao_Noturno"]],
        ["Vitoria", ["Plantao_Noturno"]],
        ["Marcela", ["Plantao_Noturno"]],
        ["Leticia", ["Plantao_Noturno"]],
    ]
    for employee in employees:
        manager.add_employee(*employee)
    manager.create_schedule()
    manager.shifts
    manager.employees
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(manager.areas)
