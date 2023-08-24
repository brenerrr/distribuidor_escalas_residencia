# %%
from random import shuffle
from collections import defaultdict
from matplotlib import pyplot as plt
from src.common_types import *
import pandas as pd
import numpy as np
from random import choices, seed
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
        self.__constraints = pd.DataFrame([])
        self.__shifts = pd.DataFrame(
            [], columns=["name", "area", "date", "duration", "period"]
        )
        self.__areas = []

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
                freq=f"W-{DAYS_PORT2ENG[day]}",
            )
            dates = dates.union(dates_)
        if exclude_dates is not None:
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
                should_be_balanced=should_be_balanced == 1,
                must_be_filled=must_be_filled == 1,
                n_employees=n_employees,
            )
        )
        if area not in self.__areas:
            self.__areas.append(area)

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

    def add_constraints(self, constraints: pd.DataFrame):
        self.__constraints = constraints

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

    def find_conflicts(self, shift, shifts):
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

    def create_schedule(self) -> None:
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

        employees = pd.merge(
            self.employees[["name"]], n_shifts, left_index=True, right_index=True
        )
        employees = pd.merge(employees, hours_worked, left_index=True, right_index=True)
        self.__employees = employees

        self.drop_extra_shifts()

        self.create_free_weekends()

    def create_free_weekends(self) -> None:
        # Create free weekends
        seed(1)
        shifts = self.shifts.copy()
        employees = self.employees
        employees["free_weekend"] = ""
        weekends_shifts = shifts.loc[shifts["is_weekend"]]
        weeks = shifts.loc[shifts["is_weekend"]]["week"].unique()
        # Loop through employees

        for i in employees.index[:-1]:
            weekends_shifts = shifts[
                (shifts["is_weekend"]) & (shifts["employee_id"] == i)
            ]
            if weekends_shifts.empty:
                continue

            sucessful_swaps = False
            counter = 0
            while not sucessful_swaps:
                counter += 1
                if counter > 50:
                    print(
                        f"Could not find a free weekend for {employees.iloc[int(i)]['name']}"
                    )
                    week = -1
                    break
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

                    candidates = shifts[
                        (shifts["area"] == shift["area"]) & (shifts["employee_id"] != i)
                    ]["employee_id"].unique()
                    candidates = candidates[
                        employees.loc[candidates]["free_weekend"] != week
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
                        # List all shifts of the same kind of this candidate that are on other weekends
                        shifts_swap = shifts[
                            (shifts["employee_id"] == j)
                            & (shifts["area"] == shift["area"])
                            & (shifts["duration"] == shift["duration"])
                            & ~(
                                shifts["is_weekend"] & (shifts["week"] == shift["week"])
                            )
                            & ~(
                                shifts["is_weekend"]
                                & (shifts["week"] == employees.loc[j, "free_weekend"])
                            )
                            & (shifts["date"] != shift["date"])
                        ]

                        shifts_i = shifts[(shifts["employee_id"] == i)]
                        shifts_j = shifts[(shifts["employee_id"] == j)]

                        for _, shift_ in shifts_swap.iterrows():
                            conflicts = self.find_conflicts(shift_, shifts_i)
                            conflicts.extend(self.find_conflicts(shift, shifts_j))
                            if conflicts:
                                continue
                            else:
                                swap.append(shift_)

                    if swap:
                        shift_to_swap = choices(swap)[0]
                        index_swap = shifts[
                            shifts["name"] == shift_to_swap["name"]
                        ].index[0]
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
                        counter = 0
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

        self.__employees = employees
        self.__shifts = shifts

        employees.sort_values("hours_worked")

    def find_solution(self):
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
                        print(f"Trying generating another initial solution {i} {j}")
                        print(shift["name"])
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
            print
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

        # If another area with shifts at the same time needs all of its employees,
        # do not assign them to this shift
        same_time_shifts = self.shifts[
            (self.shifts.date == shift.date) & (self.shifts["area"] != shift["area"])
        ]
        if not same_time_shifts.empty:
            iN = self.shifts[self.shifts["name"] == shift["name"]].index[0]
            i0 = same_time_shifts.index[0]
            i_areas = same_time_shifts["area"].str.split("_").str[0]
            n_necessary = same_time_shifts.loc[iN:].groupby(i_areas).size()
            areas = n_necessary.index
            # Count number of available employees per area
            i_exclude = solution[i0:iN]
            n_available = self.employees[~self.employees.index.isin(i_exclude)][
                areas
            ].sum()
            n_available = n_available[n_necessary.index]

            # Exclude employees that are already working on shifts at the same time
            full_areas = areas[n_available == n_necessary]
            i_unavailable = self.employees[
                self.employees[self._Manager__areas][full_areas].sum(axis=1) > 0
            ].index
            cost[i_unavailable] = LARGE_NUMBER

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
                        (next_shifts["period"].isin((AFTERNOON,)))
                        & next_shifts["must_be_filled"]
                    )
                    | (
                        (next_shifts["period"] == MORNING)
                        & (next_shifts["duration"] > 5)
                    )
                )
            ]

            today_shifts = previous_shifts[previous_shifts["date"] == shift["date"]]

            pool = self.employees[
                (self.employees[shift["area"]])
                & (self.employees[tomorrow_shifts["area"].unique()]).any(axis=1)
            ]
            pool = pool.drop(solution[today_shifts.index], errors="ignore")

            # Get areas where workers on pool can also be assigned to
            areas = tomorrow_shifts["area"].unique()
            pool_areas = pool[self._Manager__areas].any()
            pool_areas = pool_areas[pool_areas].index
            areas = [area for area in areas if area in pool_areas]

            # Get rid of shifts that do not have intersecting employees
            tomorrow_shifts = tomorrow_shifts[tomorrow_shifts["area"].isin(areas)]

            # Loop through shifts and drop employees from pool
            for _, tomorrow_shift in tomorrow_shifts.iterrows():
                area_pool = pool[pool[tomorrow_shift["area"]]]
                pool = pool.drop(area_pool.index[0])

            # Identify areas that have no spare employees
            areas_spare = [area for area in areas if pool[area].any()]
            areas_no_spare = [area for area in areas if area not in areas_spare]

            unavailable_i = self.employees[
                self.employees[areas_no_spare].any(axis=1)
            ].index

            cost[unavailable_i] = LARGE_NUMBER

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

    def export_results(self, inputs_areas, month) -> None:
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
        must_not_be_filled = inputs_areas[inputs_areas["Obrigatório"] != 1]["Área"]
        one_person_area = one_person_area[~one_person_area.isin(must_not_be_filled)]
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

        with pd.ExcelWriter(f"escala_{month}.xlsx") as writer:
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

    def drop_extra_shifts(self):
        constraints = self.__constraints
        self.__employees.loc[self.n_employees] = 0
        self.__employees.loc[self.n_employees - 1, "name"] = "R+"
        for _, constraint in constraints.iterrows():
            area = constraint["Área"]
            max_shifts = constraint["MaxTurnos"]
            employees = self.employees[self.employees[area] > max_shifts]
            n_extra = employees[area] - max_shifts
            for i, n in n_extra.items():
                shifts = self.shifts[
                    (self.shifts["employee_id"] == i) & (self.shifts["area"] == area)
                ]
                indexes = shifts.index.values.copy()
                shuffle(indexes)
                i_exclude = indexes[: int(n)]
                self.__shifts.loc[i_exclude, ["employee", "employee_id"]] = [
                    "R+",
                    self.n_employees - 1,
                ]

                self.__employees.loc[i, "hours_worked"] -= shifts.loc[i_exclude][
                    "duration"
                ].sum()
                self.__employees.loc[i, area] -= n
                self.__employees.loc[self.n_employees - 1, area] += n
                self.__employees.loc[
                    self.n_employees - 1, "hours_worked"
                ] += shifts.loc[i_exclude]["duration"].sum()
