import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math


def fraction_to_hours_minutes(fraction: float) -> str:
    hours = int(math.floor(fraction))
    minutes = round((fraction - hours) * 60)
    # if hours > 12:
    #     hours -= 12
    return f"{hours}:{minutes:02d}"


def undo_dst(df: pd.DataFrame) -> pd.DataFrame:
    """Take a dataframe loaded with prayer times and reverses
    the 1 hour dst jump in athan times"""

    last_fajr_value = None
    day_light_offset = 0

    for i in df.index:
        if last_fajr_value == None:
            last_fajr_value = df.loc[i, "fajr"]
            continue

        curr_fajr_value = df.loc[i, "fajr"]

        if last_fajr_value - curr_fajr_value > 0.5:
            day_light_offset += 1

        if last_fajr_value - curr_fajr_value < -0.5:
            day_light_offset -= 1

        last_fajr_value = curr_fajr_value

        df.loc[i, "fajr"] += day_light_offset
        df.loc[i, "sunrise"] += day_light_offset
        df.loc[i, "dhuhr"] += day_light_offset
        df.loc[i, "asr"] += day_light_offset
        df.loc[i, "maghrib"] += day_light_offset
        df.loc[i, "isha"] += day_light_offset

    return df


def find_next_time_step(time: int, incr_hr=0.5, reverse=False) -> int:
    """Given a time, returns the next (or previous) time that
    is a multiple of incr_hr"""

    options = [x * incr_hr for x in range(int(24 / incr_hr))]

    if reverse:
        for step in options[::-1]:
            if step < time:
                return step
    else:
        for step in options:
            if step > time:
                return step


def find_next_n_transitions(df: pd.DataFrame, pos, n) -> list[int]:
    """Given a series and index position, walk back and find the
    index of the last n value transitions"""

    transitions = []
    last_val = None

    while n and pos <= len(df.index) - 1:
        if last_val == None:
            last_val = df.loc[pos]
            continue

        if df.loc[pos] != last_val:
            transitions.append(pos)
            n -= 1

        last_val = df.loc[pos]
        pos += 1

    if not transitions:
        transitions = [df.index[-1]]

    return transitions


def find_last_n_transitions(df: pd.DataFrame, pos, n) -> list[int]:
    """Given a series and index position, walk back and find the
    index of the last n value transitions"""

    transitions = []
    last_val = None

    while n and pos >= 0:
        if last_val == None:
            last_val = df.loc[pos]
            continue

        if df.loc[pos] != last_val:
            transitions.append(pos)
            n -= 1

        last_val = df.loc[pos]
        pos -= 1

    return transitions


def rough_iqama_time_estimation(
    df: pd.DataFrame,
    iqama_name,
    lower_bound_name,
    lower_bound_offset,
    upper_bound_name,
    upper_bound_offset,
    iqama_step=0.5,
):
    iqama = 0
    for i in df.index:
        if iqama < df.loc[i, lower_bound_name] + lower_bound_offset:
            iqama = find_next_time_step(
                df.loc[i, lower_bound_name] + lower_bound_offset, incr_hr=iqama_step
            )
        elif iqama > df.loc[i, upper_bound_name] + upper_bound_offset:
            iqama = find_next_time_step(
                df.loc[i, upper_bound_name] + upper_bound_offset,
                reverse=True,
                incr_hr=iqama_step,
            )
        df.loc[i, iqama_name] = iqama

    return df


def smooth_iqama_transitions(
    df: pd.DataFrame,
    iqama_name,
    lower_bound_name,
    lower_bound_offset,
    upper_bound_name,
    upper_bound_offset,
    walkback=True,
):
    last_val = None
    running_cnt = 0
    for i in df.index:
        if last_val == None:
            # print("1")
            last_val = df.loc[i, iqama_name]
            running_cnt = 1
            continue
        if last_val == df.loc[i, iqama_name]:
            # print("2")
            running_cnt += 1
            continue
        else:
            # print("3")
            if running_cnt > 21:
                # print("3.1")
                last_val = df.loc[i, iqama_name]
                running_cnt = 1
                continue
            else:
                # print("3.2")
                if i - running_cnt - 1 > 0:
                    # print("3.2.1")
                    before_last_val = df.loc[i - running_cnt - 1, iqama_name]
                    # print(i)
                    # print(f"{before_last_val=}")
                    # print(f"{last_val=}")
                    # print(f"{df.loc[i, iqama_name]=}")
                    # print(f"{df.loc[i-running_cnt:i-1, 'fajr']=}")
                    # print(f"{df.loc[i-running_cnt:i-1, 'sunrise'] -30/60=}")
                    if (
                        before_last_val
                        > df.loc[i - running_cnt : i - 1, lower_bound_name]
                        + lower_bound_offset
                    ).all() and (
                        before_last_val
                        < df.loc[i - running_cnt : i - 1, upper_bound_name]
                        + upper_bound_offset
                    ).all():
                        # print("3.2.1.1")
                        df.loc[i - running_cnt : i - 1, iqama_name] = before_last_val
                        last_val = df.loc[i, iqama_name]
                        running_cnt = 1
                        continue

                next_val = df.loc[i, iqama_name]
                if (
                    next_val
                    > df.loc[i - running_cnt : i - 1, lower_bound_name]
                    + lower_bound_offset
                ).all() and (
                    next_val
                    < df.loc[i - running_cnt : i - 1, upper_bound_name]
                    + upper_bound_offset
                ).all():
                    # print("3.2.2")
                    df.loc[i - running_cnt : i - 1, iqama_name] = next_val
                    last_val = df.loc[i, iqama_name]
                    running_cnt += 1
                    continue

                if walkback:
                    j = i - 1
                    new_iqama_value = df.loc[j, iqama_name] + 15 / 60
                    last_transitions = find_last_n_transitions(
                        df.loc[:, iqama_name], j, 2
                    )
                    if len(last_transitions) != 2:
                        last_transitions = (0, 0)
                    half_way = j - (j - last_transitions[1]) // 2
                    walkback_success = False
                    while j >= half_way:
                        if new_iqama_value > df.loc[
                            j, lower_bound_name
                        ] + lower_bound_offset and (
                            new_iqama_value
                            < df.loc[j, upper_bound_name] + upper_bound_offset
                        ):
                            walkback_success = True
                            df.loc[j, iqama_name] = new_iqama_value
                            j -= 1
                        else:
                            break

                last_val = df.loc[i, iqama_name]
                running_cnt = 1

                if walkback and not walkback_success:
                    print("no conditions met:", i)

    return df


if __name__ == "__main__":
    # load timetable generated by gen_timetable.py
    df = pd.read_parquet("timetable.parquet")

    # for testing, crop the dataset
    # df = df.iloc[:365*4]

    original_index = df.index
    df = df.reset_index()

    # set iqama times for dhuhr and maghrib
    df.loc[:, "dhuhr_iqama"] = 13.5
    df.loc[:, "maghrib_iqama"] = df.loc[:, "maghrib"] + 10 / 60

    df = rough_iqama_time_estimation(
        df, "fajr_iqama", "fajr", 2 / 60, "sunrise", -40 / 60
    )
    # df = rough_iqama_time_estimation(df, "isha_iqama2", "isha", 5/60, "isha", 45/60)

    df = smooth_iqama_transitions(df, "fajr_iqama", "fajr", 0, "sunrise", -30 / 60)

    fajr_transitions = (
        [-1]
        + find_last_n_transitions(df.loc[:, "fajr_iqama"], df.index[-1], len(df.index))[
            ::-1
        ]
        + [df.index[-1]]
    )
    print(fajr_transitions)

    iqama = 0
    iqama_name = "asr_iqama"
    lower_bound_name = "asr"
    lower_bound_offset = 0 / 60
    upper_bound_name = "maghrib"
    upper_bound_offset = -90 / 60
    iqama_step = 0.5
    # df.loc[:, "asr_limit"] = df.loc[:, "maghrib"] + upper_bound_offset
    for i, transition in enumerate(fajr_transitions):
        print(1)
        if i == 0:
            continue

        print(2)
        max_lower_bound = (
            df.loc[
                fajr_transitions[i - 1] + 1 : fajr_transitions[i], lower_bound_name
            ].max()
            + lower_bound_offset
        )
        min_upper_bound = (
            df.loc[
                fajr_transitions[i - 1] + 1 : fajr_transitions[i], upper_bound_name
            ].min()
            + upper_bound_offset
        )
        print(f"{iqama=}")
        print(f"{max_lower_bound=}")
        print(f"{min_upper_bound=}")
        if iqama < max_lower_bound:
            print("A")
            iqama = find_next_time_step(max_lower_bound, incr_hr=iqama_step)
        elif iqama > min_upper_bound:
            print("B")
            iqama = find_next_time_step(
                min_upper_bound, reverse=True, incr_hr=iqama_step
            )

        print(f"{fajr_transitions[i-1]+1=}")
        print(f"{fajr_transitions[i]=}")
        print(iqama)

        df.loc[fajr_transitions[i - 1] + 1 : fajr_transitions[i], iqama_name] = iqama

    df.loc[df.index[-1], iqama_name] = iqama

    iqama = 0
    iqama_name = "isha_iqama"
    lower_bound_name = "isha"
    lower_bound_offset = 0 / 60
    upper_bound_name = "isha"
    upper_bound_offset = 1
    iqama_step = 0.5
    # df.loc[:, "isha_limit"] = df.loc[:, "isha"] + upper_bound_offset
    for i, transition in enumerate(fajr_transitions):
        print(1)
        if i == 0:
            continue

        print(2)
        max_lower_bound = (
            df.loc[
                fajr_transitions[i - 1] + 1 : fajr_transitions[i], lower_bound_name
            ].max()
            + lower_bound_offset
        )
        min_upper_bound = (
            df.loc[
                fajr_transitions[i - 1] + 1 : fajr_transitions[i], upper_bound_name
            ].min()
            + upper_bound_offset
        )
        print(f"{iqama=}")
        print(f"{max_lower_bound=}")
        print(f"{min_upper_bound=}")
        if iqama < max_lower_bound:
            print("A")
            iqama = find_next_time_step(max_lower_bound, incr_hr=iqama_step)
        elif iqama > min_upper_bound:
            print("B")
            iqama = find_next_time_step(
                min_upper_bound, reverse=True, incr_hr=iqama_step
            )

        print(f"{fajr_transitions[i-1]+1=}")
        print(f"{fajr_transitions[i]=}")
        print(iqama)

        df.loc[fajr_transitions[i - 1] + 1 : fajr_transitions[i], iqama_name] = iqama

    df.loc[df.index[-1], iqama_name] = iqama

    # minor correction to isha iqama step size
    for i in df.index:
        if df.loc[i, "isha_iqama"] < df.loc[i, "isha"]:
            next_transition = find_next_n_transitions(df.loc[:, "isha_iqama"], i, 1)[0]
            half_way = (next_transition - i) // 2 + i
            df.loc[i:half_way, "isha_iqama"] = find_next_time_step(
                df.loc[i:half_way, "isha"].max()
            )

    min_asr_time = 16  # 4 PM
    for i in df.index:
        if df.loc[i, "asr_iqama"] < min_asr_time:
            df.loc[i, "asr_iqama"] = min_asr_time

    min_isha_time = 19  # 7 PM
    for i in df.index:
        if df.loc[i, "isha_iqama"] < min_isha_time:
            df.loc[i, "isha_iqama"] = min_isha_time

    max_isha_time = find_next_time_step(df.loc[:, "isha"].max(), incr_hr=15 / 60)
    for i in df.index:
        if df.loc[i, "isha_iqama"] > max_isha_time:
            df.loc[i, "isha_iqama"] = max_isha_time

    df = smooth_iqama_transitions(
        df, "asr_iqama", "asr", 0, "maghrib", -30 / 60, walkback=False
    )
    df = smooth_iqama_transitions(
        df, "isha_iqama", "isha", 10 / 60, "isha", 1, walkback=False
    )

    ax = df.iloc[365 : 365 * 4].plot(figsize=(20, 10))
    ax.set_yticks(np.arange(0, 24, 1))
    plt.grid(True)
    plt.savefig("timetable.png")
    plt.show()

    with open("timetable.json", "w+") as f:
        timetable = {}
        for i in df.index:
            timetable[original_index[i]] = {
                "fajr": [
                    fraction_to_hours_minutes(df.loc[i, "fajr"]),
                    fraction_to_hours_minutes(df.loc[i, "fajr_iqama"]),
                ],
                "sunrise": [fraction_to_hours_minutes(df.loc[i, "sunrise"]), ""],
                "dhuhr": [
                    fraction_to_hours_minutes(df.loc[i, "dhuhr"]),
                    fraction_to_hours_minutes(df.loc[i, "dhuhr_iqama"]),
                ],
                "asr": [
                    fraction_to_hours_minutes(df.loc[i, "asr"]),
                    fraction_to_hours_minutes(df.loc[i, "asr_iqama"]),
                ],
                "maghrib": [fraction_to_hours_minutes(df.loc[i, "maghrib"]), "+10 min"],
                "isha": [
                    fraction_to_hours_minutes(df.loc[i, "isha"]),
                    fraction_to_hours_minutes(df.loc[i, "isha_iqama"]),
                ],
            }
        f.write(json.dumps(timetable, indent=4))
