"""
Functional and direct approach to create "ideal" trajectories
based on a few of their points manually created.
"""
from typing import List

from collections import namedtuple

import numpy as np
import pandas as pd

from geoutils import haversine


# A trajectory_skeleton is an ordered sequence of trajectory segments. A trajectory segment is
# a tuple of the form:
# (start point (lon-lat), end point (lon-lat), time between points in sec., number of pings per seconds, annotation)
# Note: start point and end point can be identical, if the device is steady for a segment.
TrajectorySegment = namedtuple(
    "TrajectorySegment",
    ["start_point", "end_point", "time_between", "num_pings_per_seconds", "annotation"],
)


def create_ideal_L_shape_trajectory__turin(entity_id: str = "72cac3d5") -> pd.DataFrame:
    """
    Create the L shape trajectory of a subject dwelling in 4 shops in
    two perpendicular roads. Time spent in each shop is
    - 5 min  - ice cream shop - buy ice cream
    - 10 min  - chocolate shop 1 -  buy chocolate
    - 20 min  - chocolate shop 2 -  buy chocolate
    - 40 min  - Liuthier - order a violin

    The time precision is seconds.
    """
    # 15 points (lon, lat)
    pt_a = [7.683241, 45.068314]
    pt_b = [7.683746, 45.068132]  # = d
    pt_c = [7.683692, 45.068054]  # ice cream
    pt_d = [7.683746, 45.068132]  # = b
    pt_e = [7.684296, 45.067922]  # = g
    pt_f = [7.684260, 45.067855]  # chocolate 1
    pt_g = [7.684296, 45.067922]  # = e
    pt_h = [7.685509, 45.067501]
    pt_i = [7.685364, 45.067285]  # = k
    pt_j = [7.685272, 45.067317]  # chocolate 2
    pt_k = [7.685364, 45.067285]  # = i
    pt_l = [7.685755, 45.067848]  # = n
    pt_m = [7.685823, 45.067827]  # Luthier shop
    pt_n = [7.685755, 45.067848]  # = l
    pt_o = [7.686062, 45.068290]

    avg_speed = 1  # m/s
    pings_per_sec = 0.2
    num_seconds_per_stop = (5 * 60, 10 * 60, 15 * 60, 20 * 60)
    trajectory_skeleton = [
        TrajectorySegment(
            pt_a,
            pt_b,
            haversine(pt_a, pt_b) / avg_speed,
            pings_per_sec,
            "moving to ice cream shop",
        ),
        TrajectorySegment(
            pt_b,
            pt_c,
            haversine(pt_b, pt_c) / avg_speed,
            pings_per_sec,
            "entering ice cream shop",
        ),
        TrajectorySegment(
            pt_c, pt_c, num_seconds_per_stop[0], pings_per_sec, "dwell ice cream shop"
        ),
        TrajectorySegment(
            pt_c,
            pt_d,
            haversine(pt_c, pt_d) / avg_speed,
            pings_per_sec,
            "exiting ice cream shop",
        ),
        TrajectorySegment(
            pt_d,
            pt_e,
            haversine(pt_d, pt_e) / avg_speed,
            pings_per_sec,
            "moving to chocolate shop 1",
        ),
        TrajectorySegment(
            pt_e,
            pt_f,
            haversine(pt_e, pt_f) / avg_speed,
            pings_per_sec,
            "entering chocolate shop 1",
        ),
        TrajectorySegment(
            pt_f, pt_f, num_seconds_per_stop[1], pings_per_sec, "dwell chocolate shop 1"
        ),
        TrajectorySegment(
            pt_f,
            pt_g,
            haversine(pt_f, pt_g) / avg_speed,
            pings_per_sec,
            "exiting chocolate shop 1",
        ),
        TrajectorySegment(
            pt_g,
            pt_h,
            haversine(pt_g, pt_h) / avg_speed,
            pings_per_sec,
            "moving to chocolate shop 2",
        ),
        TrajectorySegment(
            pt_h,
            pt_i,
            haversine(pt_h, pt_i) / avg_speed,
            pings_per_sec,
            "moving to chocolate shop 2",
        ),
        TrajectorySegment(
            pt_i,
            pt_j,
            haversine(pt_i, pt_j) / avg_speed,
            pings_per_sec,
            "entering chocolate shop 2",
        ),
        TrajectorySegment(
            pt_j, pt_j, num_seconds_per_stop[2], pings_per_sec, "dwell chocolate shop 2"
        ),
        TrajectorySegment(
            pt_j,
            pt_k,
            haversine(pt_j, pt_k) / avg_speed,
            pings_per_sec,
            "exiting chocolate shop 2",
        ),
        TrajectorySegment(
            pt_k,
            pt_l,
            haversine(pt_k, pt_l) / avg_speed,
            pings_per_sec,
            "moving to luthier",
        ),
        TrajectorySegment(
            pt_l,
            pt_m,
            haversine(pt_l, pt_m) / avg_speed,
            pings_per_sec,
            "entering luthier",
        ),
        TrajectorySegment(
            pt_m, pt_m, num_seconds_per_stop[3], pings_per_sec, "dwell luthier"
        ),
        TrajectorySegment(
            pt_m,
            pt_n,
            haversine(pt_m, pt_n) / avg_speed,
            pings_per_sec,
            "exiting luthier",
        ),
        TrajectorySegment(
            pt_n,
            pt_o,
            haversine(pt_n, pt_o) / avg_speed,
            pings_per_sec,
            "moving to piazza",
        ),
    ]

    # trajectory to pings.
    start_time = pd.Timestamp(year=2023, month=1, day=1, hour=12)
    # pings/sec

    return trajectory_skeleton_to_trajectory(trajectory_skeleton, start_time, entity_id)


def create_ideal_L_shape_trajectory__sydney(
    entity_id: str = "72cac3d5",
) -> pd.DataFrame:
    """
    A segment in the life of a vintage guitars and coins collector.
    The signal appears while the collector is on the move, walking on Trafalgar street, Sydney.
    The first stop is for a coffee at Trafalgar st Espresso, for 8 minutes.
    Then continues down to the corner with Great western Highway, direction west towards the
    vintage guitar shop. Here browse for 25 minutes until he purchases a collectible guitar.
    The shop has no case to provide, so he continues to the music shop nearby where he stops for
    10 minutes to buy a guitar. After this he continues to the vintage coin shops, to sell
    some ancient doubloon. Here he stops to greet the owner and negotiate the price for
    a total of 34 minutes,
    before continuing down the Great Western hyw.

    The time precision is seconds.
    """
    # Points (lon, lat)
    pt_a = [151.170552, -33.885517]
    pt_b = [151.169913, -33.886866]
    pt_c = [151.169717, -33.886844]  # cafe Espresso
    pt_d = pt_b
    pt_e = [151.169475, -33.887764]  # corner with great western hyw
    pt_f = [151.167558, -33.887918]
    pt_g = [151.167490, -33.887654]  # vintage guitar shop
    pt_h = pt_f
    pt_i = [151.166604, -33.887996]
    pt_j = [151.1665832, -33.88776]  # modern guitar shop
    pt_k = pt_i
    pt_l = [151.165055, -33.888123]
    pt_m = [151.1650690, -33.887920]  # coins shop
    pt_n = pt_l
    pt_o = [151.163432, -33.888252]

    avg_speed = 1.0  # m/s
    pings_per_sec = 0.15
    num_seconds_per_stop = (8 * 60, 25 * 60, 10 * 60, 34 * 60)
    trajectory_skeleton = [
        TrajectorySegment(
            pt_a,
            pt_b,
            haversine(pt_a, pt_b) / avg_speed,
            pings_per_sec,
            "moving towards cafe",
        ),
        TrajectorySegment(
            pt_b,
            pt_c,
            haversine(pt_b, pt_c) / avg_speed,
            pings_per_sec,
            "entering cafe",
        ),
        TrajectorySegment(
            pt_c, pt_c, num_seconds_per_stop[0], pings_per_sec, "dwell at the cafe"
        ),
        TrajectorySegment(
            pt_c, pt_d, haversine(pt_c, pt_d) / avg_speed, pings_per_sec, "exiting cafe"
        ),
        TrajectorySegment(
            pt_d,
            pt_e,
            haversine(pt_d, pt_e) / avg_speed,
            pings_per_sec,
            "moving towards shop 1",
        ),
        TrajectorySegment(
            pt_e,
            pt_f,
            haversine(pt_e, pt_f) / avg_speed,
            pings_per_sec,
            "moving towards shop 1",
        ),
        TrajectorySegment(
            pt_f,
            pt_g,
            haversine(pt_f, pt_g) / avg_speed,
            pings_per_sec,
            "entering shop 1",
        ),
        TrajectorySegment(
            pt_g, pt_g, num_seconds_per_stop[1], pings_per_sec, "dwell at shop 1"
        ),
        TrajectorySegment(
            pt_g,
            pt_h,
            haversine(pt_g, pt_h) / avg_speed,
            pings_per_sec,
            "exiting shop 1",
        ),
        TrajectorySegment(
            pt_h,
            pt_i,
            haversine(pt_h, pt_i) / avg_speed,
            pings_per_sec,
            "moving towards shop 2",
        ),
        TrajectorySegment(
            pt_i,
            pt_j,
            haversine(pt_i, pt_j) / avg_speed,
            pings_per_sec,
            "entering shop 2",
        ),
        TrajectorySegment(
            pt_j, pt_j, num_seconds_per_stop[2], pings_per_sec, "dwell shop 2"
        ),
        TrajectorySegment(
            pt_j,
            pt_k,
            haversine(pt_j, pt_k) / avg_speed,
            pings_per_sec,
            "exiting shop 2",
        ),
        TrajectorySegment(
            pt_k,
            pt_l,
            haversine(pt_k, pt_l) / avg_speed,
            pings_per_sec,
            "moving to shop 3",
        ),
        TrajectorySegment(
            pt_l,
            pt_m,
            haversine(pt_l, pt_m) / avg_speed,
            pings_per_sec,
            "entering shop 3",
        ),
        TrajectorySegment(
            pt_m, pt_m, num_seconds_per_stop[3], pings_per_sec, "dwell shop 3"
        ),
        TrajectorySegment(
            pt_m,
            pt_n,
            haversine(pt_m, pt_n) / avg_speed,
            pings_per_sec,
            "exiting shop 3",
        ),
        TrajectorySegment(
            pt_n, pt_o, haversine(pt_n, pt_o) / avg_speed, pings_per_sec, "moving west"
        ),
    ]

    start_time = pd.Timestamp(year=2023, month=1, day=1, hour=12)

    return trajectory_skeleton_to_trajectory(trajectory_skeleton, start_time, entity_id)


def create_ideal_straight_trajectory_with_equidistant_visits__turin(
    entity_id: str = "72cac3d5",
) -> pd.DataFrame:
    """
    Straight trajectory between start and end points, with equidistant stops.

    The time precision is seconds.
    """
    # 2 points (lon, lat)
    pt_start = [7.686895, 45.070310]  # start point
    pt_end = [7.698833, 45.062572]  # end point

    # Number of equidistant stops (including start and end points)
    num_stops = 5
    num_seconds_per_stop = (0, 60 * 5, 60 * 10, 60 * 15, 0)
    avg_speed = 1  # m/s
    start_time = pd.Timestamp(year=2023, month=1, day=1, hour=12)
    pings_per_sec = 0.1  # pings/sec

    list_points = list(np.linspace(pt_start, pt_end, num_stops))

    # A trajectory_skeleton is a sequence of trajectory segments. A trajectory segment is
    # a tuple of the form:
    # (start point lon-lat, end point lon-lat, time between points in sec., annotation)
    # Note: start point and end point can be identical, if the device is steady for a segment.
    trajectory_skeleton = [
        TrajectorySegment(
            pt_start, pt_start, num_seconds_per_stop[0], pings_per_sec, "point 0"
        )
    ]

    for idx, (next_pt, dwell_time) in enumerate(
        zip(list_points[1:], num_seconds_per_stop[1:])
    ):
        pt = trajectory_skeleton[-1][1]
        pt_to_next = TrajectorySegment(
            pt,
            next_pt,
            haversine(pt, next_pt) / avg_speed,
            pings_per_sec,
            f"from dwell {idx} to dwell {idx + 1}",
        )
        next_pt_dwell = TrajectorySegment(
            next_pt, next_pt, dwell_time, pings_per_sec, f"dwell {idx + 1}"
        )
        trajectory_skeleton += [pt_to_next, next_pt_dwell]
    return trajectory_skeleton_to_trajectory(trajectory_skeleton, start_time, entity_id)


def trajectory_skeleton_to_trajectory(
    trajectory_skeleton: List[TrajectorySegment],
    start_time: pd.Timestamp,
    entity_id: str,
) -> pd.DataFrame:
    """
    From a sequence of TrajectorySegments to a trajectory of pings

    NOTE: segments must not cross the antimeridian.
    """
    df = pd.DataFrame(
        columns=["entity_id", "timestamp", "latitude", "longitude", "annotation"]
    )
    time_betw_segments_sec_cumulative = 0
    for ts_idx, ts in enumerate(trajectory_skeleton):
        num_pings = int(ts.num_pings_per_seconds * ts.time_between)
        lon_lat = np.linspace(ts.start_point, ts.end_point, num_pings)

        start_time_segment = start_time + pd.Timedelta(
            time_betw_segments_sec_cumulative, "sec"
        )
        time_betw_segments_sec_cumulative += ts.time_between
        end_time_segment = start_time + pd.Timedelta(
            time_betw_segments_sec_cumulative, "sec"
        )
        timestamps = pd.date_range(
            start_time_segment, end_time_segment, periods=num_pings, tz="Europe/Rome"
        )
        if ts_idx == len(trajectory_skeleton) - 1:
            # is the last segment: take all points.
            df_segment = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "longitude": lon_lat[:, 0],
                    "latitude": lon_lat[:, 1],
                }
            )
        else:
            # is not the last segment: skip the last point, as it is the first point of the next segment.
            df_segment = pd.DataFrame(
                {
                    "timestamp": timestamps[:-1],
                    "longitude": lon_lat[:-1, 0],
                    "latitude": lon_lat[:-1, 1],
                }
            )

        df_segment["timestamp"] = df_segment["timestamp"].apply(
            lambda x: x.round(freq="S")
        )
        df_segment["entity_id"] = entity_id
        df_segment["annotation"] = ts.annotation
        df = pd.concat([df, df_segment])

    return df.reset_index(drop=True)
