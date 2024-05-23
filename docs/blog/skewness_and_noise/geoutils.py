# pylint: disable=missing-function-docstring
"""
Functional and direct approach to create "ideal" trajectories
and add noise to them based on a range of models.
"""
from typing import Tuple
import math

import numpy as np
import pandas as pd


AVG_EARTH_RADIUS_METERS = 6_371_008.8
AVG_EARTH_RADIUS_KM = 6_371.0088


def haversine(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    unit_of_measurement: str = "m",
) -> float:
    """Calculate the great-circle distance between two points on the Earth surface.

    Coordinates are passed (LONGITUDE, LATITUDE).

    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.

    Args:
        point1 (Tuple[float, float]): first point; tuple of (longitude, latitude) in decimal degrees
        point2 (Tuple[float, float]): second point; tuple of (longitude, latitude) in decimal degrees

    Returns:
        float: the distance between the two points in the requested unit, as a float in kilometers.

    Example:
        `haversine((45.7597, 4.8422), (48.8567, 2.3508))`
    """
    # unpack longitude/latitude
    lng1, lat1 = point1
    lng2, lat2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (
        math.sin(lat * 0.5) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(lng * 0.5) ** 2
    )

    avg_hr = {"Km": AVG_EARTH_RADIUS_KM, "m": AVG_EARTH_RADIUS_METERS}[
        unit_of_measurement
    ]

    return 2 * avg_hr * math.asin(math.sqrt(d))


def lat_long_linspace(start: Tuple[float, float], stop: Tuple[float, float], num: int):
    return [
        [a, b]
        for a, b in zip(
            list(np.linspace(start=start[0], stop=stop[0], num=num)),
            list(np.linspace(start=start[1], stop=stop[1], num=num)),
        )
    ]


def lat_deg_to_meters(lat_deg: pd.Series, precision: int = 6) -> pd.Series:
    return np.round(
        lat_deg.astype(float) * 2 * np.pi * AVG_EARTH_RADIUS_METERS / 360, precision
    ).astype(object)


def lat_meters_to_deg(lat_meters: pd.Series, precision=6) -> pd.Series:
    return np.round(
        lat_meters.astype(float) * 360 / (2 * np.pi * AVG_EARTH_RADIUS_METERS),
        precision,
    ).astype(object)


def lon_deg_to_meters(
    lon_deg: pd.Series, lat_deg: pd.Series, precision: int = 6
) -> pd.Series:
    return np.round(
        lon_deg.astype(float)
        * 2
        * np.pi
        * AVG_EARTH_RADIUS_METERS
        * np.cos(np.deg2rad(lat_deg.astype(float)))
        / 360,
        precision,
    ).astype(object)


def lon_meters_to_deg(
    lon_meters: pd.Series,
    lat: pd.Series,
    lat_in_meters: bool = False,
    precision: int = 6,
) -> pd.Series:
    """By default the latitude is passed in degrees. If passed in meters use lat_in_meters"""
    if lat_in_meters:
        lat = lat_meters_to_deg(lat)  # lat now in deg
    return np.round(
        lon_meters.astype(float)
        * 360
        / (2 * np.pi * AVG_EARTH_RADIUS_METERS * np.cos(np.deg2rad(lat.astype(float)))),
        precision,
    ).astype(object)


def lon_lat_meters_to_deg(
    lon_m: pd.Series, lat_m: pd.Series, precision: int = 6
) -> Tuple[pd.Series, pd.Series]:
    return (
        lon_meters_to_deg(lon_m, lat_m, lat_in_meters=True, precision=precision),
        lat_meters_to_deg(lat_m, precision),
    )


def lon_lat_deg_to_meters(
    lon_deg: pd.Series, lat_deg: pd.Series, precision: int = 6
) -> Tuple[pd.Series, pd.Series]:
    return (
        lon_deg_to_meters(lon_deg, lat_deg, precision=precision),
        lat_deg_to_meters(lat_deg, precision=precision),
    )


def meters_to_deg_average(meters: float, lat_deg: float) -> float:
    """
    What is x meters in degrees at given latitude and longitude (on average, regardless of directionality)?
    """
    lon_deg = (
        float(meters)
        * 360
        / (2 * np.pi * AVG_EARTH_RADIUS_METERS * np.cos(np.deg2rad(lat_deg)))
    )
    lat_deg = float(meters) * 360 / (2 * np.pi * AVG_EARTH_RADIUS_METERS)
    return (lat_deg + lon_deg) / 2
