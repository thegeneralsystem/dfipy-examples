# pylint: disable=missing-function-docstring
"""
Functional and direct approach to create "ideal" trajectories
and add noise to them based on a range of models.
"""
from copy import deepcopy
from typing import List, Optional, Tuple
import math

from collections import namedtuple

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point, MultiPoint

import geoutils


# A Sink is a triplet with longitude, latitude and a frequency (float betw 0 and 1) of a ping being captured
# by the "sink location" artifact (see docs for more details).
Sink = namedtuple("Sink", ["longitude", "latitude", "frequency", "reliability_model"])


class Noisyfier:
    """
    Each exposed class method provides a different kind of noise, with its parameters.
    Noise is applied to the df_pings passed as input, and it happens in place on a deep copy of the input.
    """

    def __init__(self, df_pings: pd.DataFrame, seed: int = 42):
        self.df_pings = deepcopy(df_pings)
        self.seed = seed
        self.df_pings.reset_index(drop=True, inplace=True)

        # noise model name to distribution
        self._distribution_for_selected = {  # [noise_model]
            "gaussian": self._centered_distribution_gaussian,
            "uniform": self._centered_distribution_uniform,
            "gumbel": self._centered_distribution_gumbel,
        }

        self._weights_for = {  # [reliability_model]
            "uniform": self._reliability_weights_uniform,
            "bathtub": self._reliability_weights_bathtub,
            "inverted_bathtub": self._reliability_weights_inverted_bathtub,
            "triangular_increasing": self._reliability_weights_triangular_increasing,
            "triangular_decreasing": self._reliability_weights_triangular_decreasing,
        }

    @staticmethod
    def _centered_distribution_gaussian(spread: np.ndarray) -> np.ndarray:
        return np.random.normal(0, spread)

    @staticmethod
    def _centered_distribution_uniform(spread: np.ndarray) -> np.ndarray:
        return np.random.uniform(-spread / 2, spread / 2)

    @staticmethod
    def _centered_distribution_gumbel(spread: np.ndarray) -> np.ndarray:
        return np.random.gumbel(0, spread)

    @staticmethod
    def _reliability_weights_uniform(_) -> None:
        return None

    @staticmethod
    def _reliability_weights_bathtub(ls: int) -> np.array:
        assert ls > 0
        vec = np.array([(1 / (ls / 2) ** 4) * (x - ls / 2) ** 4 for x in range(ls)])
        return vec / np.linalg.norm(vec)

    def _reliability_weights_inverted_bathtub(self, ls: int) -> np.array:
        return 1 - self._reliability_weights_bathtub(ls)

    @staticmethod
    def _reliability_weights_triangular_increasing(ls: int) -> np.array:
        assert ls > 0
        vec = np.array([(1 / ls) * x for x in range(ls)])
        return vec / np.linalg.norm(vec)

    def _reliability_weights_triangular_decreasing(self, ls: int) -> np.array:
        return 1 - self._reliability_weights_triangular_increasing(ls)

    @staticmethod
    def _time_varying_model(
        lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y, model="linear"
    ) -> object:
        """
        if model="linear" returns a linear function with given bounds on both axis.
        if model="periodic" returns abs(sin), with given upper and lower bounds on both axis.
        """
        if math.isclose(upper_bound_x - lower_bound_x, 0, abs_tol=1e-6):
            raise ValueError(
                "The upper and lower bound on the x axis can not too close to each others."
            )
        if math.isclose(upper_bound_x, 0, abs_tol=1e-6):
            raise ValueError("The upper bound on the x axis can not too close to zero.")

        def bounded_linear(x_value: float) -> float:
            if x_value < lower_bound_x:
                return lower_bound_y
            if x_value > upper_bound_x:
                return upper_bound_y
            delta_y = upper_bound_y - lower_bound_y
            delta_x = upper_bound_x - lower_bound_x
            return (x_value - lower_bound_x) * delta_y / delta_x + lower_bound_y

        def bounded_periodic(x_value: float) -> float:
            arg_sin = ((np.pi / 2) / upper_bound_x) * x_value + lower_bound_x
            return np.abs(upper_bound_y * np.sin(arg_sin) + lower_bound_y)

        map_model = {"linear": bounded_linear, "periodic": bounded_periodic}

        if model not in map_model.keys():
            raise ValueError(
                f"Allowed models are {list(map_model.keys())}. User passed {model}"
            )
        return {"linear": bounded_linear, "periodic": bounded_periodic}[model]

    @staticmethod
    def _check_keywords(keywords: Tuple[str]) -> None:
        if not isinstance(keywords, tuple):
            raise ValueError(
                f"keywords must be a tuple. User passed {keywords} of type {str(type(keywords))}."
            )

    def _check_reliability_model(self, reliability_model: str) -> None:
        if reliability_model not in self._weights_for.keys():
            raise ValueError(
                f"Allowed values for weights_model are {list(self._weights_for.keys())}. User passed {reliability_model}"
            )

    @staticmethod
    def _check_frequency(freq: float) -> None:
        if not 0 <= freq <= 1:
            raise ValueError(
                f"Parameter 'frequency' must be between 0 and 1. User passed {freq}"
            )

    def spatial_stationary_spread(
        self,
        spread: int = 5,
        keywords: Tuple[Optional[str]] = (),
        noise_model: str = "gaussian",
    ) -> None:
        """
        If noise model is "gaussian", spread = sigma in meters
        If noise model is "uniform", spread = -a/2 = b/2 in meters

        If keywords is not an empty tuple, the noise is applied only to the segments whose annotation
        contains the keyword (not case sensitive).
        """

        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            noise_lon_meters = self._distribution_for_selected[noise_model](
                np.array([spread] * len(df_segment))
            )
            noise_lat_meters = self._distribution_for_selected[noise_model](
                np.array([spread] * len(df_segment))
            )
            df_segment["longitude"] = geoutils.lon_meters_to_deg(
                geoutils.lon_deg_to_meters(
                    df_segment["longitude"], df_segment["latitude"]
                )
                + noise_lon_meters,
                df_segment["latitude"],
            )
            df_segment["latitude"] = geoutils.lat_meters_to_deg(
                geoutils.lat_deg_to_meters(df_segment["latitude"]) + noise_lat_meters
            )
            return df_segment

        self._check_keywords(keywords)

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)

        segments_with_noise = []

        for seg_annotation in self.df_pings["annotation"].drop_duplicates().to_list():
            seg_noise = self.df_pings[
                self.df_pings["annotation"] == seg_annotation
            ].copy()
            if True not in [k.lower() in seg_annotation.lower() for k in keywords]:
                segments_with_noise.append(seg_noise)
            else:
                segments_with_noise.append(noisify_segment(seg_noise))

        self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)

    def temporal_stationary_spread(
        self,
        spread: int = 5,
        noise_model: str = "gaussian",
    ) -> None:
        """
        If noise model is "gaussian", spread = sigma in seconds
        If noise model is "uniform", spread = -a/2 = b/2 in seconds

        The noise sampled by the temporal sigma is added to timestamp.
        Result is sorted and re-assigned to avoid going back in time.

        Temporal precision is 1sec. This can cause contemporary events (...as it happens in practice).
        """

        self.df_pings = self.df_pings.assign(
            timestamp=(
                pd.to_datetime(self.df_pings["timestamp"])
                + pd.to_timedelta(
                    self._distribution_for_selected[noise_model](
                        np.array([spread] * len(self.df_pings))
                    ),
                    unit="s",
                )
            )
            .dt.round(freq="S")
            .sort_values()
            .reset_index(drop=True)
        )

    def spatial_varying_spread(
        self,
        lower_bound_sec: int = 0,
        upper_bound_sec: int = 120,
        lower_bound_spread_m: int = 5,
        upper_bound_spread_m: int = 15,
        keywords: Tuple[Optional[str]] = (),
        noise_model: str = "gaussian",
        variational_model: str = "linear",
    ) -> None:
        """
        If keywords is not an empty tuple, the noise is applied only to the segments whose annotation
        contains the keyword (not case sensitive).
        """

        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            time_min = df_segment.timestamp.min()
            df_segment["time_increment"] = [
                (x - time_min).seconds for x in df_segment["timestamp"]
            ]

            df_segment["spread"] = [
                interpolation_model(ti) for ti in df_segment["time_increment"]
            ]
            df_segment["noise_latitude_m"] = self._distribution_for_selected[
                noise_model
            ](df_segment["spread"])
            df_segment["noise_longitude_m"] = self._distribution_for_selected[
                noise_model
            ](df_segment["spread"])

            df_segment["latitude"] = geoutils.lat_meters_to_deg(
                geoutils.lat_deg_to_meters(df_segment["latitude"])
                + df_segment["noise_latitude_m"]
            )
            df_segment["longitude"] = geoutils.lon_meters_to_deg(
                geoutils.lon_deg_to_meters(
                    df_segment["longitude"], df_segment["latitude"]
                )
                + df_segment["noise_longitude_m"],
                df_segment["latitude"],
            )
            df_segment.drop(
                columns=[
                    "time_increment",
                    "spread",
                    "noise_latitude_m",
                    "noise_longitude_m",
                ],
                inplace=True,
            )
            return df_segment

        self._check_keywords(keywords)

        # variational model function
        interpolation_model = self._time_varying_model(
            lower_bound_sec,
            upper_bound_sec,
            lower_bound_spread_m,
            upper_bound_spread_m,
            model=variational_model,
        )

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)

        else:
            segments_with_noise = []

            for seg_annotation in (
                self.df_pings["annotation"].drop_duplicates().to_list()
            ):
                seg_noise = self.df_pings[
                    self.df_pings["annotation"] == seg_annotation
                ].copy()
                if True not in [k.lower() in seg_annotation.lower() for k in keywords]:
                    segments_with_noise.append(seg_noise)
                else:
                    segments_with_noise.append(noisify_segment(seg_noise))

            self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)

    def temporal_varying_spread(
        self,
        lower_bound_sec: int = 0,
        upper_bound_sec: int = 120,
        lower_bound_spread_sec: int = 5,
        upper_bound_spread_sec: int = 15,
        noise_model: str = "gaussian",
        variational_model: str = "linear",
    ) -> None:
        """
        Varying the noise component of the noise over time.
        A range of noise model and variation of spread over time is allowed.
        """

        # variational model function
        interpolation_model = self._time_varying_model(
            lower_bound_sec,
            upper_bound_sec,
            lower_bound_spread_sec,
            upper_bound_spread_sec,
            model=variational_model,
        )

        time_min = self.df_pings.timestamp.min()
        self.df_pings["time_increment"] = [
            (x - time_min).seconds for x in self.df_pings["timestamp"]
        ]
        self.df_pings["spread"] = [
            interpolation_model(ti) for ti in self.df_pings["time_increment"]
        ]
        self.df_pings["noise_seconds"] = self._distribution_for_selected[noise_model](
            self.df_pings["spread"]
        )
        self.df_pings["timestamp"] = (
            (
                pd.to_datetime(self.df_pings["timestamp"])
                + pd.to_timedelta(self.df_pings["noise_seconds"], unit="s")
            )
            .dt.round(freq="S")
            .sort_values()
        )

        self.df_pings.drop(
            columns=["time_increment", "spread", "noise_seconds"], inplace=True
        )

    def missing_points(
        self,
        frequency: float = 0.4,
        reliability_model: str = "uniform",
        keywords: Tuple[Optional[str]] = (),
    ) -> None:
        """
        If keywords is not empty tuple, the noise is applied only to the segments whose annotation
        contains the keyword (not case sensitive).

        Bathtub is modelled with a simple quadratic curve for simplicity.
        ls is the length of the output segment.
        """

        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            return df_segment.sample(
                frac=1 - frequency,
                random_state=self.seed,
                weights=self._weights_for[reliability_model](len(df_segment)),
            ).sort_values(by="timestamp")

        self._check_frequency(frequency)
        self._check_reliability_model(reliability_model)
        self._check_keywords(keywords)

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)
        else:
            segments_with_noise = []
            for seg_annotation in (
                self.df_pings["annotation"].drop_duplicates().to_list()
            ):
                seg_noise = self.df_pings[
                    self.df_pings["annotation"] == seg_annotation
                ].copy()
                if True not in [k.lower() in seg_annotation.lower() for k in keywords]:
                    segments_with_noise.append(seg_noise)
                else:
                    segments_with_noise.append(noisify_segment(seg_noise))

            self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)

    def erratic_points(
        self,
        frequency: float = 0.01,
        reliability_model: str = "uniform",
        keywords: Tuple[Optional[str]] = (),
        buffer_width_meters: int = 50,
        total_sampling_for_erratic_points: int = 1000,
    ) -> None:
        """
        Only a (random) fraction of the points is affected, based on 'frequency' variable.
        A random number of points is drawn on the buffer of the pings, and then
        only the one falling in the buffer with the given buffer_width_meters are taken
        as erratic point. Empirically if there are less than 10 random point found the user
        is asked to change the parameters.

        If keywords is not (), the noise is applied only to the segments whose annotation
        contains the keyword (not case sensitive).
        """

        def random_points_in_bounds(poly, number=total_sampling_for_erratic_points):
            x_min, y_miny, x_max, y_max = poly.bounds
            x = np.random.uniform(x_min, x_max, number)
            y = np.random.uniform(y_miny, y_max, number)
            return x, y

        def bag_of_erratic_points() -> pd.DataFrame:
            gdf_pings = gpd.GeoDataFrame(
                self.df_pings.copy(),
                geometry=[
                    Point(row["longitude"], row["latitude"])
                    for _, row in self.df_pings.iterrows()
                ],
            )
            convex_hull_deg = MultiPoint([p for p in gdf_pings.geometry]).convex_hull
            buffer_spacing_deg = geoutils.meters_to_deg_average(
                buffer_width_meters, self.df_pings["latitude"].mean()
            )

            buffered_convex_hull_deg = convex_hull_deg.buffer(buffer_spacing_deg)
            buffer_of_buffer_convex_hull_deg = buffered_convex_hull_deg.buffer(
                buffer_spacing_deg
            )

            # for visualisation:
            # gdf_convex_hull_deg = gpd.GeoDataFrame(["region of randomness"], geometry=[ convex_hull_deg ])
            # gdf_buffer_deg = gpd.GeoDataFrame(["region of randomness"], geometry=[buffered_convex_hull_deg ])

            # Get bag of random points:
            x, y = random_points_in_bounds(buffer_of_buffer_convex_hull_deg)
            df = pd.DataFrame()
            df["points"] = list(zip(x, y))
            gdf_random_points = gpd.GeoDataFrame(geometry=df["points"].apply(Point))

            gdf_exterior_buffer = gpd.GeoDataFrame(
                ["region of randomness"],
                geometry=[
                    buffer_of_buffer_convex_hull_deg.difference(
                        buffered_convex_hull_deg
                    )
                ],
            )

            # gdf_random_points_within_geometry
            gdf_points_in_buffer = gpd.tools.sjoin(
                gdf_random_points, gdf_exterior_buffer, predicate="within"
            )

            df_bag_of_erratic_points = pd.DataFrame()
            df_bag_of_erratic_points["longitude"] = [
                pt.x for pt in gdf_points_in_buffer.geometry
            ]
            df_bag_of_erratic_points["latitude"] = [
                pt.y for pt in gdf_points_in_buffer.geometry
            ]
            return df_bag_of_erratic_points

        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            idx_noise_sample = df_segment.sample(
                frac=frequency,
                random_state=self.seed,
                weights=self._weights_for[reliability_model](len(df_segment)),
            ).index

            df_noisy_points = df_erratic_points.sample(
                len(idx_noise_sample), replace=True
            ).reset_index(drop=True)
            df_noisy_points.index = idx_noise_sample

            df_segment.loc[idx_noise_sample, "longitude"] = df_noisy_points["longitude"]
            df_segment.loc[idx_noise_sample, "latitude"] = df_noisy_points["latitude"]

            return df_segment

        self._check_frequency(frequency)
        self._check_reliability_model(reliability_model)
        self._check_keywords(keywords)

        df_erratic_points = bag_of_erratic_points()

        if not len(df_erratic_points) > 10:
            raise ValueError(
                "Not enough erratic points found with the given input."
                "Please Increase buffer_width_meters or increase total_sampling_for_erratic_points and start again"
            )

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)
        else:
            segments_with_noise = []

            for seg_annotation in (
                self.df_pings["annotation"].drop_duplicates().to_list()
            ):
                seg_noise = self.df_pings[
                    self.df_pings["annotation"] == seg_annotation
                ].copy()

                if len(keywords) > 0 and True not in [
                    k.lower() in seg_annotation.lower() for k in keywords
                ]:
                    segments_with_noise.append(seg_noise)
                else:
                    segments_with_noise.append(noisify_segment(seg_noise))

            self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)

    def gridding(
        self,
        precision: int = 4,
        frequency: float = 0.01,
        reliability_model: str = "uniform",
        keywords: Tuple[Optional[str]] = (),
    ) -> None:
        """
        Only a (random) fraction of the points is affected, based on frequency.
        Only 1% of points are affected by default

        If keywords is not empty tuple, the noise is applied only to the segments whose annotation
        contains the keyword (not case sensitive).
        """

        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            idx_noise_sample = df_segment.sample(
                frac=frequency,
                random_state=self.seed,
                weights=self._weights_for[reliability_model](len(df_segment)),
            ).index
            df_segment["noisy"] = False
            df_segment.loc[idx_noise_sample, "noisy"] = True
            df_segment["latitude"] = df_segment.apply(
                lambda row: np.round(row["latitude"], precision)
                if row["noisy"]
                else row["latitude"],
                axis=1,
            )
            df_segment["longitude"] = df_segment.apply(
                lambda row: np.round(row["longitude"], precision)
                if row["noisy"]
                else row["longitude"],
                axis=1,
            )
            df_segment.drop(columns=["noisy"], inplace=True)
            return df_segment

        self._check_frequency(frequency)
        self._check_keywords(keywords)
        self._check_reliability_model(reliability_model)

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)
        else:
            segments_with_noise = []

            for seg_annotation in (
                self.df_pings["annotation"].drop_duplicates().to_list()
            ):
                seg_noise = self.df_pings[
                    self.df_pings["annotation"] == seg_annotation
                ].copy()

                if len(keywords) > 0 and True not in [
                    k.lower() in seg_annotation.lower() for k in keywords
                ]:
                    segments_with_noise.append(seg_noise)
                else:
                    segments_with_noise.append(noisify_segment(seg_noise))

            self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)

    def sink_locations(
        self,
        list_sinks: List[Sink],
        keywords: Tuple[Optional[str]] = (),
    ) -> None:
        def noisify_segment(df_segment: pd.DataFrame) -> pd.DataFrame:
            for id_sink, sink in enumerate(list_sinks):
                idx_noise_sample = df_segment.sample(
                    frac=sink.frequency,
                    random_state=self.seed + id_sink,
                    weights=self._weights_for[sink.reliability_model](len(df_segment)),
                ).index
                df_segment.loc[idx_noise_sample, "longitude"] = sink.longitude
                df_segment.loc[idx_noise_sample, "latitude"] = sink.latitude

            return df_segment

        self._check_keywords(keywords)
        for sink in list_sinks:
            self._check_reliability_model(sink.reliability_model)

        if len(keywords) == 0:
            self.df_pings = noisify_segment(self.df_pings).reset_index(drop=True)
        else:
            segments_with_noise = []

            for seg_annotation in (
                self.df_pings["annotation"].drop_duplicates().to_list()
            ):
                seg_noise = self.df_pings[
                    self.df_pings["annotation"] == seg_annotation
                ].copy()

                if len(keywords) > 0 and True not in [
                    k.lower() in seg_annotation.lower() for k in keywords
                ]:
                    segments_with_noise.append(seg_noise)
                else:
                    segments_with_noise.append(noisify_segment(seg_noise))

            self.df_pings = pd.concat(segments_with_noise).reset_index(drop=True)
