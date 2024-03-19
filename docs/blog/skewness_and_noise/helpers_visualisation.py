# pylint: disable=missing-function-docstring
from copy import deepcopy

import pandas as pd
import altair as alt


from keplergl import KeplerGl

from kepler_configs import conf_trajectory
from dwells_geometries import gdf_geometries

from noisyfier import Noisyfier


alt.themes.enable("dark")


def get_map(noisifier_instance: Noisyfier) -> KeplerGl:
    df_trajectory = noisifier_instance.df_pings.copy()
    df_trajectory["ids"] = list(df_trajectory.index)
    df_trajectory["timestamp"] = df_trajectory["timestamp"].astype(str)
    df_trajectory_lines = df_trajectory.copy()
    df_trajectory_lines["longitude_next"] = df_trajectory_lines["longitude"].shift(-1)
    df_trajectory_lines["latitude_next"] = df_trajectory_lines["latitude"].shift(-1)
    df_trajectory_lines["timestamp"] = df_trajectory_lines["timestamp"].astype(str)

    return KeplerGl(
        data=deepcopy(
            {
                "pings": df_trajectory,
                "lines": df_trajectory_lines,
                "dwell locations": gdf_geometries,
            }
        ),
        height=1200,
        config=conf_trajectory,
    )


def get_timestamp_histogram(df_pings: pd.DataFrame, title: str = "title") -> alt.Chart:
    df_pings = df_pings.copy()
    bins = pd.date_range(df_pings.timestamp.min(), df_pings.timestamp.max(), 70)
    binned = (
        pd.DataFrame(
            df_pings.groupby(
                pd.cut(pd.to_datetime(df_pings.timestamp), bins=bins)
            ).size()
        )
        .reset_index(drop=True)
        .rename(columns={0: "size"})
    )
    binned["bin_min"] = bins[:-1]
    binned["bin_max"] = bins[1:]

    return (
        alt.Chart(binned, title=title)
        .mark_bar()
        .encode(
            x=alt.X("bin_min", bin="binned", title="Time"),
            x2="bin_max",
            y=alt.Y(
                "size",
                # scale=alt.Scale(type="time"),
                title="Number of records",
            ),
        )
        .properties(width=800, height=250)
        .interactive(False)
    )


def get_two_timestamp_histograms(
    df_pings1: pd.DataFrame(),
    df_pings2: pd.DataFrame(),
    title1: str,
    title2: str,
) -> alt.Chart:
    upper = get_timestamp_histogram(df_pings1, title=title1)
    lower = get_timestamp_histogram(df_pings2, title=title2)

    return (
        alt.vconcat(upper, lower)
        .configure_axis(
            labelFontSize=13,
            titleFontSize=13,
        )
        .configure_title(
            fontSize=15,
            anchor="start",
        )
    )
