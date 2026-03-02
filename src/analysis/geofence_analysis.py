from __future__ import annotations

from typing import Dict, Union, List

import pandas as pd

from src.analysis.common import logger
from src.analysis.activity_plots import (
    save_geofence_hourly_activity_plot,
    save_geofence_speed_by_hour_plot,
    save_geofence_movement_trajectory_plot,
    save_geofence_3d_visualization_plot,
)


def analyze_geofence_data(json_data: Dict[str, Union[pd.DataFrame, dict]]) -> None:
    geofence_dfs: List[pd.DataFrame] = []

    for key, data in json_data.items():
        if "geofence" not in key.lower():
            continue
        if not isinstance(data, pd.DataFrame):
            continue

        try:
            parts = key.split("_")
            if len(parts) >= 3 and parts[0] in ("player", "user"):
                player_id = parts[1]
            else:
                logger.warning(f"Key {key} does not match expected geofence naming; skipping")
                continue

            df = data.copy()
            df["player_id"] = player_id
            df["user_id"] = player_id  # backward compatibility

            required_columns = ["LATITUDE", "LONGITUDE", "ALTITUDE", "SPEED", "ERROR", "TIMESTAMP"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = pd.NaT if col == "TIMESTAMP" else 0

            geofence_dfs.append(df)

        except Exception as e:
            logger.warning(f"Error extracting player id from {key}: {e}")

    if not geofence_dfs:
        logger.warning("No geofence data found")
        return

    all_geofence_data = pd.concat(geofence_dfs, ignore_index=True)

    if "TIMESTAMP" in all_geofence_data.columns:
        all_geofence_data["TIMESTAMP"] = pd.to_datetime(
            all_geofence_data["TIMESTAMP"],
            errors="coerce",
        )

    for col in ["LATITUDE", "LONGITUDE", "ALTITUDE", "SPEED", "ERROR"]:
        if col in all_geofence_data.columns:
            all_geofence_data[col] = pd.to_numeric(
                all_geofence_data[col],
                errors="coerce",
            )

    speed_series = all_geofence_data["SPEED"].dropna()
    if speed_series.empty:
        logger.warning("No valid SPEED values in geofence data; skipping geofence analysis")
        return

    # Outlier trimming (3 * IQR)
    q1 = speed_series.quantile(0.25)
    q3 = speed_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    filtered = all_geofence_data[
        (all_geofence_data["SPEED"] >= lower_bound)
        & (all_geofence_data["SPEED"] <= upper_bound)
    ].copy()

    if filtered.empty:
        logger.warning("All geofence rows were removed after outlier filtering; skipping geofence plots")
        return

    logger.info(f"Geofence outliers removed: {len(all_geofence_data) - len(filtered)}")

    # Temporal analysis
    if "TIMESTAMP" in filtered.columns and not filtered["TIMESTAMP"].isna().all():
        filtered["hour_of_day"] = filtered["TIMESTAMP"].dt.hour
        filtered["hour_of_day"] = pd.Categorical(
            filtered["hour_of_day"],
            categories=list(range(24)),
            ordered=True,
        )

        save_geofence_hourly_activity_plot(filtered)
        save_geofence_speed_by_hour_plot(filtered)

        try:
            trajectory_data = filtered.sort_values("TIMESTAMP").dropna(
                subset=["LONGITUDE", "LATITUDE"]
            )
            save_geofence_movement_trajectory_plot(trajectory_data)
        except Exception as e:
            logger.warning(f"Could not create movement trajectory plot: {e}")

    # 3D visualization
    try:
        save_geofence_3d_visualization_plot(filtered)
    except Exception as e:
        logger.warning(f"Could not create 3D visualization: {e}")