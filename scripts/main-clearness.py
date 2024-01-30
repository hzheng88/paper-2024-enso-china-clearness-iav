# %%
# libraries and constants
import math
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np

DATETIME_REF = datetime(2000, 1, 1)
DATETIME_UNITS = f"seconds since {DATETIME_REF.isoformat()}Z"

# %%
# settings
# place the CMFD SRad data under DATAROOT
DATAROOT = Path("../data/SRad")
OUTPUT = Path("../data/clearness.3hour.nc")
DATETIME_START = datetime(1979, 1, 1)
DATETIME_STOP = datetime(2021, 1, 1)
DATETIME_STEP = timedelta(hours=3)


# %%
def cmfdfilepath(dt: datetime) -> Path:
    return DATAROOT / f"srad_CMFD_V0107_B-01_03hr_010deg_{dt:%Y%m}.nc"


# %%
with h5netcdf.File(cmfdfilepath(DATETIME_START)) as f:
    lon: np.ndarray = f["lon"][:]
    lat: np.ndarray = f["lat"][:]
lon2d, lat2d = np.meshgrid(lon, lat)

with h5netcdf.File("../data/mask.nc") as f:
    mask = f["mask"][0, :, :].astype(bool)
# %%


def extraterrestrialradiation(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    dt: datetime,
    before=timedelta(minutes=90),
    after=timedelta(minutes=90),
):
    """Calculate extraterrestrial radiation.

    Parameters
    ----------
    lat2d : float
        Latitude in degrees.
    lon2d : float
        Longitude in degrees.
    time : datetime
        Time.

    Returns
    -------
    float
        Extraterrestrial radiation in W/m^2.
    """
    rad1sec = math.pi / 12 / 3600  # 1 second in radians
    gamma = (
        2
        * math.pi
        * (dt - datetime(dt.year, 1, 1)).total_seconds()
        / (datetime(dt.year + 1, 1, 1) - datetime(dt.year, 1, 1)).total_seconds()
    )
    eccentricity = (
        1.000110
        + 0.034221 * math.cos(gamma)
        + 0.001280 * math.sin(gamma)
        + 0.000719 * math.cos(2 * gamma)
        + 0.000077 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    declination = np.deg2rad(23.45) * np.sin(gamma + 284.0 / 365.0 * 2 * np.pi)
    omega_sunset = np.arccos(-np.tan(np.deg2rad(lat2d)) * math.tan(declination))
    omega_sunrise = -omega_sunset
    eqt = (
        229.18
        * 60
        * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2 * gamma)
            - 0.040849 * math.sin(2 * gamma)
        )
    )
    tst1 = (
        dt.hour * 3600
        + dt.minute * 60
        + dt.second
        - before.total_seconds()
        + eqt
        + lon2d * 240
    )
    tst1[tst1 < 0] += 86400
    tst2 = (
        dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + after.total_seconds()
        + eqt
        + lon2d * 240
    )
    tst2[tst2 < 0] += 86400
    interday = tst2 < tst1
    omega1 = rad1sec * (tst1 - 12 * 3600)
    omega2 = rad1sec * (tst2 - 12 * 3600)
    ret1 = extraterrestrialradiation_helper(
        lat2d,
        eccentricity,
        declination,
        np.minimum(omega2, omega_sunset),
        np.maximum(omega1, omega_sunrise),
    )
    ret2 = extraterrestrialradiation_helper(
        lat2d,
        eccentricity,
        declination,
        omega2,
        omega_sunrise,
    ) + extraterrestrialradiation_helper(
        lat2d,
        eccentricity,
        declination,
        omega_sunset,
        omega1,
    )
    return np.where(interday, ret2, ret1)


def extraterrestrialradiation_helper(
    lat2d: np.ndarray,
    eccentricity: float,
    declination: float,
    omega2: np.ndarray,
    omega1: np.ndarray,
):
    SC = 1367  # solar constant in W/m^2
    rad1sec = math.pi / 12 / 3600  # 1 second in radians
    ret = (
        SC
        / rad1sec
        * eccentricity
        * (
            np.sin(np.deg2rad(lat2d)) * np.sin(declination) * (omega2 - omega1)
            + np.cos(np.deg2rad(lat2d))
            * np.cos(declination)
            * (np.sin(omega2) - np.sin(omega1))
        )
    )
    ret[omega2 < omega1] = 0
    return ret


# %%

cdt = DATETIME_START
with h5netcdf.File(OUTPUT, "w") as fo:
    fo.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
    fo.create_variable("time", ("time",), "f8")
    fo["time"].attrs.update(
        {"units": np.bytes_(DATETIME_UNITS), "calendar": np.bytes_("standard")}
    )
    fo.create_variable("lat", ("lat",), "f8", data=lat)
    fo["lat"].attrs.update(
        {"units": np.bytes_("degrees_north"), "standard_name": np.bytes_("latitude")}
    )
    fo.create_variable("lon", ("lon",), "f8", data=lon)
    fo["lon"].attrs.update(
        {"units": np.bytes_("degrees_east"), "standard_name": np.bytes_("longitude")}
    )
    fo.create_variable(
        "clearness",
        ("time", "lat", "lon"),
        "f4",
        fillvalue=np.nan,
        compression="gzip",
    )
    fo["clearness"].attrs.update(
        {
            "units": np.bytes_("1"),
            "long_name": np.bytes_("clearness index"),
            "valid_range": [0.0, 1.0],
        }
    )
    while cdt < DATETIME_STOP:
        if cdt.month == 1 and cdt.day == 1 and cdt.hour == 0:
            print(cdt)
        with h5netcdf.File(cmfdfilepath(cdt)) as f:
            if (cdt - datetime(cdt.year, cdt.month, 1)) % DATETIME_STEP != timedelta(0):
                raise ValueError(
                    f"Time step not multiple of {DATETIME_STEP.total_seconds()} s."
                )
            idt = (cdt - datetime(cdt.year, cdt.month, 1)) // DATETIME_STEP
            ghi: np.ndarray[tuple[int, int], np.dtype[np.floating]] = (
                f["srad"][idt, :, :] * 3600 * 3
            )
            ghi[~mask] = np.nan
            esr = extraterrestrialradiation(lat2d, lon2d, cdt)
            esr[esr < 120 * 3600 * 3] = np.nan
            ci = np.minimum(np.maximum(ghi / esr, 0.0), 1.0)
            fo.resize_dimension("time", len(fo.dimensions["time"]) + 1)
            fo["time"][-1] = (cdt - DATETIME_REF).total_seconds()
            fo["clearness"][-1, :, :] = ci[:, :]
            cdt += DATETIME_STEP
