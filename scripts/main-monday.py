# %%
# libraries and constants
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np

DATETIME_REF = datetime(2000, 1, 1)
DATETIME_UNITS = f"seconds since {DATETIME_REF.isoformat()}+08:00"

# %%
# settings
HOURFILE = Path("../data/clearness.3hr.nc")
DAYFILE = Path("../data/clearness.day.nc")
MONFILE = Path("../data/clearness.mon.nc")

DATETIME_START = datetime(1979, 1, 1)
DATETIME_STOP = datetime(2021, 1, 1)
DATETIME_STEP = timedelta(hours=3)
DATETIME_TIMEZONE = timedelta(hours=8)

# %%
with h5netcdf.File("../data/mask.nc") as f:
    lat: np.ndarray = f["lat"][:]
    lon: np.ndarray = f["lon"][:]
    mask = f["mask"][0, :, :].astype(bool)


# %%

with h5netcdf.File(HOURFILE) as f:
    time_3hr = [
        DATETIME_REF + DATETIME_TIMEZONE + timedelta(seconds=x) for x in f["time"][:]
    ]

# %%
# monthly file
with h5netcdf.File(MONFILE, "w") as fo:
    fo.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
    fo.create_variable("time", ("time",), "f8")
    fo["time"].attrs.update(
        {"units": np.bytes_(DATETIME_UNITS), "calendar": np.bytes_("standard")}
    )
    fo.create_variable(
        "lat", ("lat",), "f8", data=lat, compression="gzip", compression_opts=9
    )
    fo["lat"].attrs.update(
        {"units": np.bytes_("degrees_north"), "standard_name": np.bytes_("latitude")}
    )
    fo.create_variable(
        "lon", ("lon",), "f8", data=lon, compression="gzip", compression_opts=9
    )
    fo["lon"].attrs.update(
        {"units": np.bytes_("degrees_east"), "standard_name": np.bytes_("longitude")}
    )
    fo.create_variable(
        "clearness",
        ("time", "lat", "lon"),
        "f4",
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=9,
    )
    fo["clearness"].attrs.update(
        {
            "units": np.bytes_("1"),
            "long_name": np.bytes_("clearness index"),
            "valid_range": [0.0, 1.0],
        }
    )
    cmon = DATETIME_START
    while cmon < DATETIME_STOP:
        print(cmon)
        idxofday = filter(
            lambda x: x[1].year == cmon.year and x[1].month == cmon.month,
            enumerate(time_3hr),
        )
        idx = [x[0] for x in idxofday]
        with h5netcdf.File(HOURFILE) as f:
            cmondata = np.nanmean(f["clearness"][idx, :, :], axis=0)
        fo.resize_dimension("time", len(fo.dimensions["time"]) + 1)
        fo["time"][-1] = (cmon - DATETIME_REF).total_seconds()
        fo["clearness"][-1, :, :] = cmondata[:, :]
        if cmon.month == 12:
            cmon = datetime(cmon.year + 1, 1, 1)
        else:
            cmon = datetime(cmon.year, cmon.month + 1, 1)

# %%
# daily file
with h5netcdf.File(DAYFILE, "w") as fo:
    fo.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
    fo.create_variable("time", ("time",), "f8")
    fo["time"].attrs.update(
        {"units": np.bytes_(DATETIME_UNITS), "calendar": np.bytes_("standard")}
    )
    fo.create_variable(
        "lat", ("lat",), "f8", data=lat, compression="gzip", compression_opts=9
    )
    fo["lat"].attrs.update(
        {"units": np.bytes_("degrees_north"), "standard_name": np.bytes_("latitude")}
    )
    fo.create_variable(
        "lon", ("lon",), "f8", data=lon, compression="gzip", compression_opts=9
    )
    fo["lon"].attrs.update(
        {"units": np.bytes_("degrees_east"), "standard_name": np.bytes_("longitude")}
    )
    fo.create_variable(
        "clearness",
        ("time", "lat", "lon"),
        "f4",
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=9,
    )
    fo["clearness"].attrs.update(
        {
            "units": np.bytes_("1"),
            "long_name": np.bytes_("clearness index"),
            "valid_range": [0.0, 1.0],
        }
    )
    cday = DATETIME_START
    while cday < DATETIME_STOP:
        print(cday)
        idxofday = filter(lambda x: x[1].date() == cday.date(), enumerate(time_3hr))
        idx = [x[0] for x in idxofday]
        with h5netcdf.File(HOURFILE) as f:
            cdaydata = np.nanmean(f["clearness"][idx, :, :], axis=0)
        fo.resize_dimension("time", len(fo.dimensions["time"]) + 1)
        fo["time"][-1] = (cday - DATETIME_REF).total_seconds()
        fo["clearness"][-1, :, :] = cdaydata[:, :]
        cday += timedelta(days=1)
