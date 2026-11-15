"""Feature extraction for one place (copy of the code in model/main.ipynb, sections 3 and 4).

If you change a feature in the notebook, change it here too, then re-run the notebook so the
saved model (data/analog_model.joblib) matches.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---- settings (same as the notebook) ------------------------------------------------------
PATCH_HALF_DEG = 0.10          # area around each place: +/-0.1 deg (~ +/-11 km)
PATCH_PX = 48                  # image is 48 x 48 pixels (~460 m / pixel)
DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "terrax/0.1"}

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
POWER = "https://power.larc.nasa.gov/api/temporal/climatology/point"
MACROSTRAT = "https://macrostrat.org/api/v2/geologic_units/map"
LST_COLORMAP_URL = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Land_Surface_Temp.xml"
LST_MONTHS = ["2023-01-01", "2023-07-01"]
NDVI_COLORMAP_URL = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_NDVI.xml"
NDVI_MONTHS = ["2023-01-01", "2023-07-01"]
POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN"]


# ---- downloads ----------------------------------------------------------------------------
def _get(url, params=None, expect=None, tries=4):
    # GET with simple retry; `expect` = required start of the Content-Type
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200 and (expect is None or r.headers.get("Content-Type", "").startswith(expect)):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url} {params}")


def gibs_patch(layer, lat, lon, half=PATCH_HALF_DEG, px=PATCH_PX, date=None):
    # small square image of a GIBS layer centred on lat/lon -> float array (px, px, 4) in 0..1
    tag = f"_{date}" if date else ""
    f = CACHE / f"{layer}{tag}_{lat:.3f}_{lon:.3f}_{half}_{px}.png"
    if not f.exists():
        params = dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=layer,
                      CRS="EPSG:4326", STYLES="", FORMAT="image/png", WIDTH=px, HEIGHT=px,
                      BBOX=f"{lat-half},{lon-half},{lat+half},{lon+half}")
        if date:
            params["TIME"] = date
        f.write_bytes(_get(GIBS_WMS, params, expect="image").content)
    return np.asarray(Image.open(f).convert("RGBA"), dtype=np.float32) / 255.0


def power_climate(lat, lon):
    # long-term annual climate from NASA POWER
    f = CACHE / f"power_{lat:.3f}_{lon:.3f}.json"
    if not f.exists():
        r = _get(POWER, dict(parameters=",".join(POWER_PARAMS), community="AG",
                             latitude=lat, longitude=lon, format="JSON"), expect="application/json")
        f.write_text(r.text, encoding="utf-8")
    p = json.loads(f.read_text(encoding="utf-8"))["properties"]["parameter"]
    return {k: (np.nan if p[k]["ANN"] == -999 else p[k]["ANN"]) for k in POWER_PARAMS}


def srgb_to_linear(v):
    # undo the display brightness curve (sRGB gamma) -> roughly proportional to reflectance
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


# ---- surface temperature (MODIS colour image -> kelvin) -----------------------------------
_cmf = CACHE / "MODIS_Land_Surface_Temp_colormap.xml"
if not _cmf.exists():
    _cmf.write_text(_get(LST_COLORMAP_URL).text, encoding="utf-8")
_entries = re.findall(r'rgb="(\d+),(\d+),(\d+)"[^>]*?value="\[([\d.]+),([\d.]+)\)"', _cmf.read_text(encoding="utf-8"))
CM_RGB = np.array([[int(r), int(g), int(b)] for r, g, b, _, _ in _entries], dtype=np.float32) / 255
CM_K = np.array([(float(lo) + float(hi)) / 2 for *_, lo, hi in _entries])
CM_K[0], CM_K[-1] = 200.0, 350.0


def decode_lst(img):
    rgb = img[..., :3].reshape(-1, 1, 3)
    dist = ((rgb - CM_RGB[None]) ** 2).sum(-1)
    k = CM_K[dist.argmin(1)].reshape(img.shape[:2])
    nodata = (img[..., 3] == 0) | (dist.min(1).reshape(img.shape[:2]) > 0.003)
    return np.where(nodata, np.nan, k)


def lst_season(lat, lon):
    # July - January daytime ground temperature (K), per pixel
    jan, jul = [decode_lst(gibs_patch("MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day", lat, lon, date=m)) for m in LST_MONTHS]
    return np.abs(jul - jan)


# ---- vegetation (MODIS NDVI colour image -> NDVI) -----------------------------------------
_ndf = CACHE / "MODIS_NDVI_colormap.xml"
if not _ndf.exists():
    _ndf.write_text(_get(NDVI_COLORMAP_URL).text, encoding="utf-8")
_nentries = re.findall(r'rgb="(\d+),(\d+),(\d+)" transparent="false"[^>]*?value="\[([\d.]+),([\d.]+)\)"', _ndf.read_text(encoding="utf-8"))
NDVI_RGB = np.array([[int(r), int(g), int(b)] for r, g, b, _, _ in _nentries], dtype=np.float32) / 255
NDVI_VAL = np.array([(float(lo) + float(hi)) / 2 for *_, lo, hi in _nentries])


def decode_ndvi(img):
    # transparent pixels are NDVI <= 0 (bare ground, snow) -> 0
    rgb = img[..., :3].reshape(-1, 1, 3)
    v = NDVI_VAL[((rgb - NDVI_RGB[None]) ** 2).sum(-1).argmin(1)].reshape(img.shape[:2])
    return np.where(img[..., 3] == 0, 0.0, v)


def ndvi_peak(lat, lon):
    # peak-season greenness per pixel: the higher of January and July
    jan, jul = [decode_ndvi(gibs_patch("MODIS_Terra_L3_NDVI_Monthly", lat, lon, date=m)) for m in NDVI_MONTHS]
    return np.maximum(jan, jul)


# ---- rock type (Macrostrat geologic maps) -------------------------------------------------
ROCK_RULES = [
    ("impact",           ["impact", "breccia", "suevite"]),
    ("evaporite",        ["evaporite", "salt", "gypsum", "halite", "playa", "saline", "sabkha", "sebkha"]),
    ("mafic_volcanic",   ["basalt", "basaltic", "mafic volcanic", "lava"]),
    ("felsic_volcanic",  ["rhyolite", "dacite", "tuff", "ignimbrite", "andesite", "felsic volcanic", "pyroclastic"]),
    ("volcanic",         ["volcanic"]),
    ("intrusive",        ["granite", "gabbro", "anorthosite", "diorite", "intrusive", "plutonic", "granodiorite", "syenite"]),
    ("metamorphic",      ["gneiss", "schist", "metamorphic", "quartzite", "marble", "amphibolite", "greenstone"]),
    ("carbonate",        ["limestone", "dolomite", "carbonate", "chalk", "travertine"]),
    ("unconsolidated",   ["alluvium", "alluvial", "unconsolidated", "sand", "gravel", "dune", "loess", "colluvium", "till", "glacial", "eolian", "aeolian"]),
    ("clastic_sediment", ["sandstone", "shale", "siltstone", "conglomerate", "mudstone", "sedimentary", "siliciclastic", "clastic"]),
]
ROCK_CLASSES = [cls for cls, _ in ROCK_RULES] + ["unknown"]
SCALE_WEIGHT = {"large": 4, "medium": 3, "small": 2, "tiny": 1}


def classify_unit(u):
    text = f"{u.get('lith', '')} {u.get('name', '')}".lower()
    for cls, words in ROCK_RULES:
        if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):
            return cls
    return None


def rock_type(lat, lon):
    f = CACHE / f"macrostrat_{lat:.3f}_{lon:.3f}.json"
    if not f.exists():
        f.write_text(_get(MACROSTRAT, dict(lat=lat, lng=lon), expect="application/json").text, encoding="utf-8")
    votes = {}
    for u in json.loads(f.read_text(encoding="utf-8")).get("success", {}).get("data", []):
        cls = classify_unit(u)
        if cls:
            votes[cls] = votes.get(cls, 0) + SCALE_WEIGHT.get(u.get("scale"), 1)
    return max(votes, key=votes.get) if votes else "unknown"


# ---- all features for one place -----------------------------------------------------------
def place_features(lat, lon):
    # one row per land pixel around lat/lon
    c = gibs_patch("BlueMarble_NextGeneration", lat, lon)
    relief = gibs_patch("ASTER_GDEM_Greyscale_Shaded_Relief", lat, lon)[..., 0]
    r, g, b, alpha = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    brightness = (r + g + b) / 3
    lin = srgb_to_linear(c[..., :3]) + 1e-3
    f = {
        "b": b,
        "brightness": brightness,
        "slope_rg": np.log(lin[..., 0] / lin[..., 1]),
        "slope_gb": np.log(lin[..., 1] / lin[..., 2]),
        "lst_season": lst_season(lat, lon),
        "ndvi_peak": ndvi_peak(lat, lon),
    }
    df = pd.DataFrame({k: v.ravel() for k, v in f.items()})
    water = (b > r) & (b > g) & (brightness < 0.35)
    df = df[((alpha > 0) & ~water).ravel()].reset_index(drop=True)
    df["patch_roughness"] = relief.std()
    df["patch_color_var"] = brightness.std()
    df["patch_slope_gb"] = df["slope_gb"].mean()
    df["patch_ndvi"] = df["ndvi_peak"].mean()
    clim = power_climate(lat, lon)
    for k, v in clim.items():
        df[k] = v
    df["temp_range"] = clim["T2M_MAX"] - clim["T2M_MIN"]
    df["rock"] = pd.Categorical([rock_type(lat, lon)] * len(df), categories=ROCK_CLASSES)
    return df
