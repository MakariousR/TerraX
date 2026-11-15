"""TerraX - website.

Run:   python webapp/app.py      then open http://localhost:5000
Needs: data/analog_model.joblib, data/web_meta.json and data/web_sites.json, created by section 10 of model/main.ipynb.
"""
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from features import (DATA, GIBS_WMS, _get, place_features, features_from, area_layers, power_cell,
                      power_climate, rock_type)

app = Flask(__name__, static_folder="static")

model = joblib.load(DATA / "analog_model.joblib")
meta = json.loads((DATA / "web_meta.json").read_text(encoding="utf-8"))
FEATURES, CLASSES = meta["features"], meta["classes"]
NUMERIC = [f for f in FEATURES if f != "rock"]
MU, SD = pd.Series(meta["mu"]), pd.Series(meta["sd"])
ANALOGS = pd.DataFrame(meta["analogs"])
SITES = pd.read_json(DATA / "web_sites.json")                      # known training sites (for the cold-region caution)
TYPICAL = meta["typical"]
BOTH_MIN = meta.get("both_min", 0.30)        # Moon and Mars both at least this and both above Neither -> "Both" (same as the notebook)
# colour shape [ln(R/G), ln(G/B)] of real samples from NASA PDS RELAB lab spectra (section 3 of the notebook);
# shown on the card for comparison only, not used by the model
PDS_REF = meta.get("pds_ref", {"moon_soil": [0.105, 0.130], "mars_dust": [0.571, 0.867], "mars_rock": [0.062, 0.186]})
PDS_NAMES = {"moon_soil": "Apollo lunar soil", "mars_dust": "Martian dust simulant (JSC Mars-1)", "mars_rock": "Martian meteorites"}
PDS_MATCH = 0.7     # 99 % of the documented analogs are within this colour distance of a sample; half of ordinary land isn't

# NASA GIBS land mask of the whole world at 0.5 deg: lets us reject ocean clicks instantly (downloaded once)
_mask_file = DATA / "cache" / "OSM_Land_Mask_global_0.5deg.png"
if not _mask_file.exists():
    _mask_file.write_bytes(_get(GIBS_WMS, dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS="OSM_Land_Mask",
                                               CRS="EPSG:4326", STYLES="", FORMAT="image/png", WIDTH=720, HEIGHT=360,
                                               BBOX="-90,-180,90,180"), expect="image").content)
LAND = np.asarray(Image.open(_mask_file).convert("RGBA"))[..., 3] > 128          # True = land


def is_ocean(lat, lon):
    r, c = min(int((90 - lat) / 0.5), 359), min(int((lon + 180) / 0.5), 719)
    window = LAND[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]          # neighbours too, so coasts are allowed
    return not window.any()

NICE = {"patch_slope_gb": "area colour (green/blue)", "rock": "rock type", "patch_color_var": "colour patchiness",
        "RH2M": "humidity", "patch_roughness": "terrain roughness", "lst_season": "seasonal ground-temperature swing",
        "temp_range": "air-temperature range", "ALLSKY_SFC_SW_DWN": "sunlight",
        "slope_rg": "pixel colour (red/green)", "T2M": "mean temperature",
        "slope_gb": "pixel colour (green/blue)", "brightness": "brightness", "b": "blue reflectance",
        "ndvi_peak": "pixel summer greenness", "patch_ndvi": "summer greenness (NDVI)"}
# rows of the "Conditions" table; must match SHOWN in section 10 of model/main.ipynb
SHOW = [("rock", "Rock type", ""), ("RH2M", "Humidity", " %"),
        ("temp_range", "Air-temperature range", " °C"), ("lst_season", "Seasonal ground swing", " K"),
        ("patch_ndvi", "Summer greenness (NDVI)", ""),
        ("patch_roughness", "Terrain roughness", ""), ("patch_slope_gb", "Area colour (G/B)", "")]


def confidence(p):
    # how decisive the answer is: probability of the verdict (for Both: of being an analog at all)
    return "high" if p >= 0.75 else "medium" if p >= 0.55 else "low"


def judge(proba):
    # average probabilities -> (verdict, confidence), with the same Both rule as the notebook
    moon, mars, neither = CLASSES.index("Moon"), CLASSES.index("Mars"), CLASSES.index("Neither")
    if proba[moon] >= BOTH_MIN and proba[mars] >= BOTH_MIN and min(proba[moon], proba[mars]) > proba[neither]:
        return "Both", confidence(proba[moon] + proba[mars])
    best = int(np.argmax(proba))
    return CLASSES[best], confidence(proba[best])


TILE = 0.2          # one square = the same ±0.1° (about 22 km) area as a single click
MAX_TILES = 10      # at most 10 x 10 = 100 squares per scan (2° x 2°, about 220 km)
BLOCK = 10          # images are downloaded in blocks of up to 10 x 10 squares (480 x 480 pixels)
PX = 48             # pixels per square, as for a single click


def fmt(f, v):
    if f == "rock":
        return str(v).replace("_", " ")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/sites")
def sites():
    return app.response_class((DATA / "web_sites.json").read_text(encoding="utf-8"), mimetype="application/json")


@app.get("/api/predict")
def predict():
    try:
        lat = round(float(request.args["lat"]), 2)
        lon = round(((float(request.args["lon"]) + 180) % 360) - 180, 2)   # wrap clicks on repeated worlds
    except (KeyError, ValueError):
        return jsonify(error="lat and lon are required numbers"), 400
    if abs(lat) > 82:
        return jsonify(error="Too close to the pole: there is no usable imagery beyond 82°."), 400
    if is_ocean(lat, lon):
        return jsonify(error="This spot is in the ocean. Click on land."), 422
    try:
        df = place_features(lat, lon)
    except Exception as e:                                               # a data service failed
        return jsonify(error=f"Could not download the data for this spot ({e}). Try again."), 502
    if len(df) < 100:
        return jsonify(error="This spot is mostly water or has no data. Click on land."), 422

    X = df[FEATURES]
    proba = model.predict_proba(X).mean(0)
    best = int(np.argmax(proba))
    moon, mars = CLASSES.index("Moon"), CLASSES.index("Mars")
    neither = CLASSES.index("Neither")
    both = proba[moon] >= BOTH_MIN and proba[mars] >= BOTH_MIN and min(proba[moon], proba[mars]) > proba[neither]
    prediction = "Both" if both else CLASSES[best]
    contrib = model.predict(X, pred_contrib=True).reshape(len(X), len(CLASSES), len(FEATURES) + 1)
    pushed = contrib[:, moon, :-1] + contrib[:, mars, :-1] if both else contrib[:, best, :-1]   # Both: what made it Moon- and Mars-like
    c = pd.Series(pushed.mean(0), index=FEATURES).sort_values(ascending=False)

    values = {f: float(df[f].median()) for f in NUMERIC}
    values["rock"] = str(df["rock"].iloc[0])
    reasons = [f"{NICE[f]}: {fmt(f, values[f])}" for f in c.index[:3] if c[f] > 0]

    z = ((pd.Series(values)[NUMERIC].astype(float) - MU) / SD).fillna(0)
    dist = (((ANALOGS[NUMERIC] - MU) / SD).fillna(0) - z).pow(2).sum(axis=1)
    twin = ANALOGS.loc[dist.idxmin()]

    conditions = [{"name": name, "here": fmt(f, values[f]) + unit,
                   "moon": fmt(f, TYPICAL["Moon"][f]) + unit, "mars": fmt(f, TYPICAL["Mars"][f]) + unit}
                  for f, name, unit in SHOW]

    conf = confidence(proba[moon] + proba[mars] if both else proba[best])
    cautions = []
    la, lo, sla, slo = np.radians(lat), np.radians(lon), np.radians(SITES.lat.values), np.radians(SITES.lon.values)
    near_known = (6371 * 2 * np.arcsin(np.sqrt(np.sin((sla - la) / 2) ** 2 + np.cos(la) * np.cos(sla) * np.sin((slo - lo) / 2) ** 2))).min() < 50
    if prediction != "Neither" and values["T2M"] < 0 and not near_known:   # cold climate: bare tundra resembles the polar Mars analogs
        cautions.append("Cold, high-latitude terrain: bare tundra often scores like the polar Mars analogs (Svalbard, "
                        "Axel Heiberg), even where no analog has been documented. Treat this result with care.")
    # colour vs real Moon / Mars samples: distance between the place's average colour shape and each PDS reference
    rg, gb = float(df["slope_rg"].mean()), float(df["slope_gb"].mean())
    samples = sorted(({"name": PDS_NAMES.get(k, k), "distance": round(float(np.hypot(rg - v[0], gb - v[1])), 2)}
                      for k, v in PDS_REF.items()), key=lambda d: d["distance"])
    return jsonify(lat=lat, lon=lon, probabilities=dict(zip(CLASSES, map(float, proba))), prediction=prediction,
                   confidence=conf, cautions=cautions,
                   reasons=reasons, conditions=conditions, similar={"site": twin.site, "label": twin.label},
                   samples=samples, sample_match=samples[0]["distance"] <= PDS_MATCH, land_pixels=int(len(df)))


def parse_box(args):
    # request args -> (south, west, rows, cols) on the 0.2° grid, covering the whole box, at most MAX_TILES a side
    s, w, n, e = (float(args[k]) for k in ("south", "west", "north", "east"))
    s, n = min(s, n), max(s, n)
    w, e = min(w, e), max(w, e)
    shift = 360 * math.floor((w + 180) / 360)                 # boxes drawn on a repeated world copy -> -180..180
    w, e = w - shift, e - shift

    def snap(lo, hi, lo_lim, hi_lim):
        # grow the box outwards to the 0.2° grid so the squares cover the whole selection; keep at most MAX_TILES
        a, b = math.floor(round(lo / TILE, 6)) * TILE, math.ceil(round(hi / TILE, 6)) * TILE
        k = max(1, round((b - a) / TILE))
        if k > MAX_TILES:                                     # too big: keep the centre
            k = MAX_TILES
            a = math.floor(round(((lo + hi) / 2 - k * TILE / 2) / TILE, 6)) * TILE
        a = min(max(a, lo_lim), hi_lim - k * TILE)
        return round(a, 2), k

    south, rows = snap(s, n, -82.0, 82.0)
    west, cols = snap(w, e, -180.0, 180.0)
    return south, west, rows, cols


def run_scan(south, west, rows, cols, progress=lambda stage, done, total: None):
    # Scan an area as a grid of 22 km squares. Images are downloaded in blocks of up to BLOCK x BLOCK squares
    # (so no single GIBS request is too big) and cut into squares; climate is downloaded once per NASA POWER
    # grid cell; rock type once per square (in parallel). progress(stage, done, total) reports how far it got.
    north, east = round(south + rows * TILE, 2), round(west + cols * TILE, 2)
    H, W = rows * PX, cols * PX
    colour, relief = np.zeros((H, W, 4), np.float32), np.zeros((H, W), np.float32)
    lst, ndvi = np.full((H, W), np.nan, np.float32), np.zeros((H, W), np.float32)
    blocks = [(i, j) for i in range(0, rows, BLOCK) for j in range(0, cols, BLOCK)]
    for b, (i, j) in enumerate(blocks):                                 # i = square row from the north
        progress("Downloading NASA images", b, len(blocks))
        r, c = min(BLOCK, rows - i), min(BLOCK, cols - j)
        bn, bw = round(north - i * TILE, 2), round(west + j * TILE, 2)
        layers = area_layers(round(bn - r * TILE, 2), bw, bn, round(bw + c * TILE, 2), c * PX, r * PX)
        sl = (slice(i * PX, (i + r) * PX), slice(j * PX, (j + c) * PX))
        for arr, part in zip((colour, relief, lst, ndvi), layers):
            arr[sl] = part
    squares = []
    for i in range(rows):
        for j in range(cols):
            sl = (slice(i * PX, (i + 1) * PX), slice(j * PX, (j + 1) * PX))
            cc = colour[sl]; r, g, b = cc[..., 0], cc[..., 1], cc[..., 2]
            land = (cc[..., 3] > 0) & ~((b > r) & (b > g) & ((r + g + b) / 3 < 0.35))
            if land.sum() >= 100:                                       # skip ocean / mostly-water squares
                squares.append((round(north - (i + 0.5) * TILE, 2), round(west + (j + 0.5) * TILE, 2), sl))
    cells = sorted({power_cell(la, lo) for la, lo, _ in squares})
    clim, rocks, done = {}, {}, 0
    with ThreadPoolExecutor(12) as ex:
        jobs = {ex.submit(power_climate, *cell): ("clim", cell) for cell in cells}
        jobs.update({ex.submit(rock_type, la, lo): ("rock", (la, lo)) for la, lo, _ in squares})
        for f in as_completed(jobs):
            kind, key = jobs[f]
            (clim if kind == "clim" else rocks)[key] = f.result()
            done += 1
            progress("Downloading climate and rock type", done, len(jobs))
    out = []
    for k, (lat, lon, sl) in enumerate(squares):
        progress("Scoring squares", k, len(squares))
        df = features_from(colour[sl], relief[sl], lst[sl], ndvi[sl],
                           lambda: clim[power_cell(lat, lon)], lambda: rocks[(lat, lon)])
        if len(df) < 100:
            continue
        proba = model.predict_proba(df[FEATURES]).mean(0)
        verdict, conf = judge(proba)
        out.append(dict(lat=lat, lon=lon, prediction=verdict, confidence=conf,
                        probabilities=dict(zip(CLASSES, map(float, proba)))))
    return dict(bounds=[south, west, north, east], tile=TILE, squares=out, skipped=rows * cols - len(out))


@app.get("/api/scan")
def scan():
    # synchronous scan (for direct API use); the website uses /api/scan/start and /api/scan/status for progress
    try:
        box = parse_box(request.args)
    except (KeyError, ValueError):
        return jsonify(error="south, west, north and east are required numbers"), 400
    try:
        return jsonify(run_scan(*box))
    except Exception as err:                                            # a data service failed
        return jsonify(error=f"Could not download the data for this area ({err}). Try again."), 502


SCANS = {}          # job id -> progress and result of background scans (kept for an hour)


@app.get("/api/scan/start")
def scan_start():
    try:
        box = parse_box(request.args)
    except (KeyError, ValueError):
        return jsonify(error="south, west, north and east are required numbers"), 400
    for k in [k for k, v in SCANS.items() if time.time() - v["started"] > 3600]:
        SCANS.pop(k, None)
    job = uuid.uuid4().hex[:12]
    SCANS[job] = state = dict(started=time.time(), stage="Starting", done=0, total=1, result=None, error=None)

    def progress(stage, done, total):
        state.update(stage=stage, done=done, total=max(total, 1))

    def work():
        try:
            state["result"] = run_scan(*box, progress=progress)
        except Exception as err:                                        # a data service failed
            state["error"] = f"Could not download the data for this area ({err}). Try again."

    threading.Thread(target=work, daemon=True).start()
    return jsonify(job=job, squares_total=box[2] * box[3])


@app.get("/api/scan/status/<job>")
def scan_status(job):
    st = SCANS.get(job)
    if st is None:
        return jsonify(error="Unknown or expired scan."), 404
    if st["error"]:
        return jsonify(error=st["error"]), 502
    if st["result"] is not None:
        return jsonify(status="done", elapsed=round(time.time() - st["started"]), **st["result"])
    return jsonify(status="running", stage=st["stage"], done=st["done"], total=st["total"],
                   elapsed=round(time.time() - st["started"]))


if __name__ == "__main__":
    print("TerraX -> http://localhost:5000")
    app.run(port=5000, debug=False, threaded=True)
