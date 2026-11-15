"""TerraX - website.

Run:   python webapp/app.py      then open http://localhost:5000
Needs: data/analog_model.joblib, data/web_meta.json and data/web_sites.json, created by section 10 of model/main.ipynb.
"""
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from features import DATA, GIBS_WMS, _get, place_features

app = Flask(__name__, static_folder="static")

model = joblib.load(DATA / "analog_model.joblib")
meta = json.loads((DATA / "web_meta.json").read_text(encoding="utf-8"))
FEATURES, CLASSES = meta["features"], meta["classes"]
NUMERIC = [f for f in FEATURES if f != "rock"]
MU, SD = pd.Series(meta["mu"]), pd.Series(meta["sd"])
ANALOGS = pd.DataFrame(meta["analogs"])
TYPICAL = meta["typical"]

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
        "slope_gb": "pixel colour (green/blue)", "brightness": "brightness", "b": "blue reflectance"}
# rows of the "Conditions" table; must match SHOWN in section 10 of model/main.ipynb
SHOW = [("rock", "Rock type", ""), ("RH2M", "Humidity", " %"),
        ("temp_range", "Air-temperature range", " °C"), ("lst_season", "Seasonal ground swing", " K"),
        ("patch_roughness", "Terrain roughness", ""), ("patch_slope_gb", "Area colour (G/B)", "")]


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
    contrib = model.predict(X, pred_contrib=True).reshape(len(X), len(CLASSES), len(FEATURES) + 1)
    c = pd.Series(contrib[:, best, :-1].mean(0), index=FEATURES).sort_values(ascending=False)

    values = {f: float(df[f].median()) for f in NUMERIC}
    values["rock"] = str(df["rock"].iloc[0])
    reasons = [f"{NICE[f]}: {fmt(f, values[f])}" for f in c.index[:3] if c[f] > 0]

    z = ((pd.Series(values)[NUMERIC].astype(float) - MU) / SD).fillna(0)
    dist = (((ANALOGS[NUMERIC] - MU) / SD).fillna(0) - z).pow(2).sum(axis=1)
    twin = ANALOGS.loc[dist.idxmin()]

    conditions = [{"name": name, "here": fmt(f, values[f]) + unit,
                   "moon": fmt(f, TYPICAL["Moon"][f]) + unit, "mars": fmt(f, TYPICAL["Mars"][f]) + unit}
                  for f, name, unit in SHOW]
    return jsonify(lat=lat, lon=lon, probabilities=dict(zip(CLASSES, map(float, proba))), prediction=CLASSES[best],
                   reasons=reasons, conditions=conditions, similar={"site": twin.site, "label": twin.label},
                   land_pixels=int(len(df)))


if __name__ == "__main__":
    print("TerraX -> http://localhost:5000")
    app.run(port=5000, debug=False, threaded=True)
