# 🌍 → 🌕 / 🔴 TerraX

**Click any place on Earth and find out whether it looks like the Moon, like Mars, like both, or neither.**

TerraX is a machine-learning model, built on open NASA data, that scores any coordinate on Earth as a **Moon analog**, a **Mars analog**, **both**, or **neither**, for example `Moon 96 % · Mars 2 % · Neither 2 %` for Meteor Crater. It comes with an interactive map website, so anyone can explore the planet and discover new candidate analog sites.

> ✅ **The model is already trained and included.** You don't need to train anything. The first thing to try is the website (see [Quick start](#-quick-start-try-the-website)).

---

## 💡 The idea

Before astronauts walk on the Moon or rovers drive on Mars, they train and test on Earth, at places that resemble those worlds:

- **Moon analogs:** Apollo astronauts trained at Meteor Crater (Arizona), the Cinder Lake crater field and the lava fields of Craters of the Moon (Idaho).
- **Mars analogs:** rovers and crewed simulations use the hyper-arid Atacama Desert (Chile), the Mars Desert Research Station (Utah) and the acidic Rio Tinto (Spain).

These sites were found by experts, one at a time. **TerraX learns what they have in common from NASA satellite data and searches the whole planet for more.** That can help to:

- find new, closer or cheaper training and test sites for Artemis and future Mars missions;
- show students and the public *why* a place looks like another world, with real data;
- give scientists a quick first screen before a field visit.

---

## 🚀 Quick start: try the website

The trained model is in `data/`, so the website works straight away.

**1. Install** (Python 3.10 or newer):

```bash
git clone https://github.com/MakariousR/TerraX
cd TerraX
pip install -r requirements.txt
```

**2. Run the website:**

```bash
python webapp/app.py
```

**3. Open <http://localhost:5000>** in your browser.

An internet connection is needed: for every new spot, the website downloads live data from NASA services.

### What to do on the website

- **Click anywhere on land.** After about 10–20 seconds (instant for places already seen), a card shows:
  - the **verdict** (Moon, Mars, Both or Neither) and the **probabilities** for Moon, Mars and Neither;
  - a **confidence level** (high, medium or low) showing how decisive the probabilities are, plus a caution for cases the model is known to struggle with;
  - **why**: the features that pushed the model towards its answer;
  - **conditions**: the place's values next to those of typical known Moon and Mars analogs;
  - the **most similar known analog site**;
  - **colour vs real samples**: how close the place's colour is to NASA PDS lab measurements of Apollo lunar soil, Martian dust simulant and Martian meteorites.
- **Search a place by name** (for example "Atacama Desert" or "Lanzarote").
- **Scan a whole area** to find new candidate analogs. Press **▦ Select an area to scan** and drag a box on the map. TerraX covers the box with a grid of 22 km squares, up to 2° × 2° (about 220 km, 100 squares) at a time, and colours each square by its verdict, with stronger colour for higher confidence. A progress bar shows each step while it works. A summary lists the strongest analog squares; click any square for its full card. A new area takes under a minute; an area already scanned takes a few seconds.
- **Click a coloured dot** to see how the model scores one of the 110 known training sites.
- **Switch the base map** between satellite imagery, NASA Blue Marble and OpenStreetMap.

**Places worth trying:** the Lut Desert in Iran, the Namib Desert, and your own city for comparison.

---

## 🛰️ How it works

```
labelled sites  →  NASA data for a ±11 km area  →  15 features per pixel  →  LightGBM  →  Moon / Mars / Neither
```

### 1. Labelled places

[`data/sites.csv`](data/sites.csv) lists **110 places**: 26 Moon analogs, 45 Mars analogs and 39 ordinary places (forests, cities, farmland, wetlands, and also dry steppe, so that "dry" alone isn't enough to count as an analog). The analog labels come from documented training and test sites, mainly the ESA / Open University *Catalogue of Planetary Analogues* (Preston et al. 2012) and the USGS record of Apollo astronaut field training (Open-File Report 2005-1190, Table 1), plus NASA, JPL, USGS, ESA and peer-reviewed sources for the remaining sites (see [Resources](#-resources)). Every place has its reference in the `source` column.

Only analogs that are **visible from orbit** are used. Some documented analogs, such as the Sudbury and Ries impact craters, are valued for their *rocks*, but at this resolution their surroundings are forest or farmland. Including them taught the model that green land can be an analog (an area scan of Arizona called a pine forest "Moon"), so they are left out.

### 2. NASA data

For each place, TerraX downloads a 48 × 48 pixel area (about ±11 km, around 460 m per pixel):

| Source | What it gives | Why it matters |
|---|---|---|
| **NASA GIBS: Blue Marble** | true surface colour | red iron oxide (Mars), dark basalt (Moon), green vegetation (neither) |
| **NASA GIBS: ASTER GDEM shaded relief** | terrain shape | craters, lava fields and canyons are rough |
| **NASA GIBS: MODIS land surface temperature** | ground temperature in January and July | bare, dry ground heats and cools strongly |
| **NASA GIBS: MODIS vegetation index (NDVI)** | how green the ground is in January and July | analogs are deserts, hot or polar, with almost no plants even in summer |
| **NASA POWER** | temperature, humidity, sunlight | analogs are dry, with harsh temperature swings |
| **NASA PDS: RELAB spectra** | lab spectra of Apollo soils, Mars dust simulant and Martian meteorites | the real colour of the Moon and Mars: a reference in the notebook's charts and on every website card. It's not a model input: tested as extra features, it didn't improve accuracy, because many Earth analogs are redder than real lunar soil |
| **Macrostrat geologic maps** | rock type | basalt and impact rock (Moon-like) vs sediments and salts (Mars-like) |

### 3. Features

Every pixel becomes one row with **15 features**: colour and colour shape (`b`, `brightness`, `slope_rg`, `slope_gb`), the whole area's roughness, colour patchiness and average colour, the climate (mean temperature, temperature range, humidity, sunlight), the seasonal ground-temperature swing, the summer greenness of the pixel and of the whole area (NDVI), and the rock type. Latitude and longitude are **not** features, so the model learns what a place is *like*, not where it is.

### 4. Model

A **LightGBM** classifier built in 200 rounds of tiny trees (one per class each round, 600 in total), each a single yes/no rule that must hold for about a dozen places. Each tree only sees 30 % of the features, so no single feature can dominate. For a new place, the probabilities of all its land pixels are averaged.

**Both Moon and Mars.** Some places resemble both worlds: the Catalogue of Planetary Analogues describes the Atacama Desert, Iceland and Meteor Crater as analogues for both. When the Moon and Mars probabilities are **each at least 30 % and each higher than Neither**, TerraX gives the verdict **Both** instead of the single highest class. The model still learns three classes; "Both" is read from its probabilities, so it doesn't change the accuracy below. On unseen places it's uncommon: about 8 of the 110 training places on average (4 to 10 depending on the split), and never for ordinary land.

### 5. Honest evaluation

Pixels of the same place are near-copies, so testing on random pixels would leak answers. TerraX tests on **unseen places** instead: places closer than 150 km are grouped, and whole groups are held out in 5-fold cross-validation.

| | Result on unseen places |
|---|---|
| Place-level accuracy | **about 89 %** (110 places) |
| Log-loss (probability quality, lower is better) | **0.32** |
| F1: Mars / Neither / Moon | 0.88 / 0.90 / 0.90 |

The strongest signals are the summer greenness of the area, humidity, the rock type, the temperature range and the area's overall colour. Moon vs Mars is the hardest distinction, because several real sites serve as analogs for both.

---

## 📁 Project structure

```
TerraX/
├── data/
│   ├── sites.csv              # the 110 labelled places, with sources
│   ├── analog_model.joblib    # the trained model (ready to use)
│   ├── web_meta.json          # feature list and reference values for the website
│   ├── web_sites.json         # known sites shown on the map
│   └── cache/                 # downloaded NASA data (created automatically, not in git)
├── model/
│   └── main.ipynb             # the full pipeline: data, features, training, evaluation, export
├── webapp/
│   ├── app.py                 # Flask server and prediction API
│   ├── features.py            # feature extraction for one place (same as the notebook)
│   └── static/index.html      # interactive map (Leaflet)
├── requirements.txt
└── README.md
```

---

## 🔬 Retrain or explore the model (optional)

Everything is in [`model/main.ipynb`](model/main.ipynb), explained step by step. It covers the labelled places, the NASA data with plots, features, model, evaluation, feature importance, predictions for new places, and the export for the website.

```bash
cd model
jupyter notebook main.ipynb      # or open it in VS Code
```

Run all cells. The first run downloads about 900 small files, 12 places at a time (a few minutes); later runs use `data/cache/` and take seconds. The last section saves a new model into `data/`, and the website uses it the next time it starts.

**To add your own places,** add rows to `data/sites.csv` (`site, lat, lon, label, region, source`) and re-run the notebook. If you change a feature, change it in both `model/main.ipynb` and `webapp/features.py`.

### API

The website's server can also be used directly:

```
GET /api/predict?lat=35.03&lon=-111.02
```

It returns JSON with `probabilities`, `prediction`, `confidence`, `cautions`, `reasons`, `conditions`, the `similar` known site, and `samples` (the colour distance to each real Moon and Mars sample, closest first).

```
GET /api/scan?south=29.5&west=58&north=31.5&east=60
```

It scans the area as 22 km squares (at most 2° × 2°, 100 squares) and returns each square's `prediction`, `confidence` and `probabilities`. To stay fast, it downloads each NASA image layer in blocks of up to 10 × 10 squares and cuts them into squares, and fetches NASA POWER climate once per POWER grid cell, which nearby squares share. Each square gives the same result as clicking its centre.

The website runs scans in the background instead: `GET /api/scan/start?…` (same parameters) returns a `job` id, and `GET /api/scan/status/<job>` returns the current step and progress until the scan is done, then the same result as above.

---

## ⚠️ Limitations

- **110 labelled places is small.** More documented sites, especially Moon analogs, would help most.
- **Labels follow where agencies have trained,** so the model partly learns the typical colour of those regions (for example, Earth's Moon analogs are often red cinder fields).
- **Resolution is about 460 m per pixel,** so small features like a single crater rim are averaged out. For the same reason, analogs known only for their rocks, under forest or farmland, can't be recognised.
- **There is no feature for cities.** A city in a desert, such as Phoenix, scores like the desert around it.
- **In cold, high-latitude regions,** bare tundra often scores like the polar Mars analogs (Svalbard, Axel Heiberg), because these data can't tell documented and undocumented periglacial terrain apart. The website shows a caution for such places.
- **Above 82° latitude** there's no usable imagery, and oceans are rejected.
- An analog score means "looks similar in these data", not a scientific certification of a site.

---

## 📚 Resources

### NASA data and resources

- [NASA SSERVI Analog Testbeds](https://sservi.nasa.gov/testbed)
- [NASA FINESSE (Field Investigations to Enable Solar System Science and Exploration)](https://www.nasa.gov/finesse-home/)
- [NASA GIBS (Global Imagery Browse Services) API](https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api)
- [NASA GIBS WMS service (layer list)](https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0)
- [NASA Blue Marble Next Generation (surface colour)](https://visibleearth.nasa.gov/collection/1484/blue-marble)
- [NASA ASTER Global Digital Elevation Model (terrain relief)](https://lpdaac.usgs.gov/products/astgtmv003/)
- [NASA MODIS Land Surface Temperature monthly (MOD11C3)](https://lpdaac.usgs.gov/products/mod11c3v061/)
- [NASA GIBS MODIS Land Surface Temperature colormap](https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Land_Surface_Temp.xml)
- NASA GIBS MODIS vegetation index monthly layer (`MODIS_Terra_L3_NDVI_Monthly`, listed in the GIBS WMS layer list above)
- [NASA GIBS MODIS NDVI colormap](https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_NDVI.xml)
- [NASA POWER (Prediction Of Worldwide Energy Resources) climate data](https://power.larc.nasa.gov/)
- [NASA POWER climatology API](https://power.larc.nasa.gov/docs/services/api/temporal/climatology/)
- [NASA PDS Geosciences Node: Spectral Library](https://pds-geosciences.wustl.edu/spectrallibrary/default.htm)
- [NASA PDS RELAB spectra collection](https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/)
- [JPL: Mojave Desert Tests Prepare for NASA Mars Roving (Dumont Dunes)](https://www.jpl.nasa.gov/news/mojave-desert-tests-prepare-for-nasa-mars-roving/)
- [JPL: NASA Tests Drones in Death Valley, Preps for Martian Sands and Skies](https://www.jpl.nasa.gov/news/nasa-tests-drones-in-death-valley-preps-for-martian-sands-and-skies/)
- [NASA Earth Observatory: Mars on Earth (Wadi Rum)](https://science.nasa.gov/earth/earth-observatory/movie-mars-on-earth-148764/)

### Other resources

- [Catalogue of Planetary Analogues (ESA / Open University): record](https://oro.open.ac.uk/33373/)
- [Catalogue of Planetary Analogues (Preston et al. 2012): PDF](https://freight.cargo.site/m/D1821053255089660420449933921929/The-Catalogue-of-Planetary-Analogues.pdf)
- [USGS Open-File Report 2005-1190: Geologic Training of Apollo Astronauts](https://pubs.usgs.gov/of/2005/1190/)
- [USGS Open-File Report 2005-1190: Table 1 (Apollo field training sites)](https://pubs.usgs.gov/of/2005/1190/of2005-1190_table1.pdf)
- [USGS California Volcano Observatory: Can't make it to the Moon? Try California's volcanoes](https://www.usgs.gov/observatories/calvo/news/cant-make-it-moon-try-californias-volcanoes)
- [ESA: Turning astronauts into Moon explorers (PANGAEA)](https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Turning_astronauts_into_Moon_explorers)
- [ESA: PANGAEA 2019 dry run](https://www.esa.int/ESA_Multimedia/Videos/2019/08/Pangaea_2019_dry_run)
- [USGS Professional Paper 599-C: Structural Geology of the Henbury Meteorite Craters (Contributions to Astrogeology)](https://pubs.usgs.gov/publication/pp599C)
- [USGS: Oblique thermal imaging of the Pisgah lava field (NASA-funded Mars analog study)](https://www.usgs.gov/data/oblique-thermal-imaging-pisgah-lava-field-california-acquired-march-2010)
- [National Park Service: NASA in Death Valley](https://home.nps.gov/deva/nasa.htm)
- [Cavalazzi et al. 2019: The Dallol Geothermal Area, an exceptional planetary field analog on Earth (Astrobiology)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6459281)
- [Dalangtan saline playa (Qaidam Basin) as a Mars analog (PubMed)](https://www.pubmed.gov/29792755/)
- [Using an Australian Mars Analogue Research Facility for Astrobiology Education and Outreach (IAU Symposium)](https://cambridge.org/core/journals/symposium-international-astronomical-union/article/using-an-australian-mars-analogue-research-facility-for-astrobiology-education-and-outreach/AC7A2886632E2F728206A1CF51A356F3)
- [Mars Society Australia: Mars Australia Research Station (Arkaroola)](https://mars.marssociety.org.au/)
- [Mars Society Australia: Henbury Craters](https://marssociety.org.au/project/geology/place/henbury-craters)
- [Jordan Times: Space simulation in Wadi Rum](https://jordantimes.com/news/local/jordanian-palestinian-women-lead-historic-space-simulation-in-wadi-rum)
- [Wikidata (site coordinates)](https://www.wikidata.org/)
- [RELAB Spectral Database (Brown University)](https://sites.brown.edu/relab/relab-spectral-database/)
- [Macrostrat (geologic maps: rock type)](https://macrostrat.org/)
- [Macrostrat geologic units map API](https://macrostrat.org/api/v2/geologic_units/map)

### Software

Python · LightGBM · scikit-learn · pandas · NumPy · Flask · Leaflet · OpenStreetMap Nominatim (place search) · Esri World Imagery (base map)
