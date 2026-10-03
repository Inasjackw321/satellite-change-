# Satellite Change Map

A desktop app for finding things on the ground with free Sentinel-1 radar
images: **activity at a site** (aircraft at an airport, vehicles at a base),
**ships** crossing an area, or **what changed** between two dates. Radar sees
through cloud, day and night. **No account, sign-in or cloud project needed.**

## Start the app

You need **Python 3.10+** ([python.org](https://www.python.org/downloads/)) and
an internet connection. Double-click the launcher for your system:

| System | Launcher |
| --- | --- |
| Windows | `Start Change Map.bat` |
| macOS | `Start Change Map.command` (first time: right-click → Open) |
| Linux | `start_change_map.sh` (or `python3 launch.py`) |

The first start takes a minute or two while it sets up a private Python
environment in `.venv`. After that it starts in seconds. The app opens in your
browser; close the launcher window to stop it.

## Three ways to use it

First pick **what you want to find** in the sidebar:

| Mode | Good for | What you get |
| --- | --- | --- |
| ✈️ **Activity at a site** | airports, bases, ports, car parks | every spot where objects came and went, labelled **"3/12"** = something there in 3 of 12 passes; objects per pass over time |
| 🚢 **Ships** | sea, straits, rivers, harbours | a dot for every ship in every pass, with date, length and brightness; ships per pass over time |
| 🏚️ **Changes between two dates** | damage, demolition, construction, cleared land | what changed for good between a start and an end date |

Then:

1. **Area:** **drag a box on the map** (click the square button at the top
   left, then click and drag; the pencil button lets you drag its corners), or
   pick an example site, **search any place by name** (OpenStreetMap), or type
   coordinates.
2. **Dates:** a **start** and an **end** date. Activity and ships use *every*
   satellite pass between them (up to 32, the latest); the two-date mode uses
   the latest images up to each date.
3. Before anything is downloaded, the sidebar shows how many passes or images
   it found and roughly how much it will download. Press **Find activity** /
   **Find ships** / **Find changes**.

Results are saved: the last one reopens instantly next time, older ones are
under **Saved results**. Switch between **Choose area** and **Results** at the
top of the page.

### Activity and ships

- **Sensitivity** (Sensitive / Balanced / Strict) sets how bright and how big
  an object must be. Changing it updates everything instantly.
- The **timeline** shows objects (or ships) per pass. Choose **One pass** and
  slide through the dates to see what was there on each pass, with that pass's
  radar image underneath.
- Hover over a label or dot for dates, size and brightness.
- **Downloads:** a finished image with legend, mini timeline and watermark;
  the interactive map; every detection as CSV (date, position, length, size,
  brightness); for activity, every busy spot as CSV.

### Changes between two dates

- **Detection** (Sensitive / Balanced / Strict / Custom) sets what counts as a
  change. The **noise check** runs the same detection on before images compared
  with each other; close to 0 km² means what's shown is real change.
- The number next to each spot is how many times a change was seen there
  between one pass and the next; hover for the dates.
- Downloads: image (legend, watermark), interactive map, spot list (CSV).

The image download uses the satellite photo (Esri) or the radar image as
background and adds a legend, scale bar, north arrow and a watermark
(**@Kaldockhi** by default, editable).

### Where the images come from

[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
provides Sentinel-1 images free and anonymously, radiometrically terrain
corrected, as cloud-optimised GeoTIFFs. The app downloads only the pixels
covering your area. As a guide, an airport (~20 km²) over two months is about
50 MB. Images: contains modified Copernicus Sentinel data.

## How it works

Radar images are grainy ("speckle"): a pixel's brightness varies randomly from
pass to pass, following a gamma distribution whose spread is set by the number
of looks `L`. Everything here is a statistical test that keeps speckle from
being mistaken for real objects or change.

### Activity and ships: each pass against its usual state

- **Usual state, before and after.** For each pass and pixel, the app takes the
  median of up to 4 passes before it and, separately, of up to 4 passes after
  it, from the same satellite track. An object counts only if it is clearly
  brighter than **both**. Things that come and go (aircraft, vehicles, ships)
  pass this test. A lasting change (a building demolished or built) doesn't,
  and nor do slow trends such as crops growing, because the passes on one side
  already show them.
- **Exact thresholds.** The chance that speckle alone makes a pass brighter
  than the median of `n` other passes is computed exactly. The median of `n`
  gamma values is an order statistic, so this is a one-dimensional integral,
  accurate far into the tail, and the tests check it against simulation. Each
  pass's overall brightness (e.g. wetter after rain) is levelled first.
- **Objects.** Bright patches between a minimum and a maximum size: a whole
  field brightening after rain is not an object. Sizes and lengths are
  corrected for the blur of the noise filter.
- **Water** is found automatically: large areas that are usually dark. Narrow
  dark strips such as runways are excluded. Ships are only counted on water,
  activity only on land.
- **More passes.** By default, every satellite track that covers the area is
  used (for example morning and evening passes), each compared only with its
  own track.

### Changes between two dates

The method of [Detecting Changes in Sentinel-1 Imagery (Part 1)](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1):
a likelihood-ratio test per pixel for "same mean radar signal before and
after", with VV and VH summed (χ² with 2 dof). Additions:

- **Averaging** several images on each side, with **Bartlett's correction** so
  the false-alarm rate matches α.
- **Filters:** significance, a minimum change in dB, and a minimum patch size.
- **Noise check:** the same test and filters on before-vs-before images.

### Common to both

- **Noise reduction:** 3 × 3 pixel averaging, off for ships, which are found at
  full 10 m detail.
- **Speckle level measured from the data:** the spread of the log ratio of
  consecutive passes follows the log of an F(2L, 2L) distribution. It is
  measured in 9 windows and the lowest value is used, because textured ground
  (buildings, fields) has fewer effective looks than smooth ground (water,
  tarmac). The test is then calibrated for the busiest part of the scene and
  only stricter elsewhere.
- **One fixed 10 m grid** (Web Mercator, nearest-neighbour resampling, which
  keeps the speckle statistics), so the map, sizes and counts never depend on
  zoom.

### Limits

- Objects that are there in most passes become the "usual" state and are not
  counted (e.g. a ship moored for the whole period).
- At 10 m, small boats, cars and light aircraft may be missed; large aircraft,
  trucks and ships are found reliably.
- A detection means the radar signal changed, not what the object is.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest
```

- `satchange/stats.py`: the tests, thresholds (including the exact pass-vs-median
  threshold) and ENL estimator.
- `satchange/source.py`: Planetary Computer search, access tokens, and reading
  image windows.
- `satchange/timeseries.py`: activity and ships: planning, per-pass
  comparison, water, object and hotspot detection.
- `satchange/engine.py`: the two-date comparison, the grid, the cache and saved
  results.
- `satchange/render.py`: maps for both kinds of result.
- `satchange/export.py`: downloadable images (legend, timeline, watermark) and
  CSV files.
- `satchange/inputs.py`: place search, area fitting, drawn boxes.
- `app.py`: the Streamlit UI. `launch.py`: the launcher.
- `tests/`: Monte Carlo checks of the statistics, and the whole pipeline on a
  synthetic Sentinel-1 scene (real GeoTIFFs in UTM with known speckle). The
  scene contains a demolished patch, aircraft stands used on some passes, a
  patch that brightens over a large area, and a ship sailing across a strip of
  water. The tests check that each is found (or correctly ignored) at the
  right place, on the right dates and at the right size, and that false alarms
  stay at the advertised rate. There are also tests for the exported images
  and the UI.
