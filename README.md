# Satellite Change Map

A desktop app that maps where a city changed between two dates, using free
Sentinel-1 radar images. **No account, sign-in or cloud project needed.**
Kyiv is built in. Spots whose radar signal changed more than noise can explain
are shown over satellite imagery: red where the signal decreased, cyan where it
increased.

Method: [Detecting Changes in Sentinel-1 Imagery (Part 1)](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1),
with the extensions described under "The math" below.

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

In the sidebar:

1. **Area:** **drag a box on the map**: click the square button at the top
   left of the map, then click and drag. To adjust the box, click the pencil
   button, drag its corners or middle, and click **Save**. You can also pick a
   Kyiv area, **search for any place by name** (using OpenStreetMap's place
   search), or enter coordinates. Very small areas are enlarged to at least
   4 km across; very large ones are trimmed. The last box you drew is
   remembered.
2. **Dates:** pick a **start** and an **end** date, e.g. 4 July 2026 to
   4 August 2026. The app uses the latest images up to the start date as
   "before", and the latest images up to the end date (after the start) as
   "after".
3. **Images to average on each side:** the default is 3. More images remove
   more noise but mean a bigger download and reach further back from each
   date.

Before anything is downloaded, the sidebar lists the exact image dates it
found and roughly how much it will download. It warns if the nearest image is
more than two weeks before one of your dates. Press **Find changes**. Results
are saved on your computer: the last map reopens instantly next time, and
older ones are under **Saved maps**. Switch between **Choose area** and
**Change map** at the top of the page.

On the change map, **Detection** sets what counts as a change:

| Setting | Chance a pixel is noise | Smallest change | Smallest patch |
| --- | --- | --- | --- |
| Sensitive | 1 in 1,000 | 1.5 dB | 100 m² |
| **Balanced** (default) | 1 in 100,000 | 3 dB (signal halved or doubled) | 400 m² |
| Strict | 1 in 10,000,000 | 6 dB | 2,000 m² |
| Custom | your choice | your choice | your choice |

Changing it updates the map instantly.

### Where the images come from

[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
provides Sentinel-1 images free and anonymously, radiometrically terrain
corrected, as cloud-optimised GeoTIFFs. The app downloads only the pixels
covering your area. As a guide, the Irpin/Bucha/Hostomel area (~300 km²) with 3
images on each side is about 100 MB; the whole of Kyiv (~2,300 km²) is nearly
1 GB.
Images: contains modified Copernicus Sentinel data.

## Reading the map

- **Noise check** runs exactly the same detection on the before images
  compared with each other, where nothing should have changed. If it reads
  close to 0 km², what the map shows is real change. If it doesn't, choose
  **Strict** or raise the Custom limits.
- **Changed spots** counts separate patches of change.
- A flagged spot means the radar signal changed, **not necessarily damage**.
  Construction, demolition, vehicles and aircraft coming and going,
  vegetation, soil moisture, flooding, wind on water and crops all change it.
- Destroyed buildings often show a **decrease**: the wall-ground
  "double bounce" disappears. Rubble, debris and new structures can show an
  **increase**.
- The layer menu (top right of the map) shows the before and after radar
  images themselves.

## The math

A Sentinel-1 intensity pixel with mean `a` is gamma-distributed with `L` looks
(speckle). For each pixel and band (VV, VH), the likelihood ratio test for
"same mean before and after" gives `-2 log Q`, which is approximately χ² with
1 degree of freedom. The bands are summed, giving χ² with 2 dof, and a pixel is
flagged when the sum exceeds the χ² critical value at α. With one image per
side and no noise reduction this is exactly the tutorial's bivariate test.

Additions, each checked by the tests:

- **Averaging over time.** Each period averages several acquisitions from one
  relative orbit, so the viewing geometry is identical. `n` images give `n × L`
  looks, counted per pixel. In simulation, a single image pair at α = 0.01
  catches a 3 dB change only about 8% of the time; 4 images per period catch
  about 50%.
- **Bartlett correction.** At about 4 looks, the plain χ² approximation flags
  about 25% more unchanged pixels than α. Dividing by Bartlett's correction
  factor makes the false-alarm rate match α.
- **Noise reduction.** Each image is averaged over 3 × 3 pixels (Medium, the
  default) before testing, which removes most speckle. The speckle level (below)
  is measured on the filtered images, so the test stays calibrated.
- **Filters for real change.** A statistically significant pixel is only shown
  if the signal also changed by a meaningful amount (dB) and it is part of a
  patch of a minimum size. On the synthetic test scene, the old settings (10 m,
  α = 1%, no filters) flagged 1,719 spots; Balanced flags only the one real
  change, and the noise check is 0.
- **Fixed grid.** Every image is resampled with nearest neighbour (which keeps
  the speckle statistics) onto one fixed 10 m Web Mercator grid. The map and
  the areas use the same pixels at every zoom level.
- **Speckle level measured from the data.** Terrain correction resamples
  pixels, so the looks per scene can differ from ESA's nominal 4.4. The app
  measures it (after noise reduction) from consecutive before images: the spread (IQR) of their log
  ratio follows the log of an F(2L, 2L) distribution. Real change can only
  bias the estimate low, which makes the test stricter. The value is shown
  under **Details**.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest
```

- `satchange/stats.py`: the test statistic, thresholds and ENL estimator.
- `satchange/source.py`: Planetary Computer search, access tokens, and reading
  image windows.
- `satchange/engine.py`: orbit choice, download planning, the tiled
  computation, the no-change check, and the cache.
- `satchange/render.py`: the detection filters, areas and the Leaflet map.
- `satchange/inputs.py`: place search, area fitting, drawn boxes.
- `app.py`: the Streamlit UI. `launch.py`: the launcher.
- `tests/`: Monte Carlo checks of the statistics; the whole pipeline on
  synthetic Sentinel-1 GeoTIFFs with known speckle and a known changed patch
  (checks the dates chosen, detection and its location, change size, the
  false-alarm rate, ENL and the noise check);
  and UI tests.
