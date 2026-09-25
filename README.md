# Satellite Change Map

A desktop app that maps where a city changed between two periods, using
Sentinel-1 radar images from Google Earth Engine. Kyiv is built in. Pixels
whose radar backscatter changed more than speckle noise can explain are shown
over satellite imagery: red for a decrease, cyan for an increase.

Method: [Detecting Changes in Sentinel-1 Imagery (Part 1)](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1),
with the extensions described under "The math" below.

## Start the app

You need **Python 3.10+** ([python.org](https://www.python.org/downloads/))
and a **Google Cloud project registered for Earth Engine**
([register here](https://code.earthengine.google.com/register); free for
noncommercial use).

Double-click the launcher for your system:

| System | Launcher |
| --- | --- |
| Windows | `Start Change Map.bat` |
| macOS | `Start Change Map.command` (first time: right-click → Open) |
| Linux | `start_change_map.sh` (or `python3 launch.py`) |

The first start takes a minute or two while it sets up a private Python
environment in `.venv`. After that it starts in seconds. The app opens in your
browser; close the launcher window to stop it.

In the app:

1. The first time, click **Sign in with Google** and approve Earth Engine access.
2. Enter your Cloud project in the sidebar.
3. Press **Run analysis**. The defaults compare April–May 2021 with April–May
   2022 over Kyiv and Irpin, Bucha and Hostomel, the same season a year apart.
   The first run takes a few minutes. Results are saved in `cache/`, and the
   last one reopens instantly next time.
4. Use the **α slider** to trade sensitivity against false alarms. It updates
   instantly without contacting Earth Engine.

Other areas: choose **Custom area…** and enter a bounding box, or add a preset
to `CITIES` in `satchange/engine.py`.

## Reading the map

- **Expected by chance** shows how much area α alone would flag with no real
  change. Flagged area well above that is real change. Isolated single pixels
  scattered evenly are mostly those chance hits; look for clusters.
- **No-change check** runs the same test on two halves of the *before* period,
  where nothing should change. If it reads close to α, the statistics are
  calibrated for this area and these images.
- A flagged pixel means the radar signal changed, **not necessarily damage**.
  Construction, demolition, vegetation, soil moisture, flooding, wind on water,
  vehicles and crops all change backscatter.
- Destroyed buildings often show a **decrease**: the wall-ground
  "double bounce" disappears. Rubble, debris and new structures can show an
  **increase**.

## The math

A Sentinel-1 intensity pixel with mean `a` is gamma-distributed with `L` looks
(speckle). For each pixel and band (VV, VH), the likelihood ratio test for
"same mean before and after" gives `-2 log Q`, which is approximately χ² with
1 degree of freedom. The bands are summed, giving χ² with 2 dof, and a pixel is
flagged when the sum exceeds the χ² critical value at α. With one image per
period this is exactly the tutorial's bivariate test.

Additions, each checked by the tests:

- **Averaging over time.** Each period averages all acquisitions from one
  relative orbit, so the viewing geometry is identical. `n` images give `n × L`
  looks, counted per pixel. In simulation, a single image pair at α = 0.01
  catches a 3 dB change only about 8% of the time; 4 images per period catch
  about 50%.
- **Bartlett correction.** At about 4 looks, the plain χ² approximation flags
  about 25% more unchanged pixels than α. Dividing by Bartlett's correction
  factor makes the false-alarm rate match α.
- **Full resolution, independent of zoom.** Earth Engine map tiles are
  computed from averaged pixels when zoomed out, which silently changes the
  number of looks. The app computes the statistic at 10 m for every pixel,
  downloads it on a fixed Web Mercator grid, and thresholds it locally. The map,
  the areas and the α slider all use the same exact pixels.
- **Speckle level estimated from the data.** Earth Engine's terrain correction
  resamples pixels, so the looks per scene can differ from ESA's nominal 4.4.
  The app estimates it from consecutive before images: the IQR of their log
  ratio follows the log of an F(2L, 2L) distribution. Real change can only bias
  the estimate low, which makes the test stricter. The value is shown under
  **Details**.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest
```

- `satchange/stats.py`: the test statistic, thresholds and ENL estimator. The
  formula is one expression string that both Earth Engine and the NumPy tests
  evaluate.
- `satchange/engine.py`: Earth Engine search, orbit choice, compositing,
  the 10 m tiled download, the no-change check, and the cache.
- `satchange/render.py`: areas and the Leaflet map.
- `app.py`: the Streamlit UI. `launch.py`: the launcher.
- `tests/`: Monte Carlo checks of the statistics, the whole Earth Engine
  pipeline against a fake server (graph built against the real API
  signatures), and UI tests.
