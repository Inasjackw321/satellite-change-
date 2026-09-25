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

The sidebar walks you through four steps:

1. **Google account:** click **Sign in with Google**. Google's sign-in page
   opens in your browser; approve access and the app updates by itself.
2. **Cloud project:** pick your project from the list. If it isn't listed,
   type its ID. There's a link to create a free Earth Engine project if you
   don't have one.
3. **Area:** Kyiv (city plus Irpin, Bucha and Hostomel) is built in. You can
   also **search for any place by name** (using OpenStreetMap's place search)
   or enter coordinates. Very small
   areas are enlarged to at least 4 km across, and very large ones are trimmed
   to fit a 10 m analysis.
4. **Dates:** pick the date the changes happened before. The app compares the
   weeks after that date with the same weeks one year earlier, so seasons don't
   show up as change. You can also choose both periods yourself.

Then press **Find changes**. The first run takes a few minutes. Results are
saved: the last map reopens instantly next time, and older ones are under
**Saved maps**. The **sensitivity (α) slider** updates the map instantly.

If something goes wrong (for example the project isn't registered for Earth
Engine, or the Earth Engine API is switched off), the app says what to do and
links to the right Google page.

**Signing out:** click **Sign out** under "Google account" and confirm. This
revokes the app's Google access token and deletes the sign-in from this
computer. Earth Engine keeps one sign-in per computer, so other Earth Engine
tools on this computer are signed out too. Your saved maps are kept; delete
them under **Saved maps**. To switch accounts, sign out and sign in again.

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
- `satchange/account.py`: sign-in, sign-out, project list, friendly errors.
- `satchange/inputs.py`: place search, area fitting, before/after periods.
- `app.py`: the Streamlit UI. `launch.py`: the launcher.
- `tests/`: Monte Carlo checks of the statistics, the whole Earth Engine
  pipeline against a fake server (graph built against the real API
  signatures), and UI tests.
