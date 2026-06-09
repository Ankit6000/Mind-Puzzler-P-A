# 4x4 Outside-Slot Puzzle Solver

This solves the puzzle where `00` is an outside blank slot connected to the
bottom-right tile position.

Solved target:

```text
11 12 13 14
21 22 23 24
31 32 33 34
41 42 43 44
           00  outside
```

Run the prompt example:

```powershell
python .\puzzle_solver.py --show-states
```

Solve your own state. Give 17 values: the 16 grid values row by row, then the
outside-slot value last.

```powershell
python .\puzzle_solver.py --state "11 12 13 14 21 22 23 24 31 32 33 34 41 42 00 43 44" --show-states
```

Generate and solve a legal random scramble:

```powershell
python .\puzzle_solver.py --scramble 40 --seed 7
```

Choose a search algorithm:

```powershell
python .\puzzle_solver.py --scramble 40 --seed 7 --algorithm auto
python .\puzzle_solver.py --scramble 40 --seed 7 --algorithm fast
python .\puzzle_solver.py --scramble 40 --seed 7 --algorithm ida-star
python .\puzzle_solver.py --scramble 40 --seed 7 --algorithm a-star-closed
```

Some arrangements are not solvable. The program checks that and prints an error
instead of searching forever.

Algorithm notes:

- `Fast solver` is the web default. It may give a longer route, but it usually
  finds usable legal moves much sooner.
- `Auto + exact fallback` tries the fast search, then exact searches. Use it
  when you are okay waiting longer.
- `IDA*` is exact and uses very little memory, but hard or misdetected boards
  can take much longer.
- `A* (closed set)` often searches fewer states, but it keeps many states in
  memory, so Python overhead can make it slower.
- `BFS` is only practical for tiny/simple scrambles.

## Camera App

The app opens your browser camera, or accepts an uploaded photo, detects the 16
grid tiles from the reference images in `F:\puzzle correct`, lets you fix any
label, lets you pick the solve algorithm, and then shows the solving moves with
directions like `44 down` or `23 left`.

Run it:

```powershell
python .\app.py
```

Then open:

```text
http://127.0.0.1:5000
```

If your reference images move, set `PUZZLE_REFERENCE_DIR` first:

```powershell
$env:PUZZLE_REFERENCE_DIR = "F:\puzzle correct"
python .\app.py
```

Detection also checks for a mostly white empty grid cell and marks it as `00`.
For best detection from the camera, fill the square camera frame with the 4x4
puzzle grid. For photos like a full board shot with the plastic frame visible,
use Upload, then Detect. If the guess is wrong, use the dropdown on any cell
before pressing Solve.

## PWA Install

The browser app is now installable as a PWA. Start the Flask app, then open the
site in Chrome or Edge and use the browser's install option.

For a connected Android phone, keep the phone on USB and run:

```powershell
adb reverse tcp:5000 tcp:5000
python .\app.py
```

Then open this on the phone:

```text
http://127.0.0.1:5000
```

Using `127.0.0.1` through `adb reverse` lets the phone browser use the camera
without needing HTTPS. The app can still use Upload if you open it from a normal
LAN address.

## Deploy

The deploy-ready files are:

- `requirements.txt` for Python dependencies.
- `render.yaml` for Render hosting.
- `reference/` with the 16 correct tile images bundled into the project.
- `index.html`, `static/puzzle-core.js`, and `sw.js` for GitHub Pages static
  hosting.

### GitHub Pages

GitHub Pages cannot run Flask/Python, so the root `index.html` uses the
browser-only JavaScript detector and solver from `static/puzzle-core.js`.

After pushing to `Ankit6000/Mind-Puzzler-P-A`, the included GitHub Actions
workflow publishes the static PWA. The Pages URL will be:

```text
https://ankit6000.github.io/Mind-Puzzler-P-A/
```

In the repo settings, set Pages source to GitHub Actions if GitHub asks for it.

### Render

Fastest path with Render:

1. Put this folder in a GitHub repo.
2. Go to Render, create a new Web Service, and connect that repo.
3. Render can use `render.yaml` automatically. If entering values manually, use:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT app:app
```

Set this environment variable if Render does not pick it from `render.yaml`:

```text
PUZZLE_REFERENCE_DIR=reference
```

After deploy, open the Render URL in Chrome or Edge and install it as a PWA. The
camera works on deployed HTTPS URLs; local network HTTP URLs may block camera
access in some browsers.

## Train Better Detection

The GitHub Pages app can use a trained browser model from
`static/trained-model.json`. The current file is seeded from the clean reference
tiles. To improve it, add real camera photos.

1. Put puzzle photos here:

```text
training/photos/
```

2. Copy the example labels file:

```powershell
Copy-Item .\training\labels.example.csv .\training\labels.csv
```

3. For each photo, add one CSV row with the filename and the 16 visible grid
labels, row by row. Use `00` for the empty square.

```csv
filename,cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8,cell9,cell10,cell11,cell12,cell13,cell14,cell15,cell16
my_photo.jpg,11,12,13,14,21,00,23,24,31,22,33,34,41,32,42,43
```

4. Extract tile crops:

```powershell
python .\tools\extract_training_samples.py
```

5. Train the browser model:

```powershell
python .\tools\train_detector.py
```

6. Commit and push the updated model:

```powershell
git add static/trained-model.json
git commit -m "Train detector with real puzzle photos"
git push origin main
git push origin main:gh-pages
```

Raw photos and crop samples are ignored by git; only the compact model JSON is
published.
