<img width="1146" height="858" alt="image" src="https://github.com/user-attachments/assets/fd13bcdf-ea25-4038-badf-daff0951e5a3" />



We present SNAPSHOT a machine learning (XGBoost) model that takes next generation play tracking data and predicts player movements across both teams from their pre-snap positions. We further predict the yards gained from a play based on the position of the furthest forward outfield player in our simulation, assuming a successful catch. We use this proxy due to incomplete tracking data. An interactive webpage allows you to visualise the actual and predicted play, making it clear whether or not the model was trained on the play in question. This could be useful for broadcasters, as well as offensive and defensive coordinators looking to make last minute tactical adjustments to give them a competitive edge.

We were required to use the Kiro IDE as part of the competition, so agentic AI was used in code generation and instruction generation.

## Play Visualiser (web app)

An interactive Dash web app that animates NFL Big Data Bowl player tracking
data frame by frame, with an optional overlay of the trained model's predicted
play.

### Dependencies

The web app needs:

| Package        | Used for                                        |
| -------------- | ----------------------------------------------- |
| `dash`         | the web app framework and server                |
| `plotly`       | the animated field / player figure              |
| `pandas`       | reading and shaping the tracking + play CSVs    |
| `numpy`        | vectorised geometry and kinematics              |
| `xgboost`      | the trajectory model behind "Predicted" mode    |
| `scikit-learn` | the grouped train/test split used by the model  |

`dash`, `plotly`, `pandas` and `numpy` are enough to run the app in **Actual**
mode. `xgboost` and `scikit-learn` are only needed for the **Predicted** and
**Both** view modes; without them the app still starts and the prediction
toggle is simply disabled.

#### Install with conda (recommended)

```bash
conda create -n NFL_env -c conda-forge python=3.11 \
    dash plotly pandas numpy xgboost scikit-learn
conda activate NFL_env
```

#### Or with pip

```bash
pip install dash plotly pandas numpy xgboost scikit-learn
```

If a required package is missing, `visualiser.py` exits with a message telling
you exactly which packages to install.

### Data layout

The app expects the Big Data Bowl CSVs under `data/` in the project root:

```
data/
├── games.csv
├── players.csv
├── plays.csv
└── tracking/
    ├── tracking_2021090900.csv
    └── ...                       # one file per game
```

"Predicted" mode additionally reads the trained models from `model_output/`
(`model_dx.json`, `model_dy.json`, `metrics.json`). These are produced by
`predictor.py`; if they are absent the prediction toggle stays disabled.

### Running the app

From the project root, with the environment active:

```bash
python visualiser.py
```

The server starts on **http://127.0.0.1:8050** — open that URL in your browser.
Pick a game and a play, press ▶ Play to animate, and use the **View** control to
switch between the actual tracking and the model's predicted play.

> If you installed into a conda env named `NFL_env`, you can run it in one line
> without activating first:
>
> ```bash
> conda run -n NFL_env python visualiser.py
> ```
