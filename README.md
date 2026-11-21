# A Hybrid Recommender on a User–Film–Content Graph

AIM608 *Networks and Semantics I* — End-term **Project 4** · MovieLens *latest-small*

**Team:** Shikhar Mutta (MT2025114) · Hasini Yamsani (BT2024133) · D. Sreeteja (BT2024160)

Users, films and content live in **one graph**: ratings link users to films, and sentence-embedding similarity links films to films.
A *random walk with restart* (personalised PageRank) on this graph reaches films that similar users liked **and** films similar to those.
We compare item-based collaborative filtering, random walk with restart, a content model, the graph hybrid (with a tunable weight
λ of content edges), a popularity-discounted walk, a late-fusion baseline and node2vec/DeepWalk — and measure accuracy, cold start,
popularity bias, communities of taste and the best content weight.

## Main results (mean over 3 random splits, 20 % of each user's ratings hidden)

| Method | precision@10 | recall@10 |
|---|---|---|
| Popularity baseline | 0.119 | 0.097 |
| Item-based CF | 0.174 | 0.169 |
| RWR (ratings only) | 0.174 | 0.169 |
| **Hybrid RWR (λ = 0.1)** | **0.184** | **0.179** |
| RWR + popularity discount | 0.197 | 0.189 |

* Warm accuracy peaks at λ ≈ 0.1 in every split and collapses when content dominates (λ ≥ 0.3).
* **Cold start** (all ratings of 10 % of films hidden): hybrid recall@10 **0.088 vs 0.026 chance** (3.4×); collaborative methods are at chance.
* Accurate methods recommend almost only popular films (0–3 % outside the top-10 % most-rated); a degree discount improves accuracy.
* Six taste communities (modularity 0.41) differ strongly in genre mix (permutation p ≈ 5·10⁻⁴).

Full discussion: [`report/report.pdf`](report/report.pdf).

---

## Project structure

```
.
├── README.md                    this file
├── requirements.txt             Python dependencies
├── run.sh                       one-line commands (test / notebook / report / web / api)
├── hybrid_recommender.ipynb     MAIN notebook, executed: every number and figure
├── step1_collaborative_filtering.ipynb   first partial notebook (point 1 only)
├── src/
│   ├── hybrid_rec.py            all models + evaluation (item-CF, RWR, content graph, hybrid, node2vec, ...)
│   └── run_all.py               runs the 3 random splits in parallel (cached in results/seed*.pkl)
├── build/
│   ├── make_full_nb.py          writes the notebook        ┐ regenerate / execute
│   ├── exec_nb.py               executes the notebook      ┘ the notebook
│   └── export_web_data.py       exports JSON for the web app
├── tests/test_project.py        6 sanity tests (no leakage, stochastic matrix, RWR vs exact solve, cold start ...)
├── server/api.py                live API (stdlib only) for the web app
├── web/                         React (Vite) web interface
├── report/                      LaTeX report (report.tex, report.pdf) + figs/ + tables/ written by the notebook
├── results/                     cached results (seed*.pkl) and film embeddings (emb.npy)
└── data/ml-latest-small/        MovieLens dataset (downloaded)
```

---

## Step-by-step: set the project up on a new PC

### 0. Prerequisites

| Tool | Needed for | Check |
|---|---|---|
| Python **3.10 +** (developed on 3.14) | everything | `python3 --version` |
| Node.js **18 +** and npm | web interface only | `node -v` |
| LaTeX (`pdflatex`, `latexmk`) | rebuilding the report only | `latexmk -v` |
| ~3 GB disk, internet | PyTorch, sentence-transformer model (~90 MB), dataset (~1 MB) | |

### 1. Get the code

Copy / clone the project folder and open a terminal **in the project root** (the folder that contains `run.sh`).
All commands below are run from there.

### 2. Create a virtual environment and install Python packages

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(CPU-only PyTorch is enough: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then the rest.)

### 3. Download the data

`run.sh` does this automatically. To do it by hand:
```bash
mkdir -p data
curl -L -o data/ml.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -o data/ml.zip -d data          # creates data/ml-latest-small/
```
(Windows: download the zip from the URL above and extract it into `data/` so that `data/ml-latest-small/ratings.csv` exists.)

### 4. Check that everything works

```bash
./run.sh test
```
Expected: six lines starting with `PASS` (needs `results/emb.npy`; if it is missing, run step 5 first — it creates it).

### 5. Run the experiments and create the notebook outputs

```bash
./run.sh notebook
```
* If `results/seed*.pkl` exist (they are shipped with the project) this takes **≈ 2–3 minutes**.
* From scratch (`rm results/seed*.pkl results/emb.npy`) it takes **≈ 10–15 minutes** (embedding the films ≈ 1 min, 3 parallel splits ≈ 9 min).
* Output: executed `hybrid_recommender.ipynb`, figures in `report/figs/`, tables in `report/tables/`.

Open the notebook in VS Code / Jupyter (`jupyter lab`) to read the results.

### 6. Build the LaTeX report (optional)

```bash
./run.sh report            # -> report/report.pdf
```
Manual: `cd report && latexmk -pdf report.tex`. If `latexmk` complains about a corrupt `.aux`, delete `report/*.aux` and retry.

### 7. Start the web interface (optional)

Two terminals, both in the project root:
```bash
./run.sh api       # terminal 1 — live API on http://localhost:8000
./run.sh web       # terminal 2 — UI on  http://localhost:5173   (first run does `npm install`)
```
Open **http://localhost:5173/**. Without the API everything works except the live λ slider and the new-film demo.
If the UI says it cannot load data: `./run.sh web-data` (re-exports `web/public/data/*.json`).

### Windows without bash

`run.sh` is a bash script (use Git Bash or WSL). The equivalent plain commands:

| `run.sh` | command |
|---|---|
| `test` | `python tests/test_project.py` |
| `notebook` | `python build/make_full_nb.py` then `python build/exec_nb.py` |
| `report` | `cd report && latexmk -pdf report.tex` |
| `web-data` | `python build/export_web_data.py` |
| `api` | `python server/api.py` |
| `web` | `cd web && npm install && npm run dev` |

---

## `run.sh` cheat-sheet

| Command | What it does |
|---|---|
| `./run.sh test` | 6 sanity tests |
| `./run.sh notebook` | regenerate + execute the notebook (figures/tables go to `report/`) |
| `./run.sh report` | build `report/report.pdf` |
| `./run.sh all` | test → notebook → report |
| `./run.sh web-data` | re-export JSON for the web app |
| `./run.sh api` | live API on port 8000 |
| `./run.sh web` | React dev server on port 5173 |

## The web interface

| Tab | Shows |
|---|---|
| **Demo** | the guided demo: 1 pick a user → 2 every method side by side → 3 add a film with no ratings and see which users the hybrid recommends it to |
| Overview | headline numbers |
| Recommendations | any of the 610 users: favourites + top-10 of six methods side by side, live walk with λ / c / β sliders, graph explorer |
| New film (cold start) | add a film nobody rated (presets or your own title/genres/tags) and see which users it is recommended to |
| Accuracy & λ | accuracy table, effect of the content weight λ, robustness to k and c |
| Cold start & popularity | cold-start results, popularity and coverage |
| Communities | genre heatmap of the taste communities |
| Report & figures | the PDF report and all figures |

## Method in one paragraph

A film node follows a **content edge** with probability λ and a **rating edge** with probability 1 − λ; a film with no ratings
always follows content edges. The walk restarts at the user with probability *c* (power iteration, all users at once). Content
edges join each film to its 20 nearest films in `all-MiniLM-L6-v2` embedding space (title + genres + tags). Ratings are split
**before** any edge is built (20 % of each user's ratings hidden; a further 15 % of training for validation, where *c*, λ, β and
the fusion weight are tuned), and all numbers are averaged over 3 random splits.

## Reproducibility notes

* Fixed seeds (0, 1, 2); results cached in `results/seed*.pkl`. Delete them to recompute.
* The RWR implementation is verified three ways (power iteration, exact linear solve, Monte-Carlo walk) in the notebook (§2) and in `tests/`.
* node2vec is our own PyTorch implementation with first-order walks (p = q = 1, i.e. DeepWalk).
* Only *latest-small* is used; the 25M tag genome was not used.

## Declarations and citations

**GenAI:** Claude Code (Anthropic, Claude Sonnet 5.5) was used to help write the code, the notebook, the web interface and the report;
the design, experiments and interpretation were reviewed by the authors.

**References:** Linden, Smith & York (2003) item-to-item CF · Fouss, Pirotte, Renders & Saerens (2007) random-walk similarities ·
Harper & Konstan (2015) MovieLens · Page et al. (1999) PageRank · Reimers & Gurevych (2019) Sentence-BERT · Grover & Leskovec (2016) node2vec ·
Perozzi et al. (2014) DeepWalk · Blondel et al. (2008) Louvain · Hagberg et al. (2008) NetworkX · Paszke et al. (2019) PyTorch.
