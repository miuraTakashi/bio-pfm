# Biological Pattern Formation Models (BPFM)

**BPFM** / package name **`bio-pfm`**

Published computational models for biological pattern formation.

## Layout (sibling of the Lab atlas)

```text
Projects/git/
  Pattern_Formation_Model_Atlas/   # Lab atlas (source of truth)
  bio-pfm/                         # this package
```

| Layer | Location | Role |
|-------|----------|------|
| Lab | `../Pattern_Formation_Model_Atlas/` | Full atlas, draft/lab/published |
| Public (BPFM) | this repo | Status = `published` only |

Status spreadsheet: [`../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx`](../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx)

## Contents

- Models exported: **45** (see [`CATALOG.md`](CATALOG.md))
- Code layout under `models/` mirrors Lab `python/` scheme folders
- Heavy outputs (`results/`, GIF/PDF/OBJ, …) are **not** shipped; regenerate locally
- Shared plotting: `models/atlas_plotting.py` (+ `models/lib/`)

## Setup

```bash
cd bio-pfm
python -m pip install -e .
# or for model script runs:
python -m pip install -r requirements.txt
```

## Run a model

```bash
cd models/1_continuous_scheme/1_1_diffusion_term/turing_2d
python turing_2d.py
```

## Re-export from Lab

```bash
python3 python/scripts/export_published_to_bio_pfm.py
```

## Status workflow

See [`STATUS.md`](STATUS.md).
