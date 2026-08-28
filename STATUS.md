# Maintainer notes (Lab atlas ↔ BPFM)

This file is for **curators** syncing the public collection from the Lab atlas.
End users can start from [`README.md`](README.md).

## Status values

Defined in [`../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx`](../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx):

| Status | Meaning | In Lab atlas | In bio-pfm |
|--------|---------|--------------|------------|
| `draft` | Incomplete or auxiliary | yes | no |
| `lab` | Maintained lab model | yes | no |
| `published` | Cleared for public BPFM | yes | yes |

## Promotion checklist (`published`)

1. Biological pattern-formation relevance is clear (Application / Section in the spreadsheet).
2. `{model_id}.py` and `{model_id}.md` run outside the Lab tree (or list dependencies).
3. Original model is citable; no bundled third-party PDF.
4. Default run is reasonable (or documents a `QUICK` / smoke mode).
5. Excel Status set to `published`, then re-export (below).

## Re-export from Lab

```bash
cd Pattern_Formation_Model_Atlas
python3 python/scripts/export_published_to_bio_pfm.py
```

Writes `models/`, [`README.md`](README.md), [`CATALOG.md`](CATALOG.md), and updates package metadata.
Heavy media under `results/` is omitted; run each model locally to regenerate figures.
