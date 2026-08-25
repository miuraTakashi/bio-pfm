# BPFM status workflow

Statuses in `../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx`:

| Status | Meaning | In Lab atlas | In bio-pfm |
|--------|---------|--------------|------------|
| `draft` | Incomplete or auxiliary | yes | no |
| `lab` | Maintained lab model | yes | no |
| `published` | Cleared for public BPFM | yes | yes |

Promotion checklist before setting `published`:

1. `{model_id}.py` and `{model_id}.md` are self-contained enough to run outside the Lab tree (or list explicit package dependencies).
2. License / citation of the original model are documented.
3. No third-party PDF or unpublished data is required to run the demo.
4. Default run finishes in reasonable time (or documents a `QUICK` / smoke mode).
5. Row Status in the Excel list is updated to `published`, then the model is copied or synced into `bio-pfm/models/`.
