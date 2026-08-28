# Biological Pattern Formation Models (BPFM)

**BPFM** (`bio-pfm`) is a **curated collection** of computational models that
**generate biological patterns** — stripes and spots, sorting and branching,
traveling pulses, tissue-scale organization, and related spontaneous structures.

It is **not** a complete atlas of every model we study. BPFM holds a **selected,
public-ready subset** chosen for biological pattern formation and for reuse in
downstream work (teaching, comparison, and building phenomenon-specific models).

## What this collection is for

- Reproduce classic **biological pattern-formation** models in Python
- Compare mechanisms (reaction–diffusion, chemotaxis, phase field, vertex models, …)
- Compose new biological models from **published building blocks** without pulling in
  the full Lab research tree

Each entry includes a runnable script (`{model_id}.py`), documentation
(`{model_id}.md` with equations and references), and a `results/` folder for
locally regenerated figures.

## Phenomena represented

Examples of biological contexts in this collection (45 models):

- Angiogenesis
- Animal coat marking
- Animal coat marking etc.
- Bacteria cluster
- Bacterial colony
- Blood vessel / flow demo
- Branched structure
- Cell colony shape
- Cell migration
- Cell sorting
- Collective cell migration
- Collective migration
- … and 18 more (see [`CATALOG.md`](CATALOG.md))

Models are grouped mathematically as: **Continuous** (29), **Discrete** (12), **ODE** (4).
Full list: [`CATALOG.md`](CATALOG.md).

## Selection criteria

A model is included in BPFM only when **all** of the following hold:

1. **Biological pattern formation** — the target phenomenon is a living system or
   a standard biological analogue (morphogenesis, cell organization, excitable tissue,
   colony growth, etc.), not a generic physics demo alone.
2. **Published, citable basis** — the original model is documented; each entry’s
   `.md` points to standard literature or a clear attribution.
3. **Public-ready (`published` status)** — cleared for release outside the Lab atlas;
   no dependence on unpublished data or paywalled PDFs bundled in the repo.
4. **Runnable demo** — `{model_id}.py` runs as a self-contained numerical example
   (or states its extra dependencies explicitly).
5. **Curated, not exhaustive** — inclusion means “recommended for reuse,” not
   “every variant we implement.”

**Not included:** draft or lab-only models, auxiliary classification notes,
Mathematica notebooks, and large prebuilt GIF/PDF archives (regenerate locally).

## Using the collection

```bash
cd bio-pfm
python -m pip install -r requirements.txt
cd models/1_continuous_scheme/1_1_diffusion_term/turing_2d
python turing_2d.py
```

Install as a package (optional): `python -m pip install -e .`

## Provenance

BPFM is maintained alongside the Lab atlas
[`Pattern_Formation_Model_Atlas`](../Pattern_Formation_Model_Atlas/).
The authoritative inclusion list is
[`List_of_Pattern_Formation_Models.xlsx`](../Pattern_Formation_Model_Atlas/List_of_Pattern_Formation_Models.xlsx)
(Status column). Maintainer workflow: [`STATUS.md`](STATUS.md).
