# Storage layout

## Where bulk data lives
All bulk waveforms and pick outputs live under `/home/jovyan/my_data/bravoseis/` and are symlinked into the repo:

```
/home/jovyan/my_data/bravoseis/
├── waveforms/      ← bransfield-eq/data/waveforms     (~560 GB year)
├── stationxml/     ← bransfield-eq/data/stationxml    (~MBs)
├── picks/          ← bransfield-eq/catalogs/picks     (PhaseNet output)
└── picks_eqt/      ← bransfield-eq/catalogs/picks_eqt (EQTransformer output)
```

## Why
- `src/bransfield_eq/config.py` hardcodes paths relative to the repo root (`Path(__file__).resolve().parents[2] / "data"`); no env-var override
- The only writable durable filesystem with the needed capacity is `/home/jovyan` — see [`00_cluster_environment.md`](00_cluster_environment.md)
- Symlinking keeps the ~600 GB of data outside the git tree, so the repo can be wiped and re-cloned without touching data
- Matches the convention a colleague hinted at (`my_data/bravoseis/...`)

## .gitignore
Already excludes `data/waveforms/`, `data/stationxml/`, `data/*.mseed`, `catalogs/*.csv`, `catalogs/picks/`, `catalogs/picks_*/` — symlinks themselves are too, by virtue of the path patterns.

## If you add a new bulk output dir
Symlink the same way before running. Example for an EQT-instance variant:
```bash
mkdir -p /home/jovyan/my_data/bravoseis/picks_eqt_instance
ln -s /home/jovyan/my_data/bravoseis/picks_eqt_instance \
      /home/jovyan/bransfield-eq/catalogs/picks_eqt_instance
```
