# Cluster environment

**Host:** `jupyter-seismic-uw-edu---*` (OOI / UW JupyterHub PyTorch instance)

## Hardware
- **CPU:** 176 logical (88 cores × 2 threads, dual Intel Xeon Platinum 8458P)
- **RAM:** 1.5 TiB (≈1.4 TiB free idle)
- **GPU:** 1× NVIDIA L40S, 46 GB VRAM, compute cap 8.9 (Ada)
- **Disk:** `/home/jovyan` is a 100 TB NFS PVC (`prod-jupyter-homes`), 90 TB free; sequential write ≈ 617 MB/s
- **No SLURM / batch scheduler** — single fat node; use shell + Python multiprocessing
- OOI scientific data mounts (`/home/jovyan/ooi/*`, ~7 PB) are read-only Pacific cabled/uncabled — irrelevant to BRAVOSEIS

## Python env
- Created `/home/jovyan/bransfield-eq/.venv` with `python -m venv --system-site-packages` so the conda-provided `torch 2.5.1+cu124` (CUDA 12.4) is reused — avoids re-downloading a multi-GB torch wheel
- Then `pip install -r requirements.txt` brought in `obspy 1.5.0`, `seisbench 0.11.6`, etc.
- `torch.cuda.is_available() == True`, device name `NVIDIA L40S`

## Activate
```bash
source /home/jovyan/bransfield-eq/.venv/bin/activate
cd /home/jovyan/bransfield-eq
```
