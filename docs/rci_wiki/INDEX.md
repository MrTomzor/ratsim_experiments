# RCI cluster wiki (login.rci.cvut.cz/wiki) — local mirror

Raw DokuWiki markup, one file per page. Read only what you need.

## Pages

- `acknowledgment.txt` — **Acknowledgement to RCI project** (1 KB) — All RCI member should add acknowledgement to RCI project to all papers:
- `cedmo_upgrade.txt` — **CEDMO H200 nodes upgrade 2025** (2 KB) — The new cluster nodes have been installed in the summer 2025. The new nodes have got AMD Epyc Turin processors
- `contact.txt` — **Contact** (1 KB) — Mail contact to admins: cluster_admin@rci.cvut.cz
- `containers.txt` — **Running software containers on RCI cluster** (7 KB) — Running software containers on RCI cluster
- `faq.txt` — **Frequently Asked Questions** (2 KB) — How to run jobs on specific nodes
- `hardware.txt` — **RCI Cluster Hardware** (3 KB) — Cluster consists from compute nodes, management nodes, data storage and very fast network. There are three sub
- `hardware_gpu.txt` — **GPU cards** (21 KB) — 12 x GPU nodes n21-n32  each with 4 x NVIDIA Tesla V100 with 32GB graphic memory and NVLink2 interconnection. 
- `hardware_ipu.txt` — **Graphcore IPU support** (7 KB) — There is 1 node ipu01 with attached 1 https://www.graphcore.ai/products/bow-2000Graphcore Bow-2000 unit in our
- `how_to_start.txt` — **How to start to work on RCI cluster** (12 KB) — How to start to work on RCI cluster
- `interactive.txt` — **Web interactive applications** (1 KB) — For interactive work is available the interface on the address https://login2.rci.cvut.cz/ . Every user from C
- `jobs.txt` — **SLURM - job scheduler** (29 KB) — RCI cluser uses SLURM as job scheduler. https://slurm.schedmd.com/quickstart.htmlOfficial quick start is here.
- `jobs_changes.txt` — **Changes in RCI cluster scheduler from July 2020** (3 KB) — Changes in RCI cluster scheduler from July 2020
- `jupyter.txt` — **Jupyter notebooks** (6 KB) — You can use prepared interactiveweb application to run Jupyter notebook on the cluster's compute node in your 
- `matlab.txt` — **Using Matlab on RCI cluster** (14 KB) — This page is intended to help you with running parallel MATLAB codes on the RCI cluster. The latest software m
- `modules.txt` — **Software modules** (32 KB) — On all compute and login nodes it's possible to use so called software modules. ''module'' is software (http:/
- `news.txt` — **RCI cluster news** (36 KB) — 2026-06-15 installed_modules_all#pytorch-geometricPyTorch-Geometric 2.8.0 with PyTorch 2.11.0 support modules 
- `rci_upgrade.txt` — **RCI cluster upgrade 2021** (3 KB) — The new cluster nodes have been installed in the summer 2021. The new nodes have got AMD Epyc Milan processors
- `software.txt` — **RCI Cluster software** (2 KB) — There is only basic runtime environment installed on computation nodes of the cluster. All other software is a
- `start.txt` — **RCI Cluster** (1 KB) — {{ :cluster_photo.png?nolink&400}}http://rci.cvut.czRCI is the centre of scientific excellence in computer sci
- `storage.txt` — **Data Storage** (6 KB) — ^ Volume            ^ Purpose  ^ Capacity ^ Quota ^ Speed ^ Access ^ SMB/CIFS
- `tutorial.txt` — **Tutorial: How to use the RCI cluster in your reasearch** (1 KB) — Tutorial: How to use the RCI cluster in your reasearch

## tables/ — generated listings, DO NOT read whole; grep them

- `tables/installed_containers.txt` (12 KB)
- `tables/installed_modules.txt` (1126 KB)
- `tables/installed_modules_all.txt` (451 KB)
- `tables/installed_modules_amd.txt` (1029 KB)
- `tables/installed_modules_h200.txt` (452 KB)
- `tables/monitor.txt` (104 KB)

```bash
grep -i pytorch tables/installed_modules.txt   # find a module + its version
```

Dropped 9 non-pages (crawler followed external /wiki/ links): LibYAML, Software:HarfBuzz, Software:fontconfig, Software:intltool, Software:pkg-config, Software:xlibs, WikiStart, index.php:Libxc, index.php:NLopt
