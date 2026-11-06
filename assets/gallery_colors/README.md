# Original-source appearances for compact meshes

The 31 GLBs in `assets/meshes` were deliberately decimated as **geometry-only**
assets by `tools/decimate_meshes.py`. These appearance-only OBJ/MTL/PNG sidecars
restore source colors for the gallery without changing those sampler geometries.
They are not additional objects in the 40-object inventory.

Colors come from the original textured YCB scans and the existing vertex-colored
workspace reconstructions. The preparation tool aligns source and compact bounds,
then samples original source vertex colors into independent triangle UV tiles.
Texture-mapped scans first sample their original image at source vertex UVs.
This is an approximate appearance transfer: the compact meshes do not acquire
the original dense scan UV mapping or texture fidelity. No palette is invented.
The nine `assets/household` meshes instead retain their already-prepared original
appearances, with the transfer details documented in that folder.

`sources.json` records each exact
source object, source file and texture hashes, source/target bounds, and transfer
method. Legacy descriptive aliases such as `bandage_box` / `hansaplast_bandaid`,
`black_filter_box` / `toothpaste_box`, `blue_cleaner_box` /
`dishwasher_cleaner_box`, and `cleaner_canister` / `potato_chips_can` are explicit
in that manifest. Bounds alignment and reconstruction differences make these
transfers approximate.

YCB scans originate from the **YCB Object and Model Set**, by Berk Calli, Arjun
Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel and Aaron Dollar:
<https://ycb-benchmarks.s3.amazonaws.com/index.html>.
Dataset license: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The chips-can Berkeley Poisson scan and tennis-ball Google 16k scan were obtained
from that official dataset. Other scans were already present in the workspace.
The non-YCB reconstructions are the existing workspace assets named in the
manifest; they are separate from the YCB dataset.

Rebuild with `PYTHONPATH=src python tools/prepare_gallery_colors.py` after installing
the `asset-prep` extra and locating the original source workspace and downloaded
YCB sources with `--sources` and `--downloads`. The two missing workspace scans
are available at:

- <https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/001_chips_can/001_chips_can_berkeley_meshes.tgz>
- <https://ycb-benchmarks.s3.amazonaws.com/data/google/056_tennis_ball_google_16k.tgz>

Extract those archives into matching `001_chips_can` and `056_tennis_ball`
subdirectories under `--downloads`. Gallery runtime uses the checked-in
sidecars and does not require the original workspace or CoACD.
