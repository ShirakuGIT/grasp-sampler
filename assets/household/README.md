# Original household scans

The MuJoCo example uses these original appearances, with no object palette tint.
OBJ and PNG files provide the native viewer geometry, UVs and textures. GLBs
provide the same textured surfaces to the grasp sampler (Y-up, metres). OBJ
coordinates and convex contact pieces are Z-up, metres, centered on bounds.

Mug (025), plate (029), cup (065-f), mustard bottle (006), potted meat can (010)
and foam brick (061) originate from the **YCB Object and Model Set**, by Berk
Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel and Aaron
Dollar. Source: <https://ycb-benchmarks.s3.amazonaws.com/index.html>.
License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Imported from existing local Google 16k scans. Modifications: centered coordinates,
axis conversion for GLB, format conversion, and derived convex collision pieces.
Original UV coordinates and texture pixels are retained.

`classic_*` OBJ/PNG assets retain the earlier demo's exact decimated geometry
and contact shapes. Their colors are transferred from the corresponding original
YCB scan after aligning and scaling bounds, then baked onto triangle UV atlases.
That color transfer is approximate; these classic assets do not retain the
original UV mapping. This keeps the earlier geometry-based sequence reproducible.

The Ritz box uses this project's existing 2,000-face geometry and the original
dense, vertex-colored workspace reconstruction from
`imagination/meshes/ritz_cracker_box/ritz_cracker_box.mesh.metric-fp.glb`.
Source colors are baked onto a triangle UV atlas by nearest source vertex;
this is an approximation of the dense original, not a newly painted label.
This reconstruction is separate from the YCB data above.

`*_collision.npz` contains individually indexed convex parts (vertices_N and
faces_N). Mug, cup and plate are decomposed with
[CoACD](https://github.com/SarahWeiii/CoACD) (MIT), rather than filled with a
single convex hull. These are approximate simulator contact surfaces, not exact
scanned surfaces. Demo mass is 60 g per object (30 g for the lightweight plate);
these are explicit example parameters, not source YCB inertia measurements.
Cup and mug use 40 convex parts each; the thin plate uses 128 parts with higher
voxel preprocessing resolution. These hull limits can exceed CoACD's requested
error threshold, so the contact geometry remains an approximation.

Rebuild from the original workspace with `tools/prepare_household_assets.py`;
the preparation tool requires `coacd==1.0.14`. The checked-in assets need no
CoACD dependency at runtime. MuJoCo materials explicitly reference the PNGs;
MuJoCo does not import the OBJ material library.

The expanded scene also uses the existing textured workspace bowl reconstruction
from `Documents/Coding/bowl/meshes/bowl_textured.obj`. Its source UVs and PNG
pixels are preserved. The source has unit diameter and Y-up coordinates; the
preparation tool explicitly scales it to a 16 cm diameter and rotates to Z-up.
This bowl is a separate workspace reconstruction, not an attributed YCB scan.
It has 40 approximate convex contact pieces and an illustrative 60 g mass;
its inertia uses convex-piece volume weights rather than equal piece masses.
