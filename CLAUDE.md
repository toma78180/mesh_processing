# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small toolkit for generating, transforming, validating and visualizing **NYMO
polyhedral meshes** (3D finite-volume geometries). Everything is plain Python +
NumPy (matplotlib for the viewer). There is no package, build step, or test
suite — each script is run directly from the command line.

Docstrings, comments and CLI output are in **French** (without accents). Match
that style when editing.

## Layout

- `sources/` — all Python: `nymo_mesh.py` (core module, imported as `nm` by
  every script) plus the CLI tools.
- `meshes/` — committed mesh data (`.mesh3D`).
- `images/` — saved viewer output (`*.png`, gitignored).

Scripts are run from the repo root as `python sources/<tool>.py ...`. Imports
work because Python puts the script's own directory (`sources/`) on `sys.path`,
so `import nymo_mesh` resolves to `sources/nymo_mesh.py`; mesh paths on the
command line are relative to the current directory (the repo root).

## Running the tools

```sh
# Generate purely-tetrahedral test meshes (Kuhn decomposition).
# No args -> writes the DEFAULTS set into examples/ (create the dir first).
python sources/generate_tetra_mesh.py
python sources/generate_tetra_mesh.py 3 2 2 out.mesh3D [jitter] [seed]

# Merge N tetrahedra back into hexahedra/polyhedra (inverse decomposition).
python sources/tetra_to_hexa.py in.mesh3D out.mesh3D [--ntet 5] [--tol 179.99]

# Generate a cut-cell sphere mesh: core radius R1 inside a ball of radius R2,
# cartesian grid of step c = 2*R2/N, cells straddling an interface are cut.
python sources/generate_sphere_mesh.py R1 R2 N out.mesh3D [eps]

# Validate a geometry (11 structural checks G1..G11; exit 0 ok, 2 on failure).
python sources/validate.py meshes/tetra_Sphere.mesh3D

# Viewer with two subcommands (matplotlib), colored by material:
#   3d    -> skin view, optional axis clip / interior wireframe
#   coupe -> planar cross-section
python sources/view.py 3d    meshes/tetra_Sphere.mesh3D [--interior] [--clip x=0] [--save images/v.png]
python sources/view.py coupe meshes/tetra_Sphere.mesh3D x=1.5 [--interior] [--save images/v.png]
```

There is no automated test harness; `validate.py` is the closest thing — run it
on any mesh a transform produces to confirm correctness (volume conservation is
also checked inside `tetra_to_hexa.py`).

## Architecture

`sources/nymo_mesh.py` is the foundation every other script imports as `nm`. Key
pieces:

- **`Mesh` dataclass** — the single in-memory representation. A mesh is
  `points` (Nx3 array) + `faces` (each face = list of node indices) + `cells`
  (each cell = list of *face* indices) + `materials` (one id per cell). Note the
  indirection: cells reference faces, faces reference points. All indices are
  **0-based**, both in memory and on disk.
- **`read_mesh` / `write_mesh`** — parse/emit the NYMO text format. The on-disk
  layout has block headers and their counts on *separate* lines
  (`Geometrie:` / `Noeuds:` / `Faces:` / `Mailles:` / `Milieux:` / `Fin:`); the
  `Milieux` block has no count (read `nb_cells` values). The format spec lives in
  the module docstring — read it before touching the parser.
- **Geometry helpers** used everywhere: `build_face_to_cells` (face -> owning
  cells; 1 owner = boundary, 2 = interior), `cell_barycenter`, `cell_volume`,
  `face_normal`, `orient3d` (signed-volume predicate), and
  `cell_convexity_status` (`convexe` / `plate` / `rentrante`).

**Orientation convention (NYMO):** a face normal points from the left region
(rG) to the right region (rD); for a boundary face rD = `EXTERIOR` (-1).
`validate.py`'s G9 check and `get_face_regions` both depend on this.

**Data flow:** `generate_tetra_mesh.py` (Kuhn/Freudenthal split, 6 tets/cube,
optionally jittered to break coplanarity) -> tetrahedral `.mesh3D` ->
`tetra_to_hexa.py` (groups tets back into 8-node blocks, matches surface
triangle pairs into quads under a strict coplanarity tolerance, keeping
non-planar pairs as triangles so output may be hybrid polyhedra) -> hex/poly
mesh. `validate.py` and `view.py` consume any mesh. (`view.py` merges the former
`visualize.py` + `coupe.py`; its `coupe` mode assumes convex cells.)

`generate_sphere_mesh.py` is an independent generator (not part of the
tetra→hexa chain): a **cut-cell** method that grids `[-R2,R2]³` into cubes of
side `c = 2·R2/N`, keeps the regular interior cells verbatim, and clips only the
thin shell of cubes that straddle the R1 interface (split into milieu 1 / milieu
2) or the R2 boundary (kept inside, outside dropped). Each straddling cube is cut
by **a single plane** that approximates the sphere locally (a secant plane fitted
through the cube's edge–sphere intersections, with radial normal); clipping a
cube by one plane yields a **single planar cap** (`_cap_face`), so every cut cell
stays **convex** and the interface face passes G8. The trade-off: each cube has
its *own* cut plane, so cut points are NOT shared between neighbours — the mesh is
only *approximately* conformal along the interface (small gaps ~ the sphere's sag
over one cube; this shows up as extra boundary faces, still valid under G4). Grid
corners stay shared, so the regular core is exactly conformal. A cube straddling
*both* spheres at once raises an error (use a larger N). NOTE: `nm.cell_volume`
assumes triangular faces, so it under-reports volume for these polygonal cells —
fan-triangulate each face to measure volume.

File extensions are inconsistent across docstrings (`.geo`, `.msh3D`,
`.mesh3D`) but the format is identical — committed data files use `.mesh3D`.
