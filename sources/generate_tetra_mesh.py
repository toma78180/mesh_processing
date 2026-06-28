"""
generate_tet_meshes.py
======================

Genere des maillages d'essai PUREMENT TETRAEDRIQUES et conformes au format
NYMO. Une grille nx*ny*nz de cubes unite est decoupee en tetraedres par la
decomposition de Kuhn/Freudenthal (6 tetraedres par cube, partageant la
grande diagonale). Avec la meme orientation pour tous les cubes, le maillage
est conforme : chaque face interne est partagee par exactement 2 tetraedres.

Materiaux : domaine coupe en deux selon x (milieu 1 / milieu 2) pour avoir une
interface de milieux (pas de fusion a travers) et beaucoup de fusions internes.

Usage :
    python generate_tet_meshes.py             # genere le jeu par defaut
    python generate_tet_meshes.py 3 2 2 out.geo
"""

from __future__ import annotations

import sys
from itertools import permutations
from typing import Dict, List, Tuple

import nymo_mesh as nm
import numpy as np


# ---------------------------------------------------------------------------
# Decomposition de Kuhn d'un cube en 6 tetraedres
# ---------------------------------------------------------------------------
def kuhn_tets(origin: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], ...]]:
    """6 tetraedres d'un cube unite a partir de son coin 'origin'.

    Chaque permutation (a,b,c) de (x,y,z) donne le chemin
    000 -> +a -> +b -> +c = 111. Les 4 sommets forment un tetraedre.
    """
    ox, oy, oz = origin
    tets = []
    for perm in permutations(range(3)):
        v = [0, 0, 0]
        path = [tuple(v)]
        for axis in perm:
            v[axis] = 1
            path.append(tuple(v))
        tet = tuple((ox + a, oy + b, oz + c) for (a, b, c) in path)
        tets.append(tet)
    return tets


# ---------------------------------------------------------------------------
# Construction du maillage complet
# ---------------------------------------------------------------------------
def build_grid(nx: int, ny: int, nz: int,
               jitter: float = 0.0, seed: int = 0) -> nm.Mesh:
    """Construit un Mesh tetraedrique pour une grille nx*ny*nz de cubes.

    jitter : amplitude de perturbation aleatoire des sommets (en fraction du pas
             de grille). 0.0 => decomposition de Kuhn exacte (bipyramides
             *plates*, donc 0 fusion sous le filtre convexe). jitter > 0 casse
             la coplanarite et fait apparaitre de vraies fusions convexes
             (et quelques rejets rentrants), ce qui exerce reellement le filtre.
    seed   : graine du generateur aleatoire (reproductibilite).

    La perturbation deplace les POINTS (par index) apres construction : comme les
    faces et les mailles referencent les points par index et que la conformite
    est purement combinatoire, le maillage perturbe reste conforme (chaque face
    interne reste partagee par exactement 2 tetraedres) et chaque face triangle
    reste plane par construction. Une amplitude modeste (~0.15) garantit des
    tetraedres non degeneres / non retournes.
    """
    point_id: Dict[Tuple[int, int, int], int] = {}
    points: List[Tuple[float, float, float]] = []

    def get_point(coord: Tuple[int, int, int]) -> int:
        if coord not in point_id:
            point_id[coord] = len(points)
            points.append((float(coord[0]), float(coord[1]), float(coord[2])))
        return point_id[coord]

    face_id: Dict[Tuple[int, ...], int] = {}
    faces: List[List[int]] = []

    def get_face(node_ids: Tuple[int, int, int]) -> int:
        key = tuple(sorted(node_ids))                 # identite = ensemble de noeuds
        if key not in face_id:
            face_id[key] = len(faces)
            faces.append(list(key))
        return face_id[key]

    cells: List[List[int]] = []
    materials: List[int] = []
    x_mid = nx / 2.0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                mat = 1 if (i + 0.5) < x_mid else 2
                for tet in kuhn_tets((i, j, k)):
                    ids = [get_point(c) for c in tet]          # 4 sommets
                    # 4 faces triangulaires du tetraedre
                    tet_faces = [
                        get_face((ids[0], ids[1], ids[2])),
                        get_face((ids[0], ids[1], ids[3])),
                        get_face((ids[0], ids[2], ids[3])),
                        get_face((ids[1], ids[2], ids[3])),
                    ]
                    cells.append(tet_faces)
                    materials.append(mat)

    pts = np.asarray(points, dtype=float)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        pts = pts + rng.uniform(-jitter, jitter, pts.shape)

    return nm.Mesh(points=pts,
                   faces=faces, cells=cells, materials=materials)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULTS = [
    # Maillages de Kuhn EXACTS : bipyramides plates -> 0 fusion convexe.
    # Conserves comme temoins (et pour la fusion naive --no-convex).
    ("examples/tet_24.geo", (2, 2, 1), 0.0),   # 6 * 4  = 24 mailles
    ("examples/tet_48.geo", (2, 2, 2), 0.0),   # 6 * 8  = 48 mailles
    ("examples/tet_54.geo", (3, 3, 1), 0.0),   # 6 * 9  = 54 mailles
    ("examples/tet_96.geo", (4, 2, 2), 0.0),   # 6 * 16 = 96 mailles
    # Versions PERTURBEES (jitter) : coplanarite cassee -> vraies fusions
    # convexes. Ce sont elles qui exercent le filtre de convexite stricte.
    ("examples/tet_24_jitter.geo", (2, 2, 1), 0.15),
    ("examples/tet_48_jitter.geo", (2, 2, 2), 0.15),
    ("examples/tet_54_jitter.geo", (3, 3, 1), 0.15),
    ("examples/tet_96_jitter.geo", (4, 2, 2), 0.15),
]


JITTER_SEED = 0


def main(argv: List[str]) -> int:
    # python generate_tet_meshes.py nx ny nz out.geo [jitter] [seed]
    if len(argv) >= 5:
        nx, ny, nz = int(argv[1]), int(argv[2]), int(argv[3])
        out = argv[4]
        jitter = float(argv[5]) if len(argv) >= 6 else 0.0
        seed = int(argv[6]) if len(argv) >= 7 else JITTER_SEED
        mesh = build_grid(nx, ny, nz, jitter=jitter, seed=seed)
        nm.write_mesh(mesh, out)
        tag = f", jitter={jitter:g}" if jitter > 0.0 else " (Kuhn exact)"
        print(f"{out} : {mesh.nb_cells} mailles, {mesh.nb_faces} faces, "
              f"{mesh.nb_points} points{tag}")
        return 0

    for path, (nx, ny, nz), jitter in DEFAULTS:
        mesh = build_grid(nx, ny, nz, jitter=jitter, seed=JITTER_SEED)
        nm.write_mesh(mesh, path)
        tag = f"jitter={jitter:g}" if jitter > 0.0 else "Kuhn exact"
        print(f"{path} : grille {nx}x{ny}x{nz} -> {mesh.nb_cells} mailles, "
              f"{mesh.nb_faces} faces, {mesh.nb_points} points [{tag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
