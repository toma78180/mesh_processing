"""
nymo_mesh.py
============

Module central pour la manipulation des geometries d'entree NYMO.

Format REEL du fichier (indexation a partir de 0) :

    Geometrie:
    <nom_de_la_geometrie>
    Noeuds:
    <nb_noeuds>
    i  x y z                      (i a partir de 0)
    ...
    Faces:
    <nb_faces>
    i  nb_nodes  n_1 n_2 n_3 ...  (index a partir de 0)
    ...
    Mailles:
    <nb_mailles>
    i  nb_faces  f_1 f_2 f_3 ...  (index a partir de 0)
    ...
    Milieux:
    id id id id ... id            (une valeur par maille, plusieurs par ligne)
    ...
    Fin:

Particularites par rapport a l'ancien format suppose :
    - l'en-tete d'un bloc et son compteur sont sur DEUX lignes distinctes ;
    - les noeuds sont nommes "Noeuds" (pas "Points") ;
    - l'indexation est deja a partir de 0 (aucune conversion 1<->0) ;
    - le bloc Milieux n'a pas de compteur : on lit nb_mailles valeurs,
      reparties librement sur plusieurs lignes ;
    - le fichier se termine par le bloc "Fin:".

Convention d'orientation (identique a NYMO) :
    la normale sortante d'une face va de la region GAUCHE (rG) vers la
    region DROITE (rD). Pour une face de bord, rD vaut EXTERIOR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Marqueur de region exterieure (face de bord : une seule maille voisine).
EXTERIOR: int = -1

# Seuil de coplanarite (volume signe quasi nul) pour les predicats.
TINY: float = 1e-12


# ---------------------------------------------------------------------------
# Structure de donnees
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    """Geometrie polyedrique (indexation interne a partir de 0)."""

    points: np.ndarray                      # (nb_points, 3) float
    faces: List[List[int]]                  # faces[f] = liste des index de noeuds
    cells: List[List[int]]                  # cells[c] = liste des index de faces
    materials: List[int]                    # materials[c] = id de milieu de la maille c
    name: str = "geometry"                  # nom de la geometrie (bloc "Geometrie:")

    @property
    def nb_points(self) -> int:
        return len(self.points)

    @property
    def nb_faces(self) -> int:
        return len(self.faces)

    @property
    def nb_cells(self) -> int:
        return len(self.cells)


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def _tokens(line: str) -> List[str]:
    return line.split()


def read_mesh(path: str) -> Mesh:
    """Lit un fichier geometrie NYMO (.msh3D) et renvoie un objet Mesh (indexe a 0)."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    name: str = "geometry"
    points: np.ndarray = np.empty((0, 3))
    faces: List[List[int]] = []
    cells: List[List[int]] = []
    materials: List[int] = []

    i = 0
    n = len(lines)
    while i < n:
        header = lines[i].rstrip(":").strip().lower()
        i += 1

        if header == "geometrie":
            name = lines[i]; i += 1

        elif header == "noeuds":
            count = int(lines[i]); i += 1
            pts = []
            for _ in range(count):
                tok = _tokens(lines[i]); i += 1
                pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            points = np.asarray(pts, dtype=float)

        elif header == "faces":
            count = int(lines[i]); i += 1
            for _ in range(count):
                tok = _tokens(lines[i]); i += 1
                nb_nodes = int(tok[1])
                nodes = [int(tok[2 + k]) for k in range(nb_nodes)]   # deja a 0
                faces.append(nodes)

        elif header == "mailles":
            count = int(lines[i]); i += 1
            for _ in range(count):
                tok = _tokens(lines[i]); i += 1
                nb_f = int(tok[1])
                cell_faces = [int(tok[2 + k]) for k in range(nb_f)]  # deja a 0
                cells.append(cell_faces)

        elif header == "milieux":
            # Pas de compteur : on lit une valeur par maille, reparties
            # sur plusieurs lignes, jusqu'a atteindre nb_mailles valeurs.
            nb_expected = len(cells)
            while len(materials) < nb_expected and i < n:
                # On s'arrete si on tombe sur un nouvel en-tete (ex : "Fin:")
                if lines[i].endswith(":") and not _tokens(lines[i])[0].lstrip("-").isdigit():
                    break
                materials.extend(int(v) for v in _tokens(lines[i]))
                i += 1

        elif header == "fin":
            break

        else:
            raise ValueError(f"Bloc inconnu : '{lines[i - 1]}'")

    return Mesh(points=points, faces=faces, cells=cells,
                materials=materials, name=name)


# ---------------------------------------------------------------------------
# Ecriture
# ---------------------------------------------------------------------------
def write_mesh(mesh: Mesh, path: str, mat_per_line: int = 20) -> None:
    """Ecrit un objet Mesh au format geometrie NYMO reel (.msh3D, indexe a 0)."""
    with open(path, "w") as fh:
        fh.write("Geometrie:\n")
        fh.write(f"{mesh.name}\n")

        fh.write("Noeuds:\n")
        fh.write(f"{mesh.nb_points}\n")
        for i, (x, y, z) in enumerate(mesh.points):
            fh.write(f" {i} {x} {y} {z}\n")

        fh.write("Faces:\n")
        fh.write(f"{mesh.nb_faces}\n")
        for i, nodes in enumerate(mesh.faces):
            nodes_str = " ".join(str(nd) for nd in nodes)
            fh.write(f" {i} {len(nodes)} {nodes_str}\n")

        fh.write("Mailles:\n")
        fh.write(f"{mesh.nb_cells}\n")
        for i, cell_faces in enumerate(mesh.cells):
            faces_str = " ".join(str(f) for f in cell_faces)
            fh.write(f" {i} {len(cell_faces)} {faces_str}\n")

        fh.write("Milieux:\n")
        for start in range(0, len(mesh.materials), mat_per_line):
            chunk = mesh.materials[start:start + mat_per_line]
            fh.write(" " + " ".join(str(m) for m in chunk) + "\n")

        fh.write("Fin:\n")


# ---------------------------------------------------------------------------
# Helpers geometriques
# ---------------------------------------------------------------------------
def cell_node_ids(mesh: Mesh, cell: int) -> List[int]:
    """Renvoie l'ensemble (trie) des noeuds d'une maille (union de ses faces)."""
    nodes = set()
    for f in mesh.cells[cell]:
        nodes.update(mesh.faces[f])
    return sorted(nodes)


def cell_barycenter(mesh: Mesh, cell: int) -> np.ndarray:
    """Barycentre (origine) d'une maille = moyenne de ses sommets."""
    ids = cell_node_ids(mesh, cell)
    return mesh.points[ids].mean(axis=0)


def cell_volume(mesh: Mesh, cell: int) -> float:
    """Volume d'une maille a faces triangulaires."""
    G = cell_barycenter(mesh, cell)
    vol = 0.0
    for f in mesh.cells[cell]:
        nd = mesh.faces[f]
        a, b, c = mesh.points[nd[0]], mesh.points[nd[1]], mesh.points[nd[2]]
        vol += abs(float(np.dot(a - G, np.cross(b - G, c - G)))) / 6.0
    return vol


def total_volume(mesh: Mesh) -> float:
    return sum(cell_volume(mesh, c) for c in range(mesh.nb_cells))


def face_normal(mesh: Mesh, face: int) -> np.ndarray:
    """Normale faciale non normalisee : (P1-P0) x (P2-P0)."""
    n0, n1, n2 = mesh.faces[face][0], mesh.faces[face][1], mesh.faces[face][2]
    p0, p1, p2 = mesh.points[n0], mesh.points[n1], mesh.points[n2]
    return np.cross(p1 - p0, p2 - p0)


def build_face_to_cells(mesh: Mesh) -> Dict[int, List[int]]:
    """Pour chaque face, liste des mailles qui la referencent."""
    f2c: Dict[int, List[int]] = {f: [] for f in range(mesh.nb_faces)}
    for c, cell_faces in enumerate(mesh.cells):
        for f in cell_faces:
            f2c[f].append(c)
    return f2c


def is_boundary_face(f2c: Dict[int, List[int]], face: int) -> bool:
    return len(f2c[face]) == 1


def is_tetrahedral(mesh: Mesh) -> bool:
    """Vrai si toutes les mailles sont des tetraedres (4 faces triangulaires)."""
    for cell_faces in mesh.cells:
        if len(cell_faces) != 4:
            return False
        if any(len(mesh.faces[f]) != 3 for f in cell_faces):
            return False
    return True


def assert_tetrahedral(mesh: Mesh) -> None:
    """Leve une erreur explicite si le maillage n'est pas purement tetraedrique."""
    for c, cell_faces in enumerate(mesh.cells):
        if len(cell_faces) != 4:
            raise ValueError(
                f"Maille {c} : {len(cell_faces)} faces (4 attendues). "
                "Le maillage d'entree doit etre purement tetraedrique.")
        for f in cell_faces:
            if len(mesh.faces[f]) != 3:
                raise ValueError(
                    f"Face {f} : {len(mesh.faces[f])} noeuds (3 attendus). "
                    "Le maillage d'entree doit etre purement tetraedrique.")


def get_face_regions(
    mesh: Mesh,
    face: int,
    f2c: Dict[int, List[int]],
    barycenters: List[np.ndarray],
) -> Tuple[int, int]:
    """Renvoie (rG, rD) tel que la normale sortante va de rG vers rD."""
    cells = f2c[face]
    normal = face_normal(mesh, face)
    p0 = mesh.points[mesh.faces[face][0]]

    if len(cells) == 1:
        return cells[0], EXTERIOR

    ca, cb = cells
    dot = float(np.dot(normal, barycenters[ca] - p0))
    if dot < 0:
        return ca, cb
    return cb, ca


# ---------------------------------------------------------------------------
# Predicat d'orientation et test de convexite
# ---------------------------------------------------------------------------
def orient3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Volume signe x6 du tetraedre (a, b, c, d) = det[b-a, c-a, d-a].

    > 0  : d est du cote positif du plan oriente (a, b, c) ;
    < 0  : d est du cote oppose ;
    ~ 0  : les quatre points sont coplanaires.

    C'est la brique de base ("orientation predicate") pour tous les tests de
    cote / d'appartenance, sans aucune division.
    """
    return float(np.dot(b - a, np.cross(c - a, d - a)))


def cell_convexity_status(mesh: Mesh, cell: int, rel_eps: float = 1e-9) -> str:
    """Classe une maille (polyedre a faces triangulaires) en :

        "convexe"   : strictement convexe (tous les diedres < 180 deg) ;
        "plate"     : convexe-limite, au moins une paire de faces coplanaires
                      (un diedre = 180 deg, bipyramide degeneree) ;
        "rentrante" : non convexe (au moins un diedre > 180 deg).

    Principe : pour CHAQUE face, on compare le cote de chaque autre sommet de la
    maille a celui du barycentre G (strictement interieur d'un polyedre convexe).
    - un sommet du cote OPPOSE a G (au-dela du seuil) => arete rentrante ;
    - un sommet exactement sur le plan d'une face => faces coplanaires => plat.

    La comparaison de cote utilise le signe de `orient3d` (sans dimension). Le
    seuil `rel_eps` (relatif au volume de la maille) absorbe le bruit numerique :
    une violation n'est comptee que si |orient3d| > 6 * rel_eps * volume.
    """
    pts = mesh.points
    nodes = cell_node_ids(mesh, cell)
    G = cell_barycenter(mesh, cell)
    vol = max(cell_volume(mesh, cell), TINY)
    seuil = 6.0 * rel_eps * vol            # orient3d = 6 x volume signe
    plat = False

    for f in mesh.cells[cell]:
        fn = mesh.faces[f]
        face_set = set(fn)
        p0, p1, p2 = pts[fn[0]], pts[fn[1]], pts[fn[2]]
        sG = orient3d(p0, p1, p2, G)
        if abs(sG) <= seuil:
            continue                       # G ~ coplanaire : face non concluante
        for v in nodes:
            if v in face_set:
                continue                   # sommet porte par la face elle-meme
            sv = orient3d(p0, p1, p2, pts[v])
            if sv * sG < 0.0 and abs(sv) > seuil:
                return "rentrante"         # sommet du mauvais cote (prioritaire)
            if abs(sv) <= seuil:
                plat = True                # sommet coplanaire => faces coplanaires
    return "plate" if plat else "convexe"


def cell_is_convex(mesh: Mesh, cell: int, strict: bool = False,
                   rel_eps: float = 1e-9) -> bool:
    """Vrai si la maille est convexe.

    strict=False : on accepte le cas "plate" (diedre 180 deg) comme convexe-limite.
    strict=True  : seules les mailles strictement convexes passent (le cas plat
                   est rejete, conformement au filtre de fusion).
    """
    status = cell_convexity_status(mesh, cell, rel_eps)
    return status == "convexe" if strict else status != "rentrante"
