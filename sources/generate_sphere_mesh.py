"""
generate_sphere_mesh.py
=======================

Genere un maillage NYMO (.mesh3D) d'une boule de rayon R2 contenant un noyau
de rayon R1, par la methode des mailles coupees (cut-cell cartesienne).

Principe :
    - on quadrille la boite [-R2, R2]^3 en cubes de cote c = 2*R2 / N ;
    - le coeur du maillage est une grille reguliere triviale, seule une coquille
      mince de cubes pres des interfaces demande un traitement special ;
    - une maille entierement hors de R2 est jetee ; une maille a cheval sur une
      sphere est COUPEE PAR UN PLAN UNIQUE qui approxime localement la sphere :
      on ajuste un plan secant passant par le barycentre des intersections
      aretes <-> sphere, de normale radiale, puis on clippe le cube par ce plan.
      L'intersection plan <-> cube est TOUJOURS un polygone plan : le capuchon
      est donc UNE SEULE face coplanaire (critere G8) et la maille reste convexe.

Approximation assumee :
    Chaque cube possede SON propre plan de coupe -> la sphere est remplacee par
    une surface polyedrique facettee (une facette par cube coupe). Les points de
    coupe ne sont donc PAS partages entre cubes voisins : le maillage n'est que
    APPROXIMATIVEMENT conforme le long de l'interface (de petites fentes de
    l'ordre de la fleche de la sphere sur un cube subsistent). C'est le compromis
    choisi pour garantir une face de coupe plane unique par maille. Les sommets
    de grille (coins des cubes), eux, restent partages -> le coeur regulier du
    maillage est parfaitement conforme.

Surfaces :
    R1  -> interface entre le milieu 1 (r < R1) et le milieu 2 (R1 < r < R2) ;
           le capuchon est une face INTERIEURE partagee par les deux demi-mailles
           du meme cube.
    R2  -> bord du domaine (vide autour) ; le capuchon est une face de BORD
           (rD = EXTERIOR), orientee normale sortante.

Un cube a cheval sur les DEUX spheres a la fois (coquille plus fine que c)
n'est pas gere : le programme s'arrete avec une erreur explicite.

Option eps : bande de tolerance (en fraction de c) autour d'une sphere. Un cube
n'est coupe que s'il a un coin strictement a l'interieur (r < radius - eps*c) ET
un coin strictement a l'exterieur (r > radius + eps*c) : on evite ainsi les
coupes en biseau qui produiraient des slivers. eps = 0 -> coupe des qu'un coin
est dedans et un autre dehors.

Usage :
    python sources/generate_sphere_mesh.py R1 R2 N out.mesh3D [eps]
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import nymo_mesh as nm
import numpy as np

# Centre des spheres (origine).
CENTER = np.zeros(3)

# Les 8 sommets d'un cube unite, par offset (di, dj, dk).
CUBE_CORNERS: List[Tuple[int, int, int]] = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]

# Les 6 faces du cube, chacune = boucle ordonnee de 4 indices de sommets
# (winding CCW vue de l'exterieur). Cet ordre est conserve par le clip, ce qui
# donne directement la bonne orientation (normale sortante) aux faces de bord.
CUBE_FACES: List[Tuple[int, int, int, int]] = [
    (0, 3, 2, 1),   # k = 0  (normale -z)
    (4, 5, 6, 7),   # k = 1  (normale +z)
    (0, 1, 5, 4),   # j = 0  (normale -y)
    (3, 7, 6, 2),   # j = 1  (normale +y)
    (0, 4, 7, 3),   # i = 0  (normale -x)
    (1, 2, 6, 5),   # i = 1  (normale +x)
]

# Les 12 aretes du cube (paires d'indices de sommets), pour l'ajustement du plan.
CUBE_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ---------------------------------------------------------------------------
# Constructeur de maillage
# ---------------------------------------------------------------------------
class _Builder:
    """Accumulateur de points / faces / mailles avec deduplication globale."""

    def __init__(self) -> None:
        self.points: List[Tuple[float, float, float]] = []
        self.point_id: Dict[object, int] = {}          # cle -> index de point
        self.faces: List[List[int]] = []
        self.face_id: Dict[Tuple[int, ...], int] = {}  # noeuds tries -> index
        self.cells: List[List[int]] = []
        self.materials: List[int] = []

    def get_point(self, key: object, pos: np.ndarray) -> int:
        idx = self.point_id.get(key)
        if idx is None:
            idx = len(self.points)
            self.point_id[key] = idx
            self.points.append((float(pos[0]), float(pos[1]), float(pos[2])))
        return idx

    def get_face(self, nodes: List[int]) -> int:
        key = tuple(sorted(nodes))
        fid = self.face_id.get(key)
        if fid is None:
            fid = len(self.faces)
            self.face_id[key] = fid
            self.faces.append(list(nodes))
        return fid

    def add_cell(self, face_nodes: List[List[int]], material: int) -> None:
        cell_faces = [self.get_face(nd) for nd in face_nodes]
        self.cells.append(cell_faces)
        self.materials.append(material)

    def mesh(self, name: str) -> nm.Mesh:
        return nm.Mesh(points=np.asarray(self.points, dtype=float),
                       faces=self.faces, cells=self.cells,
                       materials=self.materials, name=name)


# ---------------------------------------------------------------------------
# Geometrie : intersection arete <-> sphere, ajustement du plan de coupe
# ---------------------------------------------------------------------------
def _sphere_edge_pos(pa: np.ndarray, pb: np.ndarray, radius: float) -> np.ndarray:
    """Point d'intersection de l'arete (a, b) avec la sphere centree a l'origine.
    Sert uniquement a echantillonner la sphere pour ajuster le plan de coupe."""
    d = pb - pa
    A = float(np.dot(d, d))
    B = float(np.dot(pa, d))               # centre = origine
    C = float(np.dot(pa, pa)) - radius * radius
    disc = B * B - A * C
    if disc < 0.0:
        disc = 0.0
    sq = disc ** 0.5
    t1 = (-B - sq) / A
    t2 = (-B + sq) / A
    t = t1 if 0.0 <= t1 <= 1.0 else t2
    t = min(1.0, max(0.0, t))
    return pa + t * d


def _fit_cut_plane(pos: List[np.ndarray], inside: List[bool], radius: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Ajuste le plan de coupe du cube approximant la sphere de rayon `radius`.

    On echantillonne la sphere par les intersections des aretes strictement
    traversantes (un coin dedans, un coin dehors), puis on prend le plan passant
    par le barycentre de ces points, de normale radiale (sortante, vers +r).
    `inside[a]` = le coin a est strictement a l'interieur de la sphere."""
    samples = []
    for a, c in CUBE_EDGES:
        if inside[a] != inside[c]:
            samples.append(_sphere_edge_pos(pos[a], pos[c], radius))
    pts = np.asarray(samples)
    centroid = pts.mean(axis=0)
    nlen = float(np.linalg.norm(centroid))
    if nlen > 1e-12:
        normal = centroid / nlen           # tangente radiale a la sphere
    else:
        # Cube centre sur l'origine : normale = direction principale des points.
        _, _, vt = np.linalg.svd(pts - centroid)
        normal = vt[-1]
    return centroid, normal


# ---------------------------------------------------------------------------
# Clip d'un cube par un plan
# ---------------------------------------------------------------------------
def _dedup_cycle(nodes: List[int]) -> List[int]:
    """Supprime les noeuds consecutifs identiques (boucle fermee)."""
    out: List[int] = []
    for nd in nodes:
        if not out or out[-1] != nd:
            out.append(nd)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _order_ring(chords: List[Tuple[int, int]]) -> List[int]:
    """Ordonne les sommets du capuchon en chainant les cordes (cycle unique)."""
    adj: Dict[int, List[int]] = {}
    for a, c in chords:
        adj.setdefault(a, []).append(c)
        adj.setdefault(c, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        raise ValueError("capuchon malforme (cycle non simple) : essayez un N "
                         "different ou un eps plus petit.")
    start = chords[0][0]
    ring = [start]
    prev, cur = None, start
    while True:
        a, c = adj[cur]
        nxt = a if a != prev else c
        if nxt == start:
            break
        ring.append(nxt)
        prev, cur = cur, nxt
        if len(ring) > len(adj):
            raise ValueError("capuchon malforme (cycle non ferme).")
    return ring


def _clip_cube(b: _Builder, cube_tag: object, keys: List[object],
               pos: List[np.ndarray], centroid: np.ndarray, normal: np.ndarray,
               keep_negative: bool, tol: float
               ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Clippe le cube par le plan (centroid, normal) et renvoie (faces laterales,
    cordes du capuchon). On garde le demi-espace `dot(x - centroid, normal) <= 0`
    si keep_negative (cote interieur, vers le centre), sinon `>= 0`.

    Le capuchon (plan <-> cube) est un polygone plan : ses cordes sont chainees
    plus tard en une seule face."""
    sdist = [float(np.dot(p - centroid, normal)) for p in pos]

    def edge_point(a: int, c: int) -> int:
        # Intersection arete <-> plan, dedupliquee par arete AU SEIN du cube
        # (cle = cube_tag + paire de coins) -> capuchon ferme et manifold.
        key = ("pedge", cube_tag, frozenset((keys[a], keys[c])))
        idx = b.point_id.get(key)
        if idx is not None:
            return idx
        sa, sc = sdist[a], sdist[c]
        t = sa / (sa - sc)
        t = min(1.0, max(0.0, t))
        return b.get_point(key, pos[a] + t * (pos[c] - pos[a]))

    side_faces: List[List[int]] = []
    chords: List[Tuple[int, int]] = []

    for face in CUBE_FACES:
        poly: List[int] = []
        cap_pts: List[int] = []
        m = len(face)
        for ai in range(m):
            a = face[ai]
            c = face[(ai + 1) % m]
            sa, sc = sdist[a], sdist[c]
            on_a = abs(sa) <= tol
            keep_a = on_a or (sa < -tol if keep_negative else sa > tol)
            if keep_a:
                pid = b.get_point(keys[a], pos[a])
                poly.append(pid)
                if on_a:
                    cap_pts.append(pid)
            # Arete strictement traversante (coins de part et d'autre du plan).
            if (sa < -tol and sc > tol) or (sa > tol and sc < -tol):
                ip = edge_point(a, c)
                poly.append(ip)
                cap_pts.append(ip)

        poly = _dedup_cycle(poly)
        if len(poly) >= 3:
            side_faces.append(poly)

        cap_pts = list(dict.fromkeys(cap_pts))     # dedup en gardant l'ordre
        if len(cap_pts) == 2:
            chords.append((cap_pts[0], cap_pts[1]))
        elif len(cap_pts) > 2:
            raise ValueError("face de cube coupee en plus de 2 points (cas "
                             "degenere) : essayez un N plus grand ou un eps "
                             "plus petit.")

    return side_faces, chords


def _cap_face(b: _Builder, chords: List[Tuple[int, int]], outward: bool
              ) -> Optional[List[int]]:
    """Construit le capuchon comme UNE seule face plane (ring ordonne). Si
    outward, l'oriente normale sortante (vers +r) pour une face de bord."""
    if not chords:
        return None
    ring = _order_ring(chords)
    if len(ring) < 3:
        return None
    if outward:
        pts = np.array([b.points[n] for n in ring])
        cen = pts.mean(axis=0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if float(np.dot(normal, cen - CENTER)) < 0.0:
            ring = ring[::-1]
    return ring


# ---------------------------------------------------------------------------
# Construction complete
# ---------------------------------------------------------------------------
def build_sphere_mesh(r1: float, r2: float, n: int, eps: float = 0.0) -> nm.Mesh:
    """Construit le maillage cut-cell (coupe a plan unique) de la boule r2 (noyau r1)."""
    if not (0.0 < r1 < r2):
        raise ValueError("on attend 0 < R1 < R2.")
    if n < 1:
        raise ValueError("N doit etre >= 1.")
    if not (0.0 <= eps < 0.5):
        raise ValueError("eps doit etre dans [0, 0.5[.")

    c = 2.0 * r2 / n
    band = eps * c                          # bande de tolerance autour des spheres
    tol = 1e-9 * c                          # tolerance numerique "sur le plan"
    b = _Builder()

    def corner_pos(key: Tuple[int, int, int]) -> np.ndarray:
        return CENTER + (-r2 + np.asarray(key, dtype=float) * c)

    def counts(pos: List[np.ndarray], radius: float
               ) -> Tuple[List[bool], int, int]:
        """Renvoie (inside_strict, nb_inside_strict, nb_outside_strict) pour la
        sphere `radius` avec la bande de tolerance `band`."""
        inside = []
        nin = nout = 0
        for p in pos:
            rn = float(np.sqrt(np.dot(p, p)))
            ins = rn < radius - band
            inside.append(ins)
            if ins:
                nin += 1
            elif rn > radius + band:
                nout += 1
        return inside, nin, nout

    def cut_cell(cube_tag, keys, pos, inside, radius, keep_negative,
                 material, outward_cap):
        side, chords = _clip_cube(b, cube_tag, keys, pos,
                                  *_fit_cut_plane(pos, inside, radius),
                                  keep_negative=keep_negative, tol=tol)
        cap = _cap_face(b, chords, outward=outward_cap)
        if cap is None or len(side) < 3:
            raise ValueError(f"coupe degeneree sur le cube {cube_tag} : "
                             "essayez un N plus grand.")
        b.add_cell(side + [cap], material=material)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                keys = [(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS]
                pos = [corner_pos(key) for key in keys]
                in1, nin1, nout1 = counts(pos, r1)
                in2, nin2, nout2 = counts(pos, r2)

                cut1 = nin1 >= 1 and nout1 >= 1
                cut2 = nin2 >= 1 and nout2 >= 1

                if cut1 and cut2:
                    raise ValueError(
                        f"cube ({i},{j},{k}) a cheval sur les deux spheres "
                        "(coquille plus fine que le pas c). Augmentez N ou "
                        "ecartez R1 et R2.")

                if nin2 == 0:
                    continue                       # aucun coin dans R2 -> jete

                if cut2:                           # coupe par le bord R2
                    cut_cell(("R2", i, j, k), keys, pos, in2, r2,
                             keep_negative=True, material=2, outward_cap=True)
                    continue

                if cut1:                           # coupe par l'interface R1
                    plane = _fit_cut_plane(pos, in1, r1)
                    in_side, ch_in = _clip_cube(b, ("R1", i, j, k), keys, pos,
                                                *plane, keep_negative=True, tol=tol)
                    out_side, ch_out = _clip_cube(b, ("R1", i, j, k), keys, pos,
                                                  *plane, keep_negative=False, tol=tol)
                    cap_in = _cap_face(b, ch_in, outward=False)
                    cap_out = _cap_face(b, ch_out, outward=False)
                    if cap_in is None or cap_out is None:
                        raise ValueError(f"coupe R1 degeneree sur le cube "
                                         f"({i},{j},{k}) : essayez un N plus grand.")
                    b.add_cell(in_side + [cap_in], material=1)    # noyau (r < R1)
                    b.add_cell(out_side + [cap_out], material=2)  # coquille
                    continue

                # Maille pleine : noyau si aucun coin hors de R1, coquille sinon.
                mat = 1 if nout1 == 0 else 2
                faces = [[b.get_point(keys[a], pos[a]) for a in face]
                         for face in CUBE_FACES]
                b.add_cell(faces, material=mat)

    return b.mesh(name=f"sphere_R1_{r1:g}_R2_{r2:g}_N{n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    if len(argv) not in (5, 6):
        print("Usage :\n"
              "  python sources/generate_sphere_mesh.py R1 R2 N out.mesh3D [eps]")
        return 1

    r1, r2 = float(argv[1]), float(argv[2])
    n = int(argv[3])
    out = argv[4]
    eps = float(argv[5]) if len(argv) == 6 else 0.0

    mesh = build_sphere_mesh(r1, r2, n, eps=eps)
    nm.write_mesh(mesh, out)

    n1 = sum(1 for m in mesh.materials if m == 1)
    n2 = sum(1 for m in mesh.materials if m == 2)
    print(f"{out} : {mesh.nb_cells} mailles ({n1} noyau, {n2} coquille), "
          f"{mesh.nb_faces} faces, {mesh.nb_points} points "
          f"[R1={r1:g} R2={r2:g} N={n} eps={eps:g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
