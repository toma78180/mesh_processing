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
      sphere est COUPEE PAR LA SPHERE elle-meme : les sommets de coupe sont les
      VRAIES intersections aretes <-> sphere. Ces points sont dedupliques par
      arete de GRILLE (cle = paire de coins), donc PARTAGES entre cubes voisins :
      deux cubes adjacents coupent leur face commune au meme endroit et
      l'interface est CONFORME (plus de fentes ni de faces de bord parasites).

Capuchon triangule :
    Les sommets de coupe d'un meme cube ne sont PAS coplanaires (la sphere
    bombe), donc le capuchon ne peut pas etre une face plane unique. On le
    triangule en eventail (fan) depuis un de ses sommets, SANS point ajoute : un
    capuchon a p sommets donne p-2 triangles. Chaque triangle est trivialement
    plan (critere G8) et le capuchon suit la vraie sphere. Contrepartie : la
    maille n'est plus garantie convexe -- cote coquille
    (R1 < r < R2) la sphere bombe vers l'interieur, ce qui peut rendre la maille
    legerement RENTRANTE (a verifier avec validate.py ; view.py coupe suppose des
    mailles convexes). Les coins de grille restant partages, le coeur regulier
    reste exactement conforme.

Surfaces :
    R1  -> interface entre le milieu 1 (r < R1) et le milieu 2 (R1 < r < R2) ;
           le capuchon est un jeu de triangles INTERIEURS partages par les deux
           demi-mailles du meme cube (memes sommets -> memes triangles).
    R2  -> bord du domaine (vide autour) ; le capuchon est un jeu de triangles de
           BORD (rD = EXTERIOR), orientes normale sortante.

Un cube a cheval sur les DEUX spheres a la fois (coquille plus fine que c)
n'est pas gere : le programme s'arrete avec une erreur explicite.

Option eps : seuil de volume (en fraction du volume du cube c^3) sous lequel on
ne COUPE PAS. Quand un cube est a cheval sur une sphere, on estime le volume des
deux morceaux (via un plan secant local, cf. _fit_cut_plane) ; si le plus petit
morceau v' est negligeable (v' <= eps * c^3) on renonce a la coupe et le cube
prend, plein, le milieu du plus gros morceau (pour R2 : soit le cube est garde
plein en milieu 2, soit il est entierement jete).
On evite ainsi les slivers (mailles ecrasees) sans recourir a une bande.
    eps = 0   -> on coupe des qu'un coin est dedans et un autre dehors ;
    eps = 0.5 -> aucune coupe ne survit (un morceau est toujours <= la moitie),
                 on retombe sur un maillage purement cubique (marches d'escalier).

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

# Decomposition de Kuhn du cube en 6 tetraedres autour de la diagonale 0->6.
# Sert uniquement a MESURER le volume d'un demi-cube clippe par un plan (chaque
# tetraedre est clippe analytiquement, cf. _half_volume_below).
CUBE_TETS: List[Tuple[int, int, int, int]] = [
    (0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6),
    (0, 3, 7, 6), (0, 4, 5, 6), (0, 4, 7, 6),
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
    """Ajuste un plan secant local du cube approximant la sphere de rayon `radius`.

    Sert UNIQUEMENT a estimer le volume des deux morceaux pour le critere eps ;
    la coupe geometrique, elle, suit la vraie sphere (cf. _clip_cube_sphere).
    On echantillonne la sphere par les intersections des aretes strictement
    traversantes (un coin dedans, un coin dehors), puis on prend le plan passant
    par le barycentre de ces points, de normale radiale (sortante, vers +r).
    `inside[a]` = le coin a est strictement a l'interieur de la sphere."""
    eta = 1e-9 * float(np.linalg.norm(pos[1] - pos[0]))
    samples = []
    for a, c in CUBE_EDGES:
        if inside[a] != inside[c]:
            samples.append(_sphere_edge_pos(pos[a], pos[c], radius))
    for a in range(len(pos)):               # coins poses sur la sphere
        if abs(float(np.sqrt(np.dot(pos[a], pos[a]))) - radius) <= eta:
            samples.append(pos[a])
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
# Mesure du volume d'un demi-cube clippe (pour le critere eps)
# ---------------------------------------------------------------------------
def _tet_volume(p0: np.ndarray, p1: np.ndarray,
                p2: np.ndarray, p3: np.ndarray) -> float:
    """Volume (positif) du tetraedre (p0, p1, p2, p3)."""
    return abs(float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))) / 6.0


def _tet_clip_volume(P: List[np.ndarray], d: List[float]) -> float:
    """Volume du tetraedre P (4 points) situe du cote `d <= 0` du plan, ou `d`
    sont les distances signees des 4 sommets. Convexe, ne leve jamais d'erreur :
    tous les morceaux issus d'une coupe par un plan sont convexes."""
    neg = [i for i in range(4) if d[i] <= 0.0]
    if len(neg) == 4:
        return _tet_volume(*P)
    if not neg:
        return 0.0

    def ip(a: int, c: int) -> np.ndarray:        # intersection arete a->c <-> plan
        t = d[a] / (d[a] - d[c])
        return P[a] + t * (P[c] - P[a])

    if len(neg) == 1:                            # petit tetraedre au coin interieur
        a = neg[0]
        q = [ip(a, c) for c in range(4) if c != a]
        return _tet_volume(P[a], q[0], q[1], q[2])

    if len(neg) == 3:                            # tout sauf un petit tetraedre
        a = [i for i in range(4) if d[i] > 0.0][0]
        q = [ip(a, c) for c in range(4) if c != a]
        return _tet_volume(*P) - _tet_volume(P[a], q[0], q[1], q[2])

    # len(neg) == 2 : le morceau interieur est un prisme triangulaire (convexe),
    # decompose en 3 tetraedres. Sommets : a, b (coins dedans) + 4 intersections.
    a, b = neg
    c, e = [i for i in range(4) if d[i] > 0.0]
    iac, iae = ip(a, c), ip(a, e)
    ibc, ibe = ip(b, c), ip(b, e)
    # prisme (a, iac, iae) - (b, ibc, ibe), aretes laterales a-b, iac-ibc, iae-ibe.
    return (_tet_volume(P[a], iac, iae, P[b])
            + _tet_volume(iac, iae, P[b], ibc)
            + _tet_volume(iae, P[b], ibc, ibe))


def _half_volume_below(pos: List[np.ndarray], centroid: np.ndarray,
                       normal: np.ndarray) -> float:
    """Volume du cube (8 coins `pos`) du cote NEGATIF du plan (centroid, normal),
    i.e. `dot(x - centroid, normal) <= 0`. Le cube est decompose en 6 tetraedres
    de Kuhn, chacun clippe analytiquement. Fonction pure : ne touche pas au
    builder (donc aucun point orphelin, cf. critere G7)."""
    d = [float(np.dot(p - centroid, normal)) for p in pos]
    return sum(_tet_clip_volume([pos[i] for i in tet], [d[i] for i in tet])
               for tet in CUBE_TETS)


# ---------------------------------------------------------------------------
# Clip d'un cube par la sphere (points partages) + capuchon triangule
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


def _clip_cube_sphere(b: _Builder, keys: List[object], pos: List[np.ndarray],
                      radius: float, keep_inside: bool
                      ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Clippe le cube par la SPHERE de rayon `radius` et renvoie (faces laterales,
    cordes du capuchon). On garde le morceau interieur (keep_inside) ou exterieur.

    Les sommets de coupe sont les VRAIES intersections arete <-> sphere,
    dedupliquees par ARETE DE GRILLE (cle = paire de coins de grille, partagee
    entre cubes voisins) : deux cubes adjacents lisent le meme point sur leur
    arete commune -> faces laterales coincidentes et interface CONFORME. En
    contrepartie, les sommets du capuchon ne sont pas coplanaires (la sphere
    bombe) : le chainage des cordes est triangule par _cap_faces.

    Un coin de grille pose SUR la sphere (a `tol` pres, cas frequent quand R
    tombe pile sur le pas) est garde par les DEUX morceaux et compte comme sommet
    de capuchon : on evite ainsi un point de coupe confondu avec ce coin (arete de
    longueur nulle). Les aretes ne sont coupees que si elles sont STRICTEMENT
    traversantes (les deux coins franchement de part et d'autre)."""
    tol = 1e-9 * float(np.linalg.norm(pos[1] - pos[0]))   # ~ 1e-9 x cote du cube
    sdist = [float(np.sqrt(np.dot(p, p))) - radius for p in pos]   # r - radius

    def edge_point(a: int, c: int) -> int:
        key = ("sph", radius, frozenset((keys[a], keys[c])))
        idx = b.point_id.get(key)
        if idx is not None:
            return idx
        return b.get_point(key, _sphere_edge_pos(pos[a], pos[c], radius))

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
            keep_a = on_a or (sa < -tol if keep_inside else sa > tol)
            if keep_a:
                pid = b.get_point(keys[a], pos[a])
                poly.append(pid)
                if on_a:
                    cap_pts.append(pid)         # coin pose sur la sphere = cap
            # Arete STRICTEMENT traversante (coins franchement de part et d'autre).
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
            raise ValueError("face de cube traversee par la sphere en plus de "
                             "2 points (cas degenere) : essayez un N plus grand.")

    return side_faces, chords


def _cap_faces(b: _Builder, chords: List[Tuple[int, int]], outward: bool
               ) -> Optional[List[List[int]]]:
    """Triangule le capuchon en eventail (fan) SANS point ajoute et renvoie la
    liste des triangles (ou None si capuchon vide) : un capuchon a p sommets donne
    p-2 triangles.

    Les sommets du ring sont des points sphere PARTAGES entre cubes voisins, donc
    le capuchon reste conforme. Le fan est ancre sur le sommet de plus PETIT index
    du ring : la triangulation devient canonique (independante du sens de parcours
    du ring), si bien que les deux demi-mailles d'un meme cube (R1) produisent
    EXACTEMENT les memes triangles -> capuchon partage (face interieure, G4).
    Chaque triangle est trivialement plan (critere G8). Si outward, chaque
    triangle est oriente normale sortante (vers +r) pour une face de BORD (R2) ;
    pour une face INTERIEURE (R1) l'orientation est indifferente (G9 la deduit de
    la geometrie)."""
    if not chords:
        return None
    ring = _order_ring(chords)
    if len(ring) < 3:
        return None
    a0 = min(range(len(ring)), key=lambda i: ring[i])   # apex = plus petit index
    seq = ring[a0:] + ring[:a0]                          # ring ancre sur l'apex
    apex = seq[0]

    tris: List[List[int]] = []
    for i in range(1, len(seq) - 1):
        tri = [apex, seq[i], seq[i + 1]]
        if outward:
            p0, p1, p2 = (np.asarray(b.points[n], dtype=float) for n in tri)
            normal = np.cross(p1 - p0, p2 - p0)
            tcen = (p0 + p1 + p2) / 3.0
            if float(np.dot(normal, tcen - CENTER)) < 0.0:
                tri = [tri[1], tri[0], tri[2]]
        tris.append(tri)
    return tris


# ---------------------------------------------------------------------------
# Construction complete
# ---------------------------------------------------------------------------
def build_sphere_mesh(r1: float, r2: float, n: int, eps: float = 0.0) -> nm.Mesh:
    """Construit le maillage cut-cell (coupe par la sphere, capuchon triangule) de
    la boule r2 (noyau r1)."""
    if not (0.0 < r1 < r2):
        raise ValueError("on attend 0 < R1 < R2.")
    if n < 1:
        raise ValueError("N doit etre >= 1.")
    if not (0.0 <= eps <= 0.5):
        raise ValueError("eps doit etre dans [0, 0.5].")

    c = 2.0 * r2 / n
    v_cube = c ** 3                         # volume d'un cube plein
    v_min = eps * v_cube                    # volume en deca duquel on ne coupe pas
    b = _Builder()

    def corner_pos(key: Tuple[int, int, int]) -> np.ndarray:
        return CENTER + (-r2 + np.asarray(key, dtype=float) * c)

    eta = 1e-9 * c                          # tolerance "coin pose sur la sphere"

    def counts(pos: List[np.ndarray], radius: float
               ) -> Tuple[List[bool], int, int]:
        """Renvoie (inside_strict, nb_inside_strict, nb_outside_strict) pour la
        sphere `radius`, avec une tolerance eta : un coin a moins de eta de la
        sphere (cas frequent quand R tombe pile sur le pas de grille) n'est compte
        NI dedans NI dehors. Il ne declenche donc pas a lui seul une coupe -- sans
        cela un coin pose sur la sphere, vu "dedans" par le bruit flottant,
        provoquerait une coupe degeneree de volume nul."""
        inside = []
        nin = nout = 0
        for p in pos:
            rn = float(np.sqrt(np.dot(p, p)))
            ins = rn < radius - eta
            inside.append(ins)
            if ins:
                nin += 1
            elif rn > radius + eta:
                nout += 1
        return inside, nin, nout

    def add_full_cube(keys, pos, material):
        """Ajoute le cube entier (8 coins, 6 faces) avec le milieu donne."""
        faces = [[b.get_point(keys[a], pos[a]) for a in face]
                 for face in CUBE_FACES]
        b.add_cell(faces, material=material)

    def cut_cell(cube_tag, keys, pos, radius, material, outward_cap):
        side, chords = _clip_cube_sphere(b, keys, pos, radius, keep_inside=True)
        cap = _cap_faces(b, chords, outward=outward_cap)
        if cap is None or len(side) < 3:
            raise ValueError(f"coupe degeneree sur le cube {cube_tag} : "
                             "essayez un N plus grand.")
        b.add_cell(side + cap, material=material)

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
                    plane = _fit_cut_plane(pos, in2, r2)     # plan secant : estimation eps
                    v_in = _half_volume_below(pos, *plane)   # cote interieur (garde)
                    v_out = v_cube - v_in                    # cote exterieur (jete)
                    if v_in <= v_min:
                        continue                   # capuchon interieur negligeable -> jete
                    if v_out <= v_min:
                        add_full_cube(keys, pos, material=2)  # sliver dehors -> cube plein
                        continue
                    cut_cell(("R2", i, j, k), keys, pos, r2,
                             material=2, outward_cap=True)
                    continue

                if cut1:                           # coupe par l'interface R1
                    plane = _fit_cut_plane(pos, in1, r1)     # plan secant : estimation eps
                    v_in = _half_volume_below(pos, *plane)   # noyau (r < R1)
                    v_out = v_cube - v_in                    # coquille (R1 < r < R2)
                    if min(v_in, v_out) <= v_min:
                        # un morceau negligeable -> pas de coupe, milieu du plus gros.
                        add_full_cube(keys, pos, material=1 if v_in >= v_out else 2)
                        continue
                    in_side, ch_in = _clip_cube_sphere(b, keys, pos, r1,
                                                       keep_inside=True)
                    out_side, ch_out = _clip_cube_sphere(b, keys, pos, r1,
                                                         keep_inside=False)
                    # Memes cordes des deux cotes + apex canonique (plus petit
                    # index) -> memes triangles -> capuchon partage (face int., G4).
                    cap_in = _cap_faces(b, ch_in, outward=False)
                    cap_out = _cap_faces(b, ch_out, outward=False)
                    if cap_in is None or cap_out is None:
                        raise ValueError(f"coupe R1 degeneree sur le cube "
                                         f"({i},{j},{k}) : essayez un N plus grand.")
                    b.add_cell(in_side + cap_in, material=1)    # noyau (r < R1)
                    b.add_cell(out_side + cap_out, material=2)  # coquille
                    continue

                # Maille pleine : noyau si aucun coin hors de R1, coquille sinon.
                add_full_cube(keys, pos, material=1 if nout1 == 0 else 2)

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
