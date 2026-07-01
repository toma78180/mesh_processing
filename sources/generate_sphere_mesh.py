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

Option eps : SNAPPING par coin (en fraction du cote c). Chaque coin de grille est
classe DEDANS / DEHORS / SUR l'interface : il est SUR si sa distance radiale a la
sphere est <= eps*c. Un coin SUR est mis exactement sur l'interface (partage par
les deux morceaux), et ses aretes ne portent plus de coupe distincte. On supprime
ainsi les micro-faces / regions tres fines le long de l'interface, au prix d'une
legere approximation (l'interface passe par des coins de grille).

La decision est prise PAR COIN (uniquement son rayon), donc identique pour tous
les cubes voisins ET pour les trois aretes d'un meme coin : pas d'incoherence (un
coin n'est jamais "rabattu" sur une arete et "bord franc" sur une autre). C'est ce
qui garantit la validite a tout eps -- une decision par arete, elle, produit des
replis (triangle de cap plat sur un mur) et des colinearites. La conformite est
preservee (coins et coupes franches partages par cle de grille).
    eps = 0   -> pas de snapping : resolution maximale ;
    eps grand -> de plus en plus de coins mis SUR l'interface : interface plus
                 "en marches", nettement moins de micro-mailles (eps <= 0.5).

Usage :
    python sources/generate_sphere_mesh.py R1 R2 N out.mesh3D [eps]
"""

from __future__ import annotations

import os
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
# Geometrie : intersection arete <-> sphere
# ---------------------------------------------------------------------------
def _sphere_edge_pos(pa: np.ndarray, pb: np.ndarray, radius: float) -> np.ndarray:
    """Point d'intersection de l'arete (a, b) avec la sphere centree a l'origine
    (sommet de coupe partage entre cubes voisins, cf. _clip_cube_sphere)."""
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


def _order_ring(chords: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Ordonne les sommets du capuchon en chainant les cordes (cycle unique).
    Renvoie None si le chainage est malforme (cycle non simple ou non ferme) : le
    capuchon est alors degenere et l'appelant abandonne / remplit la maille."""
    adj: Dict[int, List[int]] = {}
    for a, c in chords:
        adj.setdefault(a, []).append(c)
        adj.setdefault(c, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None                            # cycle non simple
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
            return None                        # cycle non ferme
    return ring


def _clip_cube_sphere(b: _Builder, keys: List[object], pos: List[np.ndarray],
                      radius: float, keep_inside: bool, tol: float, snap: float
                      ) -> Tuple[Optional[List[List[int]]],
                                 Optional[List[Tuple[int, int]]],
                                 Optional[set]]:
    """Clippe le cube par la SPHERE de rayon `radius` et renvoie (faces laterales,
    cordes du capuchon). On garde le morceau interieur (keep_inside) ou exterieur.

    Les sommets de coupe sont les VRAIES intersections arete <-> sphere,
    dedupliquees par ARETE DE GRILLE (cle = paire de coins de grille, partagee
    entre cubes voisins) : deux cubes adjacents lisent le meme point sur leur
    arete commune -> faces laterales coincidentes et interface CONFORME. En
    contrepartie, les sommets du capuchon ne sont pas coplanaires (la sphere
    bombe) : le chainage des cordes est triangule par _cap_faces.

    SNAPPING (eps) -- classification PAR COIN : chaque coin est DEDANS, DEHORS, ou
    SUR l'interface si sa distance radiale a la sphere est <= la bande
    `band = max(snap, tol)` (snap = eps*c). Un coin SUR est garde par les DEUX
    morceaux et sert de sommet de capuchon. Une arete n'est coupee FRANCHEMENT que
    si elle relie un coin strictement DEDANS a un coin strictement DEHORS ; une
    arete touchant un coin SUR n'a pas de coupe distincte (le bord est au coin).
    La decision ne depend que du COIN (son rayon), identique pour tous ses voisins
    et toutes ses aretes -> conforme et coherente (plus de coin "tantot rabattu,
    tantot bord franc", source des replis et colinearites). Au plancher (eps=0,
    band=tol) seul un coin exactement sur la sphere est SUR."""
    band = max(snap, tol)
    sdist = [float(np.sqrt(np.dot(p, p))) - radius for p in pos]   # r - radius
    on_ids: set = set()                         # ids des coins SUR l'interface

    def is_on(x: int) -> bool:
        return abs(sdist[x]) <= band
    def kept(x: int) -> bool:                   # garde-t-on x dans ce morceau ?
        return sdist[x] <= band if keep_inside else sdist[x] >= -band

    def edge_point(a: int, c: int) -> int:      # coupe FRANCHE arete <-> sphere
        key = ("sph", radius, frozenset((keys[a], keys[c])))
        idx = b.point_id.get(key)
        if idx is not None:
            return idx
        return b.get_point(key, _sphere_edge_pos(pos[a], pos[c], radius))

    side_faces: List[List[int]] = []
    chords: List[Tuple[int, int]] = []

    for face in CUBE_FACES:
        poly: List[int] = []
        trans: List[int] = []                   # points de bord (entree/sortie)
        m = len(face)
        for ai in range(m):
            a = face[ai]
            c = face[(ai + 1) % m]
            if kept(a):
                pid = b.get_point(keys[a], pos[a])
                poly.append(pid)
                if is_on(a):
                    on_ids.add(pid)
            if kept(a) != kept(c):              # transition garde <-> non garde
                if is_on(a):                    # bord = coin SUR (deja dans poly)
                    bp = b.get_point(keys[a], pos[a])
                elif is_on(c):                  # bord = coin SUR (ajoute au tour suivant)
                    bp = b.get_point(keys[c], pos[c])
                else:                           # arete franche DEDANS <-> DEHORS
                    bp = edge_point(a, c)
                    poly.append(bp)
                trans.append(bp)

        poly = _dedup_cycle(poly)
        if len(poly) >= 3:
            side_faces.append(poly)

        tu = list(dict.fromkeys(trans))         # 2 points de bord -> une corde
        if len(tu) == 2:
            chords.append((tu[0], tu[1]))
        elif len(tu) > 2:
            return None, None, None            # face coupee en >2 points : degenere

    return side_faces, chords, on_ids


def _cap_faces(b: _Builder, chords: List[Tuple[int, int]], on_ids: set
               ) -> Optional[List[List[int]]]:
    """Triangule le capuchon en eventail (fan) SANS point ajoute et renvoie la
    liste des triangles (ou None si capuchon vide / degenere) : un capuchon a p
    sommets donne p-2 triangles.

    Les sommets du ring sont des points sphere PARTAGES entre cubes voisins, donc
    le capuchon reste conforme. Le fan est ancre sur un sommet qui n'est PAS un
    coin SUR l'interface (`on_ids`) -- c.-a-d. un vrai point de coupe d'arete, hors
    des murs du cube -- de plus petit index : ainsi chaque triangle contient ce
    sommet et aucun triangle ne tombe a plat sur un mur du cube (ce qui creait des
    replis : deux mailles du meme cote d'une face). Le choix de l'apex est
    canonique (meme ensemble de sommets vu des deux cotes), donc les deux
    demi-mailles produisent EXACTEMENT les memes triangles -> capuchon partage
    (face interieure, G4). Chaque triangle est plan (G8).

    Si TOUS les sommets du capuchon sont des coins SUR l'interface (`on_ids`), le
    capuchon est entierement plaque sur la grille : le morceau est une dalle
    degeneree -> on renvoie None et l'appelant le jette / remplit."""
    if not chords:
        return None
    ring = _order_ring(chords)
    if ring is None or len(ring) < 3:
        return None
    # Apex du fan : de preference un VRAI point de coupe (hors `on_ids`, donc hors
    # des murs du cube) pour qu'aucun triangle ne tombe a plat sur un mur (repli).
    # Si le ring n'a que des coins SUR (interface alignee grille), on eventaille
    # quand meme depuis le plus petit (capuchon ferme, sinon on creerait un trou) ;
    # ces coins etant ~sur la sphere, les triangles ne sont en general pas plats.
    cand = [v for v in ring if v not in on_ids]
    apex = min(cand) if cand else min(ring)
    i0 = ring.index(apex)
    seq = ring[i0:] + ring[:i0]
    return [[apex, seq[i], seq[i + 1]] for i in range(1, len(seq) - 1)]


def _drop_orphan_points(mesh: nm.Mesh) -> None:
    """Supprime les noeuds n'appartenant a aucune face -- des points orphelins
    peuvent rester apres l'abandon d'un morceau degenere (le point avait ete
    alloue avant le rejet de la maille) -- et renumerote en consequence (G7)."""
    used = sorted({n for f in mesh.faces for n in f})
    if len(used) == mesh.nb_points:
        return
    remap = {old: new for new, old in enumerate(used)}
    mesh.points = mesh.points[used]
    mesh.faces = [[remap[n] for n in f] for f in mesh.faces]


def _orient_boundary_faces(mesh: nm.Mesh) -> None:
    """Oriente chaque face de BORD normale sortante (convention NYMO : la maille
    voisine doit etre du cote oppose a la normale, cf. validate G9). Les faces
    interieures (2 voisines) sont laissees telles quelles (G9 y est automatique).

    Passe globale et robuste : elle rend l'orientation independante des aleas du
    snapping (capuchons de bord dont le sens de winding peut varier d'une maille a
    l'autre)."""
    f2c = nm.build_face_to_cells(mesh)
    bary = [nm.cell_barycenter(mesh, c) for c in range(mesh.nb_cells)]
    for f in range(mesh.nb_faces):
        owners = f2c[f]
        if len(owners) != 1:
            continue
        nodes = mesh.faces[f]
        if len(nodes) < 3:
            continue
        normal = nm.face_normal(mesh, f)
        p0 = mesh.points[nodes[0]]
        if float(np.dot(normal, bary[owners[0]] - p0)) > 0.0:
            mesh.faces[f] = nodes[::-1]        # normale vers l'interieur -> on retourne


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
    tol = 1e-9 * c          # tolerance numerique : coin (quasi) exactement sur la sphere
    snap = eps * c          # distance de snapping de sommet (cf. edge_point / eps)
    b = _Builder()

    def corner_pos(key: Tuple[int, int, int]) -> np.ndarray:
        return CENTER + (-r2 + np.asarray(key, dtype=float) * c)

    def counts(pos: List[np.ndarray], radius: float) -> Tuple[int, int]:
        """Renvoie (nb_inside_strict, nb_outside_strict) pour la sphere `radius`,
        avec la tolerance numerique `tol` : un coin a moins de tol de la sphere
        (cas frequent quand R tombe pile sur le pas) n'est compte NI dedans NI
        dehors. Il ne declenche donc pas a lui seul une coupe degeneree."""
        nin = nout = 0
        for p in pos:
            rn = float(np.sqrt(np.dot(p, p)))
            if rn < radius - tol:
                nin += 1
            elif rn > radius + tol:
                nout += 1
        return nin, nout

    def add_full_cube(keys, pos, material):
        """Ajoute le cube entier (8 coins, 6 faces) avec le milieu donne."""
        faces = [[b.get_point(keys[a], pos[a]) for a in face]
                 for face in CUBE_FACES]
        b.add_cell(faces, material=material)

    stats = {"drop": 0, "fill": 0}        # mailles jetees / remplies par le snapping

    def clip_piece(keys, pos, radius, keep_inside):
        """Clippe un morceau (interieur si keep_inside, exterieur sinon) et
        triangule son capuchon. Renvoie (faces_laterales, triangles_capuchon), ou
        (None, None) si la coupe est DEGENEREE -- typiquement quand le snapping
        (eps) ecrase le capuchon d'un coin isole : le morceau est alors negligeable
        et l'appelant le jette (R2) ou remplit le cube du milieu dominant (R1)."""
        side, chords, on_ids = _clip_cube_sphere(
            b, keys, pos, radius, keep_inside=keep_inside, tol=tol, snap=snap)
        if side is None or len(side) < 3:
            return None, None
        if not chords:
            # Aucune transition garde/non-garde : l'interface ne traverse pas ce
            # morceau (tous les coins gardes, p.ex. tous dans la bande). C'est le
            # CUBE PLEIN, pas une dalle -> on l'ajoute tel quel, sans capuchon.
            return side, []
        cap = _cap_faces(b, chords, on_ids)
        if cap is None:
            return None, None
        return side, cap

    for i in range(n):
        for j in range(n):
            for k in range(n):
                keys = [(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS]
                pos = [corner_pos(key) for key in keys]
                nin1, nout1 = counts(pos, r1)
                nin2, nout2 = counts(pos, r2)

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
                    side, cap = clip_piece(keys, pos, r2, keep_inside=True)
                    if side is None:
                        stats["drop"] += 1         # interieur ecrase -> jete
                    else:
                        b.add_cell(side + cap, material=2)
                    continue

                if cut1:                           # coupe par l'interface R1
                    # Memes cordes des deux cotes + apex canonique -> memes triangles
                    # de capuchon -> face d'interface partagee (interieure, G4).
                    in_side, cap_in = clip_piece(keys, pos, r1, keep_inside=True)
                    out_side, cap_out = clip_piece(keys, pos, r1, keep_inside=False)
                    if in_side is None or out_side is None:
                        # un morceau ecrase par le snapping -> cube plein du dominant.
                        stats["fill"] += 1
                        add_full_cube(keys, pos, material=1 if nin1 >= nout1 else 2)
                        continue
                    b.add_cell(in_side + cap_in, material=1)    # noyau (r < R1)
                    b.add_cell(out_side + cap_out, material=2)  # coquille
                    continue

                # Maille pleine : noyau si aucun coin hors de R1, coquille sinon.
                add_full_cube(keys, pos, material=1 if nout1 == 0 else 2)

    if stats["drop"] or stats["fill"]:
        print(f"  snapping (eps={eps:g}) : {stats['drop']} maille(s) jetee(s), "
              f"{stats['fill']} remplie(s) du milieu dominant")

    mesh = b.mesh(name=f"sphere_R1_{r1:g}_R2_{r2:g}_N{n}")
    _drop_orphan_points(mesh)                   # noeuds sans face (morceaux abandonnes) -> G7
    _orient_boundary_faces(mesh)               # faces de bord normale sortante (G9)
    return mesh


# ---------------------------------------------------------------------------
# Equivalence des mailles PAR TRANSLATION SEULE
# ---------------------------------------------------------------------------
def _region_signature(mesh: nm.Mesh, c: int, ndigits: int = 9) -> tuple:
    """Empreinte d'une maille invariante par translation (geometrie seule).

    On rapporte tous les sommets de la maille a son coin inferieur (min par
    composante), puis on construit une empreinte ordonnee de ses faces (chaque
    face = ensemble trie des coordonnees relatives de ses noeuds). Deux mailles
    ont la meme empreinte si, et seulement si, l'une est la TRANSLATEE exacte de
    l'autre (ni rotation ni symetrie ne sont reconnues). Le materiau est ignore.
    """
    node_set = set()
    for f in mesh.cells[c]:
        node_set.update(mesh.faces[f])
    ref = mesh.points[sorted(node_set)].min(axis=0)
    rel = {n: tuple(round(float(v), ndigits) for v in (mesh.points[n] - ref))
           for n in node_set}
    face_sigs = tuple(sorted(
        tuple(sorted(rel[n] for n in mesh.faces[f])) for f in mesh.cells[c]))
    return face_sigs


def region_translation_classes(mesh: nm.Mesh) -> int:
    """Nombre de classes d'equivalence de mailles modulo translation."""
    return len({_region_signature(mesh, c) for c in range(mesh.nb_cells)})


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
    # Le nom de la geometrie (bloc "Geometrie:") suit le fichier de sortie demande,
    # et non un nom fabrique, pour que l'aval ne cree pas un fichier d'un autre nom.
    mesh.name = os.path.splitext(os.path.basename(out))[0]
    nm.write_mesh(mesh, out)

    n1 = sum(1 for m in mesh.materials if m == 1)
    n2 = sum(1 for m in mesh.materials if m == 2)
    print(f"{out} : {mesh.nb_cells} mailles ({n1} noyau, {n2} coquille), "
          f"{mesh.nb_faces} faces, {mesh.nb_points} points "
          f"[R1={r1:g} R2={r2:g} N={n} eps={eps:g}]")

    # Equivalence par translation seule (geometrie, materiau ignore).
    nreg = mesh.nb_cells
    neq = region_translation_classes(mesh)
    ratio = neq / nreg if nreg else 0.0
    print(f"  regions : {nreg} ; equiregions (translation seule) : {neq} ; "
          f"equiregion/region : {ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
