"""
view.py
=======

Visualisation d'une geometrie NYMO. Regroupe deux modes complementaires
derriere une seule commande :

    3d     : vue 3D de la peau exterieure, coloriee par milieu. Option de coupe
             (--clip) par retrait de mailles d'un cote d'un plan axe, ce qui
             revele l'interieur "en escalier".
    coupe  : coupe PLANE (section) par un plan axe. Pour chaque maille, le
             polygone d'intersection avec le plan est trace dans le plan des
             deux axes restants, colorie par milieu.

Usage :
    python view.py 3d    fichier.geo [--interior] [--clip x=0] [--save img.png]
    python view.py coupe fichier.geo x=1.5 [--interior] [--eps 1e-9] [--save img.png]

Conventions communes aux deux modes :
    - axes : x=0, y=1, z=2 ;
    - milieux coloriés via la palette tab10 (memes couleurs partout) ;
    - --interior trace le decoupage en mailles (fil de fer / contours) ;
    - --save enregistre un PNG au lieu d'ouvrir une fenetre interactive.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import nymo_mesh as nm

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}

# Tolerance geometrique pour la coupe plane (distance au plan / aire degeneree).
EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Couleurs (partagees par les deux modes)
# ---------------------------------------------------------------------------
def material_colors(materials: List[int]) -> Dict[int, tuple]:
    """Associe une couleur a chaque id de milieu present (palette tab10)."""
    distinct = sorted(set(materials))
    cmap = plt.get_cmap("tab10")
    return {mat: cmap(i % 10) for i, mat in enumerate(distinct)}


def material_legend(colors: Dict[int, tuple], present: List[int]) -> List[Line2D]:
    """Construit les poignees de legende 'milieu N' pour les milieux presents.

    `present` peut etre une liste par-polygone (avec doublons) : on deduplique
    pour n'avoir qu'une poignee par milieu."""
    return [Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   markerfacecolor=colors[m], markeredgecolor="k",
                   label=f"milieu {m}") for m in sorted(set(present))]


def face_polygon(mesh: nm.Mesh, face: int) -> np.ndarray:
    """Coordonnees (n_noeuds, 3) du polygone d'une face."""
    return mesh.points[mesh.faces[face]]


def finish(fig, save: Optional[str]) -> None:
    """Enregistre en PNG ou ouvre la fenetre interactive."""
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Image enregistree : {save}")
    else:
        plt.show()


# ===========================================================================
# MODE 3D : peau exterieure, avec coupe optionnelle par retrait de mailles
# ===========================================================================
def boundary_faces_by_material(mesh: nm.Mesh, f2c: Dict[int, List[int]]):
    """Renvoie (polygones, materiaux) des faces de bord uniquement."""
    polys, mats = [], []
    for face in range(mesh.nb_faces):
        if nm.is_boundary_face(f2c, face):
            owner = f2c[face][0]
            polys.append(face_polygon(mesh, face))
            mats.append(mesh.materials[owner])
    return polys, mats


def interior_edges(mesh: nm.Mesh, f2c: Dict[int, List[int]]) -> List[np.ndarray]:
    """Segments (paires de points) des aretes des faces interieures."""
    segments = []
    for face in range(mesh.nb_faces):
        if nm.is_boundary_face(f2c, face):
            continue
        nodes = mesh.faces[face]
        for j in range(len(nodes)):
            a, b = nodes[j], nodes[(j + 1) % len(nodes)]
            segments.append(np.array([mesh.points[a], mesh.points[b]]))
    return segments


def parse_clip(spec: str) -> Tuple[int, float, Callable[[float], bool], str]:
    """'x=0' / 'x<0' / 'z>=1.5' -> (axe, valeur, predicat_garde, op_affiche).

    L'operateur porte sur la coordonnee du barycentre de la maille selon l'axe.
    '=' est interprete comme '<=' (on garde la moitie inferieure).
    """
    s = spec.replace(" ", "").lower()
    op = next((o for o in ("<=", ">=", "<", ">", "=") if o in s), None)
    if op is None:
        raise ValueError(f"Coupe invalide '{spec}' : attendu p.ex. x=0, x<0, z>=1.5")
    axis_char, _, val = s.partition(op)
    if axis_char not in AXIS_INDEX:
        raise ValueError(f"Axe inconnu '{axis_char}' (attendu x, y ou z)")
    try:
        value = float(val)
    except ValueError:
        raise ValueError(f"Valeur de coupe illisible : '{val}'")
    axis = AXIS_INDEX[axis_char]
    if op == "<":
        return axis, value, (lambda c: c < value), "<"
    if op in ("<=", "="):
        return axis, value, (lambda c: c <= value), "≤"   # <=
    if op == ">":
        return axis, value, (lambda c: c > value), ">"
    return axis, value, (lambda c: c >= value), "≥"        # >=


def cell_centroid(mesh: nm.Mesh, cell: int) -> np.ndarray:
    """Barycentre (moyenne des sommets) d'une maille."""
    return mesh.points[nm.cell_node_ids(mesh, cell)].mean(axis=0)


def clipped_skin_by_material(
    mesh: nm.Mesh,
    f2c: Dict[int, List[int]],
    axis: int,
    keep: Callable[[float], bool],
):
    """Peau de la moitie conservee : faces dont UN SEUL voisin est garde.

    Renvoie (polygones, materiaux, mailles_gardees). Une face est tracee si
    exactement une de ses mailles adjacentes est conservee ; cela reunit les
    faces de bord d'origine du cote garde ET les faces interieures exposees par
    la coupe (la "surface de coupe"). Le milieu utilise est celui de la maille
    conservee situee derriere la face.
    """
    kept = {c for c in range(mesh.nb_cells)
            if keep(float(cell_centroid(mesh, c)[axis]))}
    polys, mats = [], []
    for face in range(mesh.nb_faces):
        kept_owners = [c for c in f2c[face] if c in kept]
        if len(kept_owners) == 1:
            polys.append(face_polygon(mesh, face))
            mats.append(mesh.materials[kept_owners[0]])
    return polys, mats, kept


def kept_interior_edges(
    mesh: nm.Mesh, f2c: Dict[int, List[int]], kept: set
) -> List[np.ndarray]:
    """Aretes des faces interieures a la moitie conservee (2 voisins gardes)."""
    segments = []
    for face in range(mesh.nb_faces):
        owners = f2c[face]
        if len(owners) == 2 and owners[0] in kept and owners[1] in kept:
            nodes = mesh.faces[face]
            for j in range(len(nodes)):
                a, b = nodes[j], nodes[(j + 1) % len(nodes)]
                segments.append(np.array([mesh.points[a], mesh.points[b]]))
    return segments


def set_equal_aspect(ax, points: np.ndarray) -> None:
    """Cadrage cubique (aspect 1:1:1) autour du nuage de points."""
    mins, maxs = points.min(axis=0), points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_mesh(
    mesh: nm.Mesh,
    show_interior: bool = False,
    save: Optional[str] = None,
    clip: Optional[Tuple[int, float, Callable[[float], bool], str]] = None,
) -> None:
    f2c = nm.build_face_to_cells(mesh)
    colors = material_colors(mesh.materials)

    if clip is None:
        polys, mats = boundary_faces_by_material(mesh, f2c)
        kept = None
        title = (f"{mesh.nb_cells} mailles | {mesh.nb_faces} faces | "
                 f"peau exterieure coloriee par milieu")
    else:
        axis, value, keep, op = clip
        polys, mats, kept = clipped_skin_by_material(mesh, f2c, axis, keep)
        if not polys:
            lo, hi = mesh.points[:, axis].min(), mesh.points[:, axis].max()
            print(f"Coupe {AXIS_NAME[axis]}{op}{value:g} : aucune maille gardee "
                  f"(domaine {AXIS_NAME[axis]} in [{lo:.4g}, {hi:.4g}]).")
            return
        title = (f"{len(kept)}/{mesh.nb_cells} mailles | "
                 f"coupe {AXIS_NAME[axis]} {op} {value:g} | "
                 f"interieur colorie par milieu")

    facecolors = [colors[m] for m in mats]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    skin = Poly3DCollection(polys, facecolors=facecolors,
                            edgecolors="k", linewidths=0.4, alpha=0.9)
    ax.add_collection3d(skin)

    if show_interior:
        segs = (kept_interior_edges(mesh, f2c, kept) if kept is not None
                else interior_edges(mesh, f2c))
        ax.add_collection3d(Line3DCollection(
            segs, colors="0.5", linewidths=0.5, linestyles=":"))

    # Cadrage : sur la moitie visible si coupe, sinon tout le maillage.
    framing = np.vstack(polys) if clip is not None else mesh.points
    set_equal_aspect(ax, framing)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(handles=material_legend(colors, mats), loc="upper left")

    finish(fig, save)


# ===========================================================================
# MODE COUPE : section plane, polygone d'intersection maille / plan
# ===========================================================================
def parse_plane(spec: str) -> Tuple[int, float]:
    """'x=0' / 'y=6' / 'z = -2' -> (axe, valeur). axe : 0=x, 1=y, 2=z."""
    s = spec.replace(" ", "").lower()
    if "=" not in s:
        raise ValueError(f"Plan invalide '{spec}' : forme attendue x=..., y=..., z=...")
    axis_char, _, val = s.partition("=")
    if axis_char not in AXIS_INDEX:
        raise ValueError(f"Axe inconnu '{axis_char}' (attendu x, y ou z)")
    try:
        value = float(val)
    except ValueError:
        raise ValueError(f"Valeur de coupe illisible : '{val}'")
    return AXIS_INDEX[axis_char], value


def cell_edges(mesh: nm.Mesh, cell: int) -> set:
    """Ensemble des aretes (frozenset de 2 noeuds) d'une maille."""
    edges = set()
    for f in mesh.cells[cell]:
        nodes = mesh.faces[f]
        m = len(nodes)
        for j in range(m):
            edges.add(frozenset((nodes[j], nodes[(j + 1) % m])))
    return edges


def _dedupe(pts: np.ndarray, tol: float) -> np.ndarray:
    """Supprime les points doublons (a tol pres)."""
    kept: List[np.ndarray] = []
    for p in pts:
        if not any(np.linalg.norm(p - q) <= tol for q in kept):
            kept.append(p)
    return np.asarray(kept)


def _polygon_area_2d(poly: np.ndarray) -> float:
    """Aire (valeur absolue) d'un polygone 2D, formule du lacet."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def section_polygon(
    mesh: nm.Mesh, cell: int, axis: int, value: float, eps: float = EPS
) -> Optional[np.ndarray]:
    """Polygone 2D (projete) de l'intersection d'une maille avec le plan, ou None.

    Le polygone est exprime dans les deux axes restants. Renvoie None si la
    maille ne coupe pas le plan, ou ne le touche qu'en un point / une arete
    (intersection d'aire nulle).

    HYPOTHESE : mailles CONVEXES (tetraedres ou bipyramides). L'intersection
    d'un polyedre convexe par un plan est un polygone convexe : on ordonne donc
    les points de percee par angle polaire autour de leur barycentre.
    """
    pts = mesh.points
    nodes = nm.cell_node_ids(mesh, cell)
    # Distance signee de chaque noeud au plan (coordonnee selon l'axe - valeur).
    dist = {n: float(pts[n][axis] - value) for n in nodes}

    crossings: List[np.ndarray] = []

    # (1) Sommets situes SUR le plan.
    for n in nodes:
        if abs(dist[n]) <= eps:
            crossings.append(pts[n])

    # (2) Aretes traversant strictement le plan (extremites de signes opposes).
    for edge in cell_edges(mesh, cell):
        a, b = tuple(edge)
        da, db = dist[a], dist[b]
        if abs(da) <= eps or abs(db) <= eps:
            continue                       # extremite sur le plan : deja traitee
        if da * db < 0.0:
            t = da / (da - db)             # da + t*(db-da) = 0
            crossings.append(pts[a] + t * (pts[b] - pts[a]))

    if len(crossings) < 3:
        return None

    P = _dedupe(np.asarray(crossings), eps)
    if len(P) < 3:
        return None

    # Projection : on retire la coordonnee de l'axe de coupe.
    others = [i for i in range(3) if i != axis]
    Q = P[:, others]

    # Ordre angulaire autour du barycentre (valide car coupe convexe).
    c = Q.mean(axis=0)
    ang = np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0])
    poly = Q[np.argsort(ang)]

    if _polygon_area_2d(poly) <= eps:
        return None                        # tangence (point ou arete) : ignoree
    return poly


def plot_section(
    mesh: nm.Mesh,
    axis: int,
    value: float,
    eps: float = EPS,
    show_interior: bool = False,
    save: Optional[str] = None,
) -> int:
    """Construit et affiche la coupe. Renvoie le nombre de mailles coupees."""
    colors = material_colors(mesh.materials)
    others = [i for i in range(3) if i != axis]
    u_name, v_name = AXIS_NAME[others[0]], AXIS_NAME[others[1]]

    polys: List[np.ndarray] = []
    facecolors: List[tuple] = []
    per_mat: Dict[int, int] = {}

    for c in range(mesh.nb_cells):
        poly = section_polygon(mesh, c, axis, value, eps)
        if poly is None:
            continue
        mat = mesh.materials[c]
        polys.append(poly)
        facecolors.append(colors[mat])
        per_mat[mat] = per_mat.get(mat, 0) + 1

    if not polys:
        lo, hi = mesh.points[:, axis].min(), mesh.points[:, axis].max()
        print(f"Aucune maille coupee par le plan {AXIS_NAME[axis]}={value:g}. "
              f"(domaine {AXIS_NAME[axis]} in [{lo:.4g}, {hi:.4g}])")
        return 0

    fig, ax = plt.subplots(figsize=(8, 7))
    coll = PolyCollection(polys, facecolors=facecolors,
                          edgecolors="k", linewidths=0.4, alpha=0.85)
    ax.add_collection(coll)

    if show_interior:
        # Contour de chaque polygone en trait fin (visualise le decoupage).
        segs = []
        for poly in polys:
            closed = np.vstack([poly, poly[0]])
            segs.extend([closed[i:i + 2] for i in range(len(poly))])
        ax.add_collection(LineCollection(segs, colors="0.3",
                                         linewidths=0.4, linestyles=":"))

    allpts = np.vstack(polys)
    ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
    ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(u_name)
    ax.set_ylabel(v_name)
    ax.set_title(f"Coupe {AXIS_NAME[axis]} = {value:g}  |  "
                 f"{len(polys)} mailles coupees  |  plan ({u_name}, {v_name})")
    ax.legend(handles=material_legend(colors, list(per_mat)), loc="upper left")

    finish(fig, save)

    detail = ", ".join(f"milieu {m} : {per_mat[m]}" for m in sorted(per_mat))
    print(f"Coupe {AXIS_NAME[axis]}={value:g} : {len(polys)} mailles coupees ({detail})")
    return len(polys)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualisation d'une geometrie NYMO (vue 3D ou coupe plane).")
    sub = parser.add_subparsers(dest="mode", required=True)

    p3d = sub.add_parser("3d", help="vue 3D de la peau, coloriee par milieu")
    p3d.add_argument("geo", help="fichier geometrie .geo / .mesh3D")
    p3d.add_argument("--interior", action="store_true",
                     help="tracer aussi les aretes interieures")
    p3d.add_argument("--clip", default=None, metavar="PLAN",
                     help="coupe axe revelant l'interieur : 'x=0' (garde x<=0), "
                          "'x>0', 'z<=1.5' ... operateurs <, <=, >, >=, = acceptes")
    p3d.add_argument("--save", default=None,
                     help="enregistrer en PNG au lieu d'afficher")

    pc = sub.add_parser("coupe", help="coupe plane (section) coloriee par milieu")
    pc.add_argument("geo", help="fichier geometrie .geo / .mesh3D")
    pc.add_argument("plan", help="plan de coupe axe : x=..., y=..., z=...")
    pc.add_argument("--eps", type=float, default=EPS,
                    help=f"tolerance geometrique (defaut {EPS:g})")
    pc.add_argument("--interior", action="store_true",
                    help="trace aussi le contour de chaque maille coupee")
    pc.add_argument("--save", default=None,
                    help="enregistrer en PNG au lieu d'afficher")

    args = parser.parse_args(argv)
    mesh = nm.read_mesh(args.geo)

    if args.mode == "3d":
        clip = parse_clip(args.clip) if args.clip else None
        plot_mesh(mesh, show_interior=args.interior, save=args.save, clip=clip)
    else:
        axis, value = parse_plane(args.plan)
        plot_section(mesh, axis, value, eps=args.eps,
                     show_interior=args.interior, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
