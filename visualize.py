"""
visualize.py
============

Affichage 3D d'une geometrie NYMO. Les faces de BORD (peau exterieure du
maillage) sont coloriees par leur milieu : on voit donc directement la
repartition des materiaux exterieurs. Les aretes interieures peuvent etre
tracees en option pour visualiser le decoupage en mailles.

Option --clip : coupe le maillage par un plan axe et n'affiche que la moitie
CONSERVEE, ce qui revele l'INTERIEUR. Les faces exposees par la coupe sont
coloriees par le milieu de la maille situee derriere. La coupe se fait par
RETRAIT DE MAILLES (le critere porte sur le barycentre de chaque maille) : la
surface de coupe suit donc les faces des mailles -- elle est "en escalier" sur
un maillage grossier, quasi-plane sur un maillage fin.

Usage :
    python visualize.py fichier.geo [--interior] [--save image.png]
    python visualize.py fichier.geo --clip x=0           # garde x <= 0
    python visualize.py fichier.geo --clip "x>0"         # garde l'autre moitie
    python visualize.py fichier.geo --clip z=1.5 --interior

    --interior   trace aussi les aretes des faces interieures (fil de fer).
                 Avec --clip, restreint a la moitie conservee.
    --clip PLAN  coupe axe : x=..., y=..., z=... ; operateurs <, <=, >, >=, =
                 acceptes ('=' equivaut a <=, garde la moitie inferieure).
    --save PNG   enregistre l'image (sinon : fenetre interactive)
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import nymo_mesh as nm

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}


# ---------------------------------------------------------------------------
# Construction des donnees a tracer
# ---------------------------------------------------------------------------
def face_polygon(mesh: nm.Mesh, face: int) -> np.ndarray:
    """Coordonnees (n_noeuds, 3) du polygone d'une face."""
    return mesh.points[mesh.faces[face]]


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


# ---------------------------------------------------------------------------
# Coupe (clipping) : retrait des mailles d'un cote du plan
# ---------------------------------------------------------------------------
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
        return axis, value, (lambda c: c <= value), "\u2264"   # <=
    if op == ">":
        return axis, value, (lambda c: c > value), ">"
    return axis, value, (lambda c: c >= value), "\u2265"        # >=


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


# ---------------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------------
def material_colors(materials: List[int]) -> Dict[int, tuple]:
    """Associe une couleur a chaque id de milieu present."""
    distinct = sorted(set(materials))
    cmap = plt.get_cmap("tab10")
    return {mat: cmap(i % 10) for i, mat in enumerate(distinct)}


# ---------------------------------------------------------------------------
# Mise a l'echelle des axes (aspect cubique)
# ---------------------------------------------------------------------------
def set_equal_aspect(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
def plot_mesh(
    mesh: nm.Mesh,
    show_interior: bool = False,
    save: Optional[str] = None,
    clip: Optional[Tuple[int, float, Callable[[float], bool], str]] = None,
):
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

    legend = [Line2D([0], [0], marker="s", linestyle="", markersize=12,
                     markerfacecolor=colors[m], markeredgecolor="k",
                     label=f"milieu {m}") for m in sorted(colors)]
    ax.legend(handles=legend, loc="upper left")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Image enregistree : {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation 3D d'une geometrie NYMO")
    parser.add_argument("geo", help="fichier geometrie .geo")
    parser.add_argument("--interior", action="store_true",
                        help="tracer aussi les aretes interieures")
    parser.add_argument("--clip", default=None, metavar="PLAN",
                        help="coupe axe revelant l'interieur : 'x=0' (garde x<=0), "
                             "'x>0', 'z<=1.5' ... operateurs <, <=, >, >=, = acceptes")
    parser.add_argument("--save", default=None, help="enregistrer en PNG au lieu d'afficher")
    args = parser.parse_args()

    mesh = nm.read_mesh(args.geo)
    clip = parse_clip(args.clip) if args.clip else None
    plot_mesh(mesh, show_interior=args.interior, save=args.save, clip=clip)


if __name__ == "__main__":
    main()
