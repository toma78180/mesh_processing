"""
coupe.py
========

Affichage d'une COUPE plane d'une geometrie NYMO. Le plan de coupe est
axe (perpendiculaire a x, y ou z), specifie en ligne de commande sous la
forme "x=0", "y=6", "z=-2" (espaces tolerees : "z = -2").

Pour chaque maille, on calcule le polygone d'intersection avec le plan, puis
on le colorie selon le milieu de la maille. La coupe est dessinee dans le plan
des deux axes restants (p.ex. coupe x=cste -> trace dans le plan (y, z)).

HYPOTHESE : mailles CONVEXES (tetraedres ou bipyramides, comme produit par la
chaine NYMO). L'intersection d'un polyedre convexe par un plan est un polygone
convexe : on ordonne donc les points de percee par angle polaire autour de leur
barycentre, ce qui suffit a reconstruire le contour sans chainage d'aretes.

Usage :
    python coupe.py fichier.geo x=1.5
    python coupe.py fichier.geo z=0.5 --save coupe.png
    python coupe.py fichier.geo y=2 --interior --eps 1e-9
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

import nymo_mesh as nm

# Tolerance geometrique (distance au plan en-dessous de laquelle un sommet est
# considere SUR le plan ; et aire de polygone en-dessous de laquelle la coupe
# est consideree degeneree -> ignoree).
EPS: float = 1e-9

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}


# ---------------------------------------------------------------------------
# Lecture du plan de coupe
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Geometrie de l'intersection maille / plan
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Couleurs (memes conventions que visualize.py)
# ---------------------------------------------------------------------------
def material_colors(materials: List[int]) -> Dict[int, tuple]:
    distinct = sorted(set(materials))
    cmap = plt.get_cmap("tab10")
    return {mat: cmap(i % 10) for i, mat in enumerate(distinct)}


# ---------------------------------------------------------------------------
# Trace de la coupe
# ---------------------------------------------------------------------------
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

    legend = [Line2D([0], [0], marker="s", linestyle="", markersize=12,
                     markerfacecolor=colors[m], markeredgecolor="k",
                     label=f"milieu {m}") for m in sorted(per_mat)]
    ax.legend(handles=legend, loc="upper left")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Image enregistree : {save}")
    else:
        plt.show()

    detail = ", ".join(f"milieu {m} : {per_mat[m]}" for m in sorted(per_mat))
    print(f"Coupe {AXIS_NAME[axis]}={value:g} : {len(polys)} mailles coupees ({detail})")
    return len(polys)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Coupe plane d'une geometrie NYMO, coloriee par milieu.")
    parser.add_argument("geo", help="fichier geometrie .geo")
    parser.add_argument("plan", help="plan de coupe axe : x=..., y=..., z=...")
    parser.add_argument("--eps", type=float, default=EPS,
                        help=f"tolerance geometrique (defaut {EPS:g})")
    parser.add_argument("--interior", action="store_true",
                        help="trace aussi le contour de chaque maille coupee")
    parser.add_argument("--save", default=None,
                        help="enregistrer en PNG au lieu d'afficher")
    args = parser.parse_args(argv[1:])

    axis, value = parse_plane(args.plan)
    mesh = nm.read_mesh(args.geo)
    plot_section(mesh, axis, value, eps=args.eps,
                 show_interior=args.interior, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
