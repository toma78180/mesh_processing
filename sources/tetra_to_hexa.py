"""
tetra_to_hexa.py  (v3 - Polyèdres stricts)
=========================================

Fusion inverse d'une decomposition "N tetras par hexaedre" : reconstruit le
maillage a partir d'un maillage tetraedrique NYMO.

Dans cette version, la fusion des triangles en faces quadrangulaires est
soumise a une contrainte STRICTE de coplanarite. Si une face "hexaedrique"
est gauche (non planaire au-dela de la tolerance specifiee), elle reste
decomposee en 2 triangles parfaitement plans. Le maillage resultant peut
donc contenir des polyedres hybrides (ex: 5 quads, 2 triangles).

Usage :
    python tetra_to_hexa.py entree.msh3D sortie.msh3D [--ntet N] [--tol ANGLE]
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

import nymo_mesh as nm


# ===========================================================================
# Volumes
# ===========================================================================
def polyhedral_cell_volume(mesh: nm.Mesh, cell: int) -> float:
    G = nm.cell_barycenter(mesh, cell)
    vol = 0.0
    for f in mesh.cells[cell]:
        nd = mesh.faces[f]
        P = mesh.points[nd]
        C = P.mean(axis=0)
        m = len(nd)
        for j in range(m):
            a, b = P[j], P[(j + 1) % m]
            vol += abs(float(np.dot(a - G, np.cross(b - G, C - G)))) / 6.0
    return vol


def total_polyhedral_volume(mesh: nm.Mesh) -> float:
    return sum(polyhedral_cell_volume(mesh, c) for c in range(mesh.nb_cells))


def hexa_volume_diagonal_consistent(mesh: nm.Mesh, cell: int) -> float:
    G = nm.cell_barycenter(mesh, cell)
    vol = 0.0
    for f in mesh.cells[cell]:
        q = mesh.faces[f]
        if len(q) == 4:
            p1, a, p2, b = q
            Pa, Pb, P1, P2 = mesh.points[[a, b, p1, p2]]
            for X, Y, Z in ((Pa, Pb, P1), (Pa, Pb, P2)):
                vol += abs(float(np.dot(X - G, np.cross(Y - G, Z - G)))) / 6.0
        else:
            # Traitement exact pour les triangles conserves (ou autre polygone)
            P = mesh.points[q]
            C = P.mean(axis=0)
            m = len(q)
            for j in range(m):
                a, b = P[j], P[(j + 1) % m]
                vol += abs(float(np.dot(a - G, np.cross(b - G, C - G)))) / 6.0
    return vol


# ===========================================================================
# Outils geometrie / couplage
# ===========================================================================
def triangle_pair_flatness(points: np.ndarray, a: int, b: int,
                           p1: int, p2: int) -> float:
    Pa, Pb = points[a], points[b]
    u = Pb - Pa
    nu = float(np.linalg.norm(u))
    if nu < nm.TINY:
        return 0.0
    u = u / nu
    w1 = points[p1] - Pa
    w2 = points[p2] - Pa
    w1 = w1 - float(np.dot(w1, u)) * u
    w2 = w2 - float(np.dot(w2, u)) * u
    n1 = float(np.linalg.norm(w1))
    n2 = float(np.linalg.norm(w2))
    if n1 < nm.TINY or n2 < nm.TINY:
        return 0.0
    c = float(np.dot(w1, w2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def max_weight_perfect_matching(n: int, edges: List[Tuple[int, int, float]]
                                ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))

    best_score = float("-inf")
    best_match: Optional[List[Tuple[int, int]]] = None
    used = [False] * n
    cur: List[Tuple[int, int]] = []

    def rec(score: float) -> None:
        nonlocal best_score, best_match
        u = -1
        for k in range(n):
            if not used[k]:
                u = k
                break
        if u == -1:
            if score > best_score:
                best_score = score
                best_match = list(cur)
            return
        used[u] = True
        for v, w in adj[u]:
            if not used[v]:
                used[v] = True
                cur.append((u, v))
                rec(score + w)
                cur.pop()
                used[v] = False
        used[u] = False

    rec(0.0)
    return best_match, best_score


# ===========================================================================
# Reconstruction et filtrage de coplanarite
# ===========================================================================
def reconstruct_hexa_faces(mesh: nm.Mesh, cells: Sequence[int], tol: float
                           ) -> List[List[int]]:
    fcount: Dict[int, int] = {}
    for c in cells:
        for f in mesh.cells[c]:
            fcount[f] = fcount.get(f, 0) + 1
    outer = [f for f, k in fcount.items() if k == 1]
    tris = [mesh.faces[f] for f in outer]
    
    if len(tris) != 12:
        raise ValueError(f"{len(tris)} faces exterieures (12 attendues)")
    if any(len(t) != 3 for t in tris):
        raise ValueError("face exterieure non triangulaire")
    nodes = {v for t in tris for v in t}
    if len(nodes) != 8:
        raise ValueError(f"{len(nodes)} noeuds (8 attendus)")

    edge2tri: Dict[FrozenSet[int], List[int]] = {}
    for ti, t in enumerate(tris):
        for j in range(3):
            e = frozenset((t[j], t[(j + 1) % 3]))
            edge2tri.setdefault(e, []).append(ti)

    dual_edges: List[Tuple[int, int, float]] = []
    meta: Dict[Tuple[int, int], Tuple[FrozenSet[int], int, int, float]] = {}
    
    for e, ts in edge2tri.items():
        if len(ts) != 2:
            raise ValueError("arete de surface non partagee par 2 triangles")
        i, j = ts
        a, b = tuple(e)
        p1 = next(v for v in tris[i] if v not in e)
        p2 = next(v for v in tris[j] if v not in e)
        w = triangle_pair_flatness(mesh.points, a, b, p1, p2)
        dual_edges.append((i, j, w))
        meta[(i, j)] = (e, p1, p2, w)
        meta[(j, i)] = (e, p1, p2, w)

    match, _ = max_weight_perfect_matching(len(tris), dual_edges)
    if match is None or len(match) != 6:
        raise ValueError("pas de couplage parfait (groupe non hexaedrique)")

    faces_out: List[List[int]] = []
    diagonals: set = set()
    
    # Filtrage selon le seuil strict de tolerance (tol)
    for i, j in match:
        e, p1, p2, w = meta[(i, j)]
        a, b = sorted(e)
        diagonals.add(e)
        
        if w >= tol:
            faces_out.append([p1, a, p2, b])     # Quadrangle parfaitement plan
        else:
            faces_out.append([a, b, p1])         # Triangle 1
            faces_out.append([a, b, p2])         # Triangle 2

    # Verification topologique du squelette cubique
    skeleton = set(edge2tri.keys()) - diagonals
    deg: Dict[int, int] = {}
    for e in skeleton:
        for v in e:
            deg[v] = deg.get(v, 0) + 1
    if len(skeleton) != 12 or len(deg) != 8 or any(d != 3 for d in deg.values()):
        raise ValueError("squelette non cubique (couplage incoherent)")
        
    return faces_out


def find_groups(mesh: nm.Mesh, ntet: int = 5, tol: float = 179.99
                ) -> List[Tuple[List[int], List[List[int]]]]:
    if ntet < 4:
        raise ValueError("ntet doit valoir au moins 4.")
    if mesh.nb_cells % ntet != 0:
        raise ValueError(f"nb_mailles non multiple de ntet ({ntet}).")

    nodes_of = [frozenset(nm.cell_node_ids(mesh, c)) for c in range(mesh.nb_cells)]
    node2cells: Dict[int, set] = {}
    for c in range(mesh.nb_cells):
        for v in nodes_of[c]:
            node2cells.setdefault(v, set()).add(c)
            
    tri2cells: Dict[FrozenSet[int], set] = {}
    for c in range(mesh.nb_cells):
        for f in mesh.cells[c]:
            tri = frozenset(mesh.faces[f])
            if len(tri) == 3:
                tri2cells.setdefault(tri, set()).add(c)

    assigned = [False] * mesh.nb_cells
    groups: List[Tuple[List[int], List[List[int]]]] = []

    for seed in range(mesh.nb_cells):
        if assigned[seed]:
            continue
            
        seen = {nodes_of[seed]}
        frontier = [nodes_of[seed]]
        node_sets_8: List[FrozenSet[int]] = []
        
        while frontier:
            un = frontier.pop()
            if len(un) == 8:
                node_sets_8.append(un)
                continue
            candidates: set = set()
            for tri in combinations(sorted(un), 3):
                t = frozenset(tri)
                if t in tri2cells:
                    candidates |= tri2cells[t]
            for c in candidates:
                if assigned[c]:
                    continue
                nu = un | nodes_of[c]
                if len(nu) <= 8 and nu not in seen:
                    seen.add(nu)
                    frontier.append(nu)
                    
        found: Optional[Tuple[set, List[List[int]]]] = None
        for H in node_sets_8:
            cells = {c for v in H for c in node2cells[v]
                     if not assigned[c] and nodes_of[c] <= H}
            if len(cells) == ntet and seed in cells:
                try:
                    group_faces = reconstruct_hexa_faces(mesh, cells, tol)
                except ValueError:
                    continue
                found = (cells, group_faces)
                break
                
        if found is None:
            raise ValueError(f"Maille {seed} orpheline. Probleme de topologie.")
            
        cells, group_faces = found
        for c in cells:
            assigned[c] = True
        groups.append((sorted(cells), group_faces))

    print(f"{len(groups)} blocs volumiques reconstitues.")
    return groups


# ===========================================================================
# Assemblage du maillage
# ===========================================================================
def merge_to_hexa(mesh: nm.Mesh, ntet: int = 5, tol: float = 179.99,
                  name: Optional[str] = None) -> nm.Mesh:
    groups = find_groups(mesh, ntet=ntet, tol=tol)

    face_index: Dict[FrozenSet[int], int] = {}
    new_faces: List[List[int]] = []
    new_cells: List[List[int]] = []
    new_materials: List[int] = []

    for cells, group_faces in groups:
        mats = {mesh.materials[c] for c in cells}
        if len(mats) != 1:
            raise ValueError(f"Milieux heterogenes dans le groupe {sorted(cells)}")
        new_materials.append(mats.pop())

        cell_faces: List[int] = []
        for fnodes in group_faces:
            key = frozenset(fnodes)
            idx = face_index.get(key)
            if idx is None:
                idx = len(new_faces)
                face_index[key] = idx
                new_faces.append(fnodes)
            cell_faces.append(idx)
        new_cells.append(cell_faces)

    used = sorted({v for fc in new_faces for v in fc})
    remap = {old: new for new, old in enumerate(used)}
    points = mesh.points[used]
    faces = [[remap[v] for v in fc] for fc in new_faces]

    out_name = name or (mesh.name.replace("tetra", "hexa")
                        if "tetra" in mesh.name else mesh.name + "_hexa")
    return nm.Mesh(points=points, faces=faces, cells=new_cells,
                   materials=new_materials, name=out_name)


# ===========================================================================
# Controles de qualite
# ===========================================================================
def quad_planarity(mesh: nm.Mesh, face: int) -> float:
    P = mesh.points[mesh.faces[face]]
    n = np.cross(P[1] - P[0], P[2] - P[0])
    norm = float(np.linalg.norm(n))
    if norm < nm.TINY:
        return float("inf")
    d = abs(float(np.dot(P[3] - P[0], n))) / norm
    scale = max(float(np.linalg.norm(P[2] - P[0])),
                float(np.linalg.norm(P[3] - P[1])))
    return d / scale


def validate(tet: nm.Mesh, hexa: nm.Mesh) -> None:
    vol_t = total_polyhedral_volume(tet)
    vol_h = sum(hexa_volume_diagonal_consistent(hexa, c) for c in range(hexa.nb_cells))
    rel = abs(vol_t - vol_h) / vol_t
    print(f"Volume tetra            : {vol_t:.10g}")
    print(f"Volume fusion (strict)  : {vol_h:.10g}   (ecart relatif {rel:.3e})")

    nb_quads = sum(1 for fc in hexa.faces if len(fc) == 4)
    nb_tris = sum(1 for fc in hexa.faces if len(fc) == 3)
    print(f"Topologie de surface    : {nb_quads} quadrangles, {nb_tris} triangles.")

    planar = [quad_planarity(hexa, f) for f in range(hexa.nb_faces) if len(hexa.faces[f]) == 4]
    if planar:
        print(f"Planeite des quads (d/L): max {max(planar):.3e}, moyenne {np.mean(planar):.3e}")

    if rel > 1e-9:
        print("ATTENTION : volume non conserve au-dela de la tolerance.")
    else:
        print("Volume parfaitement conserve.")


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fusion tetra -> hexa stricte (génère des polyèdres si non-plans).")
    parser.add_argument("entree", help="maillage tetraedrique .msh3D")
    parser.add_argument("sortie", help="maillage hexaedrique/polyedrique produit")
    parser.add_argument("--ntet", type=int, default=5,
                        help="nombre de tetras par bloc (defaut 5).")
    parser.add_argument("--tol", type=float, default=179.99,
                        help="Angle diedre MINIMUM (en degres) pour fusionner deux "
                             "triangles en quadrangle (defaut: 179.99).")
    parser.add_argument("--name", default=None, help="nom de la geometrie")
    args = parser.parse_args(argv[1:])

    tet = nm.read_mesh(args.entree)
    print(f"Lu : {tet.nb_points} noeuds, {tet.nb_faces} faces, {tet.nb_cells} mailles.")

    hexa = merge_to_hexa(tet, ntet=args.ntet, tol=args.tol, name=args.name)
    validate(tet, hexa)

    nm.write_mesh(hexa, args.sortie)
    print(f"Ecrit : {args.sortie}  ({hexa.nb_points} noeuds, "
          f"{hexa.nb_faces} faces, {hexa.nb_cells} mailles).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
