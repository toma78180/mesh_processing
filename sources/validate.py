"""
validate.py
===========

Verification de la coherence geometrique d'une geometrie NYMO.
Chaque critere est explicite et renvoie OK / ECHEC avec un detail.

Criteres verifies :
  G1   index de noeuds des faces dans [0, nb_points[
  G2   index de faces des mailles dans [0, nb_faces[
  G3   exactement un milieu par maille
  G4   chaque face appartient a 1 (bord) ou 2 (interieure) mailles
  G5   chaque maille est un polyedre FERME (toute arete vue 2x)
  G6   aucune face degeneree (3 noeuds distincts, aire > 0)
  G7   aucun point orphelin (tout point est utilise par >= 1 face)
  G8   faces polygonales coplanaires (tous les noeuds dans le meme plan)
  G9   ordre des noeuds coherent (normale sortante de la face vers l'exterieur)
  G10  pas de noeud duplique au sein d'une face
  G11  pas de face dupliquee au sein d'une maille

Usage :
    python validate.py geometrie.geo
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import List

import numpy as np

import nymo_mesh as nm

TOL = 1e-9


# ---------------------------------------------------------------------------
# Petit conteneur de resultats
# ---------------------------------------------------------------------------
class Report:
    def __init__(self, titre: str):
        self.titre = titre
        self.criteres: List[tuple] = []

    def add(self, nom: str, ok: bool, detail: str = "") -> None:
        self.criteres.append((nom, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.criteres)

    def show(self) -> None:
        print(f"\n=== {self.titre} ===")
        for nom, ok, detail in self.criteres:
            tag = "OK   " if ok else "ECHEC"
            line = f"  [{tag}] {nom}"
            if detail:
                line += f"  ({detail})"
            print(line)
        print(f"  --> {'TOUT VALIDE' if self.ok else 'AU MOINS UN ECHEC'}")


# ---------------------------------------------------------------------------
# CRITERE 1 : coherence d'une geometrie
# ---------------------------------------------------------------------------
def check_geometry(mesh: nm.Mesh) -> Report:
    """Verifie qu'une geometrie est intrinsequement valide.

    Criteres :
      G1   index de noeuds des faces dans [0, nb_points[
      G2   index de faces des mailles dans [0, nb_faces[
      G3   exactement un milieu par maille
      G4   chaque face appartient a 1 (bord) ou 2 (interieure) mailles
      G5   chaque maille est un polyedre FERME (toute arete vue 2x)
      G6   aucune face degeneree (3 noeuds distincts, aire > 0)
      G7   aucun point orphelin (tout point est utilise par >= 1 face)
      G8   faces polygonales coplanaires (tous les noeuds dans le meme plan)
      G9   ordre des noeuds coherent (normale sortante de la face vers l'exterieur)
      G10  pas de noeud duplique au sein d'une face
      G11  pas de face dupliquee au sein d'une maille
    """
    rep = Report("Coherence de la geometrie")
    f2c = nm.build_face_to_cells(mesh)

    # G1
    bad_nodes = [f for f in range(mesh.nb_faces)
                 if any(n < 0 or n >= mesh.nb_points for n in mesh.faces[f])]
    rep.add("G1 index noeuds valides", not bad_nodes,
            "" if not bad_nodes else f"faces {[f+1 for f in bad_nodes[:5]]}")

    # G2
    bad_faces = [c for c in range(mesh.nb_cells)
                 if any(f < 0 or f >= mesh.nb_faces for f in mesh.cells[c])]
    rep.add("G2 index faces valides", not bad_faces,
            "" if not bad_faces else f"mailles {[c+1 for c in bad_faces[:5]]}")

    # G3
    g3 = len(mesh.materials) == mesh.nb_cells
    rep.add("G3 un milieu par maille", g3,
            f"{len(mesh.materials)} milieux / {mesh.nb_cells} mailles")

    # G4
    owners = Counter(len(v) for v in f2c.values())
    g4_bad = [f for f, v in f2c.items() if len(v) not in (1, 2)]
    rep.add("G4 voisinage des faces (1 ou 2)", not g4_bad,
            f"bord={owners.get(1,0)} int={owners.get(2,0)}"
            + ("" if not g4_bad else f" PROBLEME faces {[f+1 for f in g4_bad[:5]]}"))

    # G5
    not_closed = []
    for c in range(mesh.nb_cells):
        edges: Counter = Counter()
        for f in mesh.cells[c]:
            nodes = mesh.faces[f]
            for j in range(len(nodes)):
                edges[frozenset((nodes[j], nodes[(j + 1) % len(nodes)]))] += 1
        if any(n != 2 for n in edges.values()):
            not_closed.append(c)
    rep.add("G5 mailles fermees (manifold)", not not_closed,
            "" if not not_closed else f"mailles {[c+1 for c in not_closed[:5]]}")

    # G6
    degen = []
    for f in range(mesh.nb_faces):
        n = mesh.faces[f]
        if len(set(n)) < 3:
            degen.append(f); continue
        a, b, c = mesh.points[n[0]], mesh.points[n[1]], mesh.points[n[2]]
        if np.linalg.norm(np.cross(b - a, c - a)) < TOL:
            degen.append(f)
    rep.add("G6 faces non degenerees", not degen,
            "" if not degen else f"faces {[f+1 for f in degen[:5]]}")

    # G7
    used = {n for f in mesh.faces for n in f}
    orphans = [p for p in range(mesh.nb_points) if p not in used]
    rep.add("G7 aucun point orphelin", not orphans,
            "" if not orphans else f"points {[p+1 for p in orphans[:5]]}")

    # G8 : coplanarité des faces polygonales (> 3 noeuds)
    # Pour chaque noeud supplementaire, on verifie que sa distance au plan
    # defini par les 3 premiers noeuds est inferieure a TOL_COPLAN.
    # TOL_COPLAN est adaptatif : proportionnel a la taille caracteristique
    # de la face (longueur de la diagonale bounding-box) pour eviter les
    # faux positifs sur les grands maillages.
    non_coplan = []
    for f in range(mesh.nb_faces):
        n = mesh.faces[f]
        if len(n) <= 3:
            continue
        p0, p1, p2 = mesh.points[n[0]], mesh.points[n[1]], mesh.points[n[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < TOL:
            continue   # face degeneree, deja signalee en G6
        normal_unit = normal / norm_len
        pts = mesh.points[n]
        scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) or 1.0
        tol_coplan = max(TOL, 1e-6 * scale)
        if any(abs(float(np.dot(mesh.points[nk] - p0, normal_unit))) > tol_coplan
               for nk in n[3:]):
            non_coplan.append(f)
    rep.add("G8 faces coplanaires", not non_coplan,
            "" if not non_coplan
            else f"{len(non_coplan)} face(s) non coplanaires : {[f+1 for f in non_coplan[:5]]}")

    # G9 : ordre des noeuds coherent (normale sortante vers l'exterieur)
    # Pour chaque face, on calcule la normale via le produit vectoriel des
    # 3 premiers noeuds, puis on verifie que le barycentre de la maille
    # voisine (pour une face interieure) ou le barycentre global (pour une
    # face de bord) est bien du cote oppose a la normale.
    # Convention NYMO : la normale va de la region gauche (rG) vers la
    # region droite (rD).  Pour une face de bord (rD = EXTERIOR), le
    # barycentre de l'unique maille voisine doit etre du cote OPPOSE a la
    # normale (i.e. dot(normale, G_maille - P0) < 0).
    barycenters = [nm.cell_barycenter(mesh, c) for c in range(mesh.nb_cells)]
    bad_orient = []
    for f in range(mesh.nb_faces):
        n = mesh.faces[f]
        normal = nm.face_normal(mesh, f)
        if np.linalg.norm(normal) < TOL:
            continue   # face degeneree, ignoree ici
        p0 = mesh.points[n[0]]
        cells = f2c[f]
        if len(cells) == 1:
            # Face de bord : le barycentre de la maille doit etre oppose a la normale
            dot = float(np.dot(normal, barycenters[cells[0]] - p0))
            if dot >= 0.0:
                bad_orient.append(f)
        else:
            # Face interieure : les deux barycentres doivent etre de cotes opposes
            d0 = float(np.dot(normal, barycenters[cells[0]] - p0))
            d1 = float(np.dot(normal, barycenters[cells[1]] - p0))
            if d0 * d1 >= 0.0:
                bad_orient.append(f)
    rep.add("G9 ordre des noeuds coherent (normales sortantes)", not bad_orient,
            "" if not bad_orient
            else f"{len(bad_orient)} face(s) mal orientee(s) : {[f+1 for f in bad_orient[:5]]}")

    # G10 : pas de noeud duplique au sein d'une face
    dup_nodes = [f for f in range(mesh.nb_faces)
                 if len(mesh.faces[f]) != len(set(mesh.faces[f]))]
    rep.add("G10 pas de noeud duplique dans une face", not dup_nodes,
            "" if not dup_nodes else f"faces {[f+1 for f in dup_nodes[:5]]}")

    # G11 : pas de face dupliquee au sein d'une maille
    dup_faces = [c for c in range(mesh.nb_cells)
                 if len(mesh.cells[c]) != len(set(mesh.cells[c]))]
    rep.add("G11 pas de face dupliquee dans une maille", not dup_faces,
            "" if not dup_faces else f"mailles {[c+1 for c in dup_faces[:5]]}")

    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    if len(argv) == 2:
        mesh = nm.read_mesh(argv[1])
        rep = check_geometry(mesh)
        rep.show()
        return 0 if rep.ok else 2

    print("Usage :\n"
          "  python validate.py geometrie.geo")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
