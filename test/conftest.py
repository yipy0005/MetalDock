import math
import os
import sys

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def hexaammine_ruthenium():
    """[Ru(NH3)6]: Ru, 6 N on the axes at 2.14 A, 3 H on each N. 104 electrons when neutral."""
    atoms = [("Ru", 0.0, 0.0, 0.0)]
    for ax in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        n = tuple(2.14 * c for c in ax)
        atoms.append(("N", *n))
        # three hydrogens on a ring around the Ru-N axis, 1.02 A from N
        helper = (0, 0, 1) if ax[2] == 0 else (1, 0, 0)
        u = [ax[1] * helper[2] - ax[2] * helper[1], ax[2] * helper[0] - ax[0] * helper[2],
             ax[0] * helper[1] - ax[1] * helper[0]]
        norm = math.sqrt(sum(c * c for c in u))
        u = [c / norm for c in u]
        v = [ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2], ax[0] * u[1] - ax[1] * u[0]]
        for k in range(3):
            angle = 2 * math.pi * k / 3
            ring = [math.cos(angle) * u[i] + math.sin(angle) * v[i] for i in range(3)]
            atoms.append(("H", *[n[i] + 0.96 * ring[i] + 0.34 * ax[i] for i in range(3)]))
    return atoms


def write_xyz(path, atoms, comment="test complex"):
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\n{comment}\n")
        for sym, x, y, z in atoms:
            fh.write(f"{sym:<2} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    return str(path)


def pdb_line(serial, name, resname, chain, number, xyz, record="ATOM", element=None, altloc=" "):
    element = element or name[0]
    return (f"{record:<6}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}{number:>4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2}\n")


def write_pdb(path, hetero=False, models=False):
    """A tiny two-chain 'protein' with a few residues around the origin, optionally with a ligand and waters."""
    lines, serial = [], 1
    residues = [("A", 10, "HIS", (5.0, 0.0, 0.0)), ("A", 11, "CYS", (8.0, 1.0, 0.0)),
                ("A", 12, "ASP", (11.0, 0.0, 1.0)), ("B", 10, "GLU", (5.0, 12.0, 0.0))]
    if models:
        lines.append("MODEL        1\n")
    for chain, number, resname, base in residues:
        for name, off in (("N", (0, 0, 0)), ("CA", (1.4, 0, 0)), ("C", (2.2, 1.1, 0)), ("O", (2.0, 2.2, 0.1))):
            lines.append(pdb_line(serial, name, resname, chain, number,
                                  tuple(base[i] + off[i] for i in range(3))))
            serial += 1
    if models:
        lines.append("ENDMDL\nMODEL        2\n")
        lines.append(pdb_line(serial, "N", "HIS", "A", 10, (5.0, 0.0, 0.0)))
        lines.append("ENDMDL\n")
    if hetero:
        lines.append(pdb_line(serial, "ZN", " ZN", "A", 100, (6.0, 2.0, 2.0), record="HETATM", element="ZN"))
        serial += 1
        lines.append(pdb_line(serial, "O", "HOH", "A", 200, (9.0, 4.0, 2.0), record="HETATM", element="O"))
    lines.append("END\n")
    with open(path, "w") as fh:
        fh.writelines(lines)
    return str(path)


@pytest.fixture
def xyz_file(tmp_path):
    return write_xyz(tmp_path / "ru complex.xyz", hexaammine_ruthenium())


@pytest.fixture
def pdb_file(tmp_path):
    return write_pdb(tmp_path / "my protein.pdb")
