"""Regression tests for the parsing of AutoDock output in src/metal_dock/docking.py.

Each of these crashed or silently misread the file before: the files AutoDock writes are fixed-column formats, so
values can touch their neighbours, and energies carry an explicit '+' when positive.
"""
from types import SimpleNamespace

import pytest

from conftest import pdb_line
from src.metal_dock.docking import Docking, pdbqt_xyz


def test_pdbqt_xyz_normal_line():
    line = pdb_line(1, "C", "LIG", "A", 1, (-27.93, -16.774, 0.413), record="ATOM")
    assert pdbqt_xyz(line) == (-27.93, -16.774, 0.413)


def test_pdbqt_xyz_energy_term_touching_z():
    # AutoDock wrote '+18.99' straight after the z coordinate: splitting on whitespace gives '0.413+18.99'
    line = "ATOM      2 C     LIG A   1    -27.930 -16.774   0.413+18.99 +0.00    -0.014 C \n"
    assert pdbqt_xyz(line) == (-27.93, -16.774, 0.413)


def test_pdbqt_xyz_falls_back_to_splitting():
    line = "ATOM 2 C LIG A 1 -27.930 -16.774 0.413 1.00 0.00 -0.014 C\n"
    assert pdbqt_xyz(line) == (-27.93, -16.774, 0.413)


def test_pdbqt_xyz_coordinates_that_run_together():
    line = pdb_line(1, "CA", "ALA", "A", 5, (-100.123, -200.456, 300.789))
    assert pdbqt_xyz(line) == (-100.123, -200.456, 300.789)


def fake_docking(num_poses=3, heavy=10):
    return SimpleNamespace(par=SimpleNamespace(num_poses=num_poses, n_heavy_atoms=heavy))


def test_extract_dlg_reads_positive_and_negative_energies(tmp_path):
    dlg = tmp_path / "x.dlg"
    dlg.write_text(
        "USER    Estimated Free Energy of Binding    = +221.74 kcal/mol  [=(1)+(2)+(3)-(4)]\n"
        "USER    Estimated Free Energy of Binding    =  -9.44 kcal/mol  [=(1)+(2)+(3)-(4)]\n"
        "USER    Estimated Free Energy of Binding    = +0.50 kcal/mol  [=(1)+(2)+(3)-(4)]\n"
        "USER    Estimated Free Energy of Binding    = -1.00 kcal/mol  [=(1)+(2)+(3)-(4)]\n")
    energies, efficiencies = Docking._extract_dlg(fake_docking(num_poses=3, heavy=10), str(dlg))
    assert energies == [221.74, -9.44, 0.5]
    assert efficiencies == pytest.approx([22.174, -0.944, 0.05])


def test_interacting_residues_with_large_residue_numbers_and_big_coordinates(tmp_path):
    pose = tmp_path / "pose.xyz"
    pose.write_text("1\n\nRu 10.000 10.000 10.000\n")
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        pdb_line(1, "CA", "HIS", "A", 1307, (10.0, 10.0, 12.0)) +        # 2 A away, residue number 1307
        pdb_line(2, "CA", "ASP", "B", 12, (-100.5, -200.5, 300.5)) +     # far away, coordinates run together
        pdb_line(3, "CA", "GLU", "A", 20, (10.0, 10.0, 30.0)))
    found = Docking._extract_interacting_residues(fake_docking(), str(pose), str(protein))
    assert [tuple(r) for r in found] == [("HIS", "1307")]
