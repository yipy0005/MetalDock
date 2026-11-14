"""Tests for the guided interface (metaldock_cli.py).

Run with:  pixi run -e dev test
None of these need ORCA, SLURM or a long calculation. SLURM and the QM program are mocked or not reached.
"""
import builtins
import configparser
import os
import shutil
import subprocess

import pytest

import metaldock_cli as m
from conftest import hexaammine_ruthenium, write_pdb, write_xyz


@pytest.fixture(autouse=True)
def clear_warnings(monkeypatch):
    # pytest's temporary folders are under /tmp, which the wizard refuses for cluster jobs (see test_node_local_folder)
    monkeypatch.setattr(m, "NODE_LOCAL_PREFIXES", ())
    m.WARNINGS.clear()
    yield
    m.WARNINGS.clear()


# ---- the metal complex ---------------------------------------------------------------------------------------

def test_read_xyz(xyz_file):
    info = m.read_xyz(xyz_file)
    assert info.n_atoms == 25
    assert info.counts == {"Ru": 1, "N": 6, "H": 18}
    assert info.metals == ["Ru"]
    assert info.electrons == 44 + 6 * 7 + 18
    assert info.formula == "H18N6Ru"
    assert info.min_distance > 0.9
    assert info.has_hydrogen


def test_read_xyz_atom_count_mismatch(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("5\ncomment\nC 0 0 0\nH 0 0 1\n")
    with pytest.raises(ValueError, match="says 5 atoms but the file has 2"):
        m.read_xyz(str(path))


def test_read_xyz_not_an_xyz(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("hello\nworld\nagain\n")
    with pytest.raises(ValueError, match="number of atoms"):
        m.read_xyz(str(path))


def test_read_xyz_bad_element_and_coordinate(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("2\nc\nXx 0 0 0\nH 0 0 1\n")
    with pytest.raises(ValueError, match="not a chemical element"):
        m.read_xyz(str(path))
    path.write_text("2\nc\nC 0 0\nH 0 0 1\n")
    with pytest.raises(ValueError, match="element x y z"):
        m.read_xyz(str(path))


def test_read_xyz_accepts_numbered_symbols(tmp_path):
    path = tmp_path / "n.xyz"
    path.write_text("2\nc\nRU1 0 0 0\nH2 0 0 1.5\n")
    info = m.read_xyz(str(path))
    assert info.counts == {"Ru": 1, "H": 1}


def test_complex_without_supported_metal(tmp_path):
    path = write_xyz(tmp_path / "organic.xyz", [("C", 0, 0, 0), ("O", 0, 0, 1.2)])
    info = m.read_xyz(path)
    assert info.metals == []
    m.report_complex(info)  # must not raise


def test_complex_warnings(tmp_path, capsys):
    path = write_xyz(tmp_path / "c.xyz", [("Ru", 0, 0, 0), ("N", 0.2, 0, 0)])
    m.report_complex(m.read_xyz(path))
    text = " ".join(m.WARNINGS)
    assert "apart" in text and "no hydrogen" in text


@pytest.mark.parametrize("charge,spin,ok", [(0, 0, True), (2, 0, True), (1, 0, False), (1, 1, True),
                                            (0, 1, False), (0, 2, True), (0, 105, False), (200, 0, False)])
def test_spin_problem(xyz_file, charge, spin, ok):
    # 104 electrons when neutral
    info = m.read_xyz(xyz_file)
    assert (m.spin_problem(info, charge, spin) is None) is ok


def test_spin_problem_message_is_helpful(xyz_file):
    info = m.read_xyz(xyz_file)
    text = m.spin_problem(info, 1, 0)
    assert "103 electrons" in text and "odd" in text


# ---- the protein ---------------------------------------------------------------------------------------------

def test_read_pdb(tmp_path):
    info = m.read_pdb(write_pdb(tmp_path / "p.pdb", hetero=True))
    assert info.chains == ["A", "B"]
    assert info.residue_count == 5   # 4 residues + the zinc
    assert info.hetero == {"ZN": 1}
    assert info.waters == 1
    assert not info.hydrogens
    assert info.n_atoms == 16 + 2


def test_read_pdb_models(tmp_path):
    info = m.read_pdb(write_pdb(tmp_path / "p.pdb", models=True))
    assert info.models == 2


def test_read_pdb_rejects_non_pdb(tmp_path):
    path = tmp_path / "x.pdb"
    path.write_text("data_1abc\nloop_\n")
    with pytest.raises(ValueError, match="no ATOM records"):
        m.read_pdb(str(path))


def test_read_pdb_bad_coordinates(tmp_path):
    path = tmp_path / "x.pdb"
    path.write_text("ATOM      1  N   HIS A  10        x.xxx   0.000   0.000  1.00  0.00           N\n")
    with pytest.raises(ValueError, match="coordinates"):
        m.read_pdb(str(path))


def test_find_residues(pdb_file):
    info = m.read_pdb(pdb_file)
    assert [k for k, _ in m.find_residues(info, "A10")] == [("A", 10)]
    assert [k for k, _ in m.find_residues(info, "HIS10")] == [("A", 10)]
    assert [k for k, _ in m.find_residues(info, "A:HIS10")] == [("A", 10)]
    assert [k for k, _ in m.find_residues(info, "A:10, B10")] == [("A", 10), ("B", 10)]
    assert len(m.find_residues(info, "10")) == 2     # no chain given: both chains have a residue 10


@pytest.mark.parametrize("text,expected", [
    ("57", (None, None, 57)), ("A57", ("A", None, 57)), ("A:57", ("A", None, 57)), ("HIS57", (None, "HIS", 57)),
    ("A:HIS57", ("A", "HIS", 57)), ("AHIS57", ("A", "HIS", 57)), ("ALA57", (None, "ALA", 57)),
    ("1:57", ("1", None, 57)), ("A-5", ("A", None, -5)), ("histidine", None), ("A:", None), ("", None),
])
def test_parse_residue_token(text, expected):
    assert m.parse_residue_token(text) == expected


def test_find_residues_errors(pdb_file):
    info = m.read_pdb(pdb_file)
    with pytest.raises(ValueError, match="Not found in the protein: A99. Chains in the file: A, B"):
        m.find_residues(info, "A99")
    with pytest.raises(ValueError, match="not a residue"):
        m.find_residues(info, "histidine")
    with pytest.raises(ValueError, match="at least one"):
        m.find_residues(info, " , ")
    with pytest.raises(ValueError, match="Not found"):
        m.find_residues(info, "CYS10")  # residue 10 of chain A is a histidine


def test_residue_centre_and_extent(pdb_file):
    info = m.read_pdb(pdb_file)
    centre = m.residue_centre(m.find_residues(info, "A10"))
    assert centre == pytest.approx((6.4, 0.825, 0.025), abs=0.01)
    lo, hi = m.protein_extent(info)
    assert lo[0] == 5.0 and hi[1] > 12
    assert m.inside_protein_box(info, (6, 1, 0))
    assert not m.inside_protein_box(info, (100, 0, 0))


# ---- the input file ------------------------------------------------------------------------------------------

def make_settings(pdb_file, xyz_file, **kw):
    s = m.Settings(pdb_src=pdb_file, xyz_src=xyz_file, metal="Ru", ncpu=4, memory=2000,
                   jobname="job", jobdir="/tmp/job")
    s.pdb = m.read_pdb(pdb_file)
    s.cplx = m.read_xyz(xyz_file)
    for key, value in kw.items():
        setattr(s, key, value)
    return s


def test_staged_names_are_safe():
    assert m.staged_name("/a/b/my protein (1).pdb", ".pdb") == "my_protein__1_.pdb"
    assert m.staged_name("/a/b/x.PDB", ".pdb") == "x.pdb"


def test_ini_orca(pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file, centre=(1.5, 2.0, -3.25), box=[20.0, 20.0, 20.0],
                      orca_simple="PBE def2-SVP", geom_opt=False, charge=2)
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(m.build_ini(s))
    assert config["DEFAULT"]["metal_symbol"] == "Ru"
    assert config["DEFAULT"]["ncpu"] == "4"
    assert config["PROTEIN"]["pdb_file"] == "my_protein.pdb"
    assert config["QM"]["engine"] == "ORCA"
    assert config["QM"]["orcasimpleinput"] == "PBE def2-SVP"
    assert config["QM"]["orcablocks"] == "%maxcore 2000"
    assert config["METAL_COMPLEX"]["xyz_file"] == "ru_complex.xyz"
    assert config["METAL_COMPLEX"]["geom_opt"] == "False"
    assert config["METAL_COMPLEX"]["charge"] == "2"
    assert config["DOCKING"]["dock_x"] == "1.5" and config["DOCKING"]["dock_z"] == "-3.25"
    assert config["DOCKING"]["box_size"] == "20"


def test_ini_keeps_user_maxcore_and_blocks(pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file, orca_blocks='%basis newECP Ru "def2-SD" end end')
    assert m.effective_orca_blocks(s).startswith("%maxcore 2000 %basis newECP")
    s = make_settings(pdb_file, xyz_file, orca_blocks="%maxcore 4000")
    assert m.effective_orca_blocks(s) == "%maxcore 4000"


def test_ini_without_site_uses_metal_position(pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file, centre=None)
    assert "dock_x" not in m.build_ini(s)


def test_ini_box_and_metal_energies(pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file, box=[20.0, 30.0, 25.0], ini_parameters=True,
                      energies=[1.0, 2.0, 3.0, 4.5])
    text = m.build_ini(s)
    assert "box_size = 20,30,25" in text
    assert "ini_parameters = True" in text and "e_NA = 1" in text and "e_HD = 4.5" in text
    assert "ini_parameters" not in m.build_ini(make_settings(pdb_file, xyz_file))


def test_only_orca_is_offered(pdb_file, xyz_file):
    """ADF and Gaussian are not available on the cluster, so the wizard writes only ORCA input."""
    text = m.build_ini(make_settings(pdb_file, xyz_file))
    assert "engine = ORCA" in text and "orcasimpleinput" in text
    for word in ("ADF", "Gaussian", "basis_set", "functional"):
        assert word not in text
    assert not hasattr(m, "engine_status") and not hasattr(m, "QM_ENVIRONMENT")


def test_ini_is_accepted_by_metaldock_itself(tmp_path, pdb_file, xyz_file, monkeypatch):
    """The real MetalDock parser must read what the wizard writes, including the unusual box and ORCA blocks."""
    from src.metal_dock.parser_metal_dock import DockParser
    s = make_settings(pdb_file, xyz_file, centre=(1.5, 2.0, -3.25), box=[20.0, 30.0, 25.0], num_poses=4,
                      num_evals=12345, orca_blocks='%basis newECP Ru "def2-SD" end end', charge=2, rmsd=True)
    job = tmp_path / "job"
    s.jobdir = str(job)
    m.write_job_folder(s)
    monkeypatch.chdir(job)
    par = DockParser("input.ini")
    assert par.metal_symbol == "Ru"
    assert (par.dock_x, par.dock_y, par.dock_z) == (1.5, 2.0, -3.25)
    assert par.box_size == [20.0, 30.0, 25.0]
    assert par.num_poses == 4 and par.ga_dock_num_evals == 12345
    assert par.engine == "ORCA" and par.charge == 2 and par.rmsd is True
    assert par.orcablocks.startswith("%maxcore 2000 %basis newECP Ru")
    assert par.name_protein == "my_protein" and par.name_ligand == "ru_complex"
    assert par.pH == 7.4 and par.clean_pdb is True and par.vacant_site is True


def test_job_folder_copies_inputs_and_keeps_originals(tmp_path, pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file)
    s.jobdir = str(tmp_path / "run")
    m.write_job_folder(s)
    assert sorted(os.listdir(s.jobdir)) == ["input.ini", "my_protein.pdb", "ru_complex.xyz"]
    assert os.path.exists(pdb_file) and os.path.exists(xyz_file)


def test_job_folder_in_the_data_folder_does_not_copy_onto_itself(tmp_path):
    pdb = write_pdb(tmp_path / "prot.pdb")
    xyz = write_xyz(tmp_path / "cplx.xyz", hexaammine_ruthenium())
    s = make_settings(pdb, xyz)
    s.jobdir = str(tmp_path)
    m.write_job_folder(s)  # would raise SameFileError if it tried to copy a file onto itself
    assert m.folder_only_holds(str(tmp_path), [pdb, xyz, os.path.join(tmp_path, "input.ini")])


# ---- the job script ------------------------------------------------------------------------------------------

def test_script_orca(pdb_file, xyz_file, monkeypatch):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/opt/pixi/bin/pixi" if name == "pixi" else None)
    s = make_settings(pdb_file, xyz_file, time_limit="2:00:00", jobdir="/data/my job", jobname="job1")
    text = m.build_script(s)
    assert "#SBATCH --partition=ncpu" in text
    assert "#SBATCH --ntasks=4" in text and "#SBATCH --nodes=1" in text   # mpirun counts tasks as slots
    assert "cpus-per-task" not in text
    assert "#SBATCH --mem-per-cpu=2600M" in text
    assert "#SBATCH --time=2:00:00" in text
    assert text.index("module load ORCA/6.1.0") < text.index("module load OpenMPI/4.1.6-GCC-13.2.0")
    assert "cd '/data/my job'" in text
    assert 'export PATH="/opt/pixi/bin:$PATH"' in text
    last = text.strip().splitlines()[-1]
    assert last.startswith("pixi run --manifest-path ") and last.endswith("-i input.ini -m dock")
    assert os.path.join(m.REPO_DIR, "metaldock") in last


def test_script_without_pixi_uses_this_python(pdb_file, xyz_file, monkeypatch):
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    text = m.build_script(make_settings(pdb_file, xyz_file))
    assert m.sys.executable in text.strip().splitlines()[-1]


def test_script_does_not_export_other_programs_settings(pdb_file, xyz_file, monkeypatch):
    monkeypatch.setenv("AMSHOME", "/opt/ams")
    monkeypatch.setenv("g16root", "/opt/g16")
    text = m.build_script(make_settings(pdb_file, xyz_file))
    assert "AMSHOME" not in text and "g16root" not in text


def test_time_estimates(pdb_file, xyz_file):
    quick = make_settings(pdb_file, xyz_file, geom_opt=False, qm_level="quick", effort="quick")
    full = make_settings(pdb_file, xyz_file, geom_opt=True, qm_level="standard", effort="standard")
    assert m.estimate_minutes(quick) < m.estimate_minutes(full) <= 7 * 24 * 60
    assert m.fmt_minutes(90) == "1:30:00"
    assert m.fmt_minutes(26 * 60) == "1-02:00:00"
    assert m.time_validator("2:00:00") and m.time_validator("1-00:00:00")
    with pytest.raises(ValueError):
        m.time_validator("two hours")


# ---- folders and earlier runs --------------------------------------------------------------------------------

def test_validate_folder(tmp_path):
    assert m.validate_folder(str(tmp_path / "new"), []) is None
    assert "file" in m.validate_folder(__file__, [])
    assert "itself" in m.validate_folder(str(tmp_path), [str(tmp_path)])
    assert "outside the MetalDock installation" in m.validate_folder(os.path.join(m.REPO_DIR, "x"), [])


def test_node_local_folder_is_refused_for_cluster_jobs(monkeypatch):
    monkeypatch.setattr(m, "NODE_LOCAL_PREFIXES", ("/tmp", "/var/tmp", "/dev/shm"))
    for path in ("/tmp/job", "/tmp", "/var/tmp/x/y", "/dev/shm/run"):
        assert "shared storage" in m.validate_folder(path, [], for_cluster=True)
        assert "shared storage" not in (m.validate_folder(path, []) or "")   # fine when running here
    # only /tmp itself and what is below it: a folder that merely starts with the same letters is not node-local
    assert "shared storage" not in (m.validate_folder("/tmpfoo/job", [], for_cluster=True) or "")


def test_previous_run_detection(tmp_path, pdb_file, xyz_file):
    s = make_settings(pdb_file, xyz_file)
    s.jobdir = str(tmp_path / "run")
    assert m.previous_run(s, s.jobdir) is None
    m.write_job_folder(s)
    assert m.previous_run(s, s.jobdir) == ([], False)
    os.makedirs(os.path.join(s.jobdir, "output", "results"))
    open(os.path.join(s.jobdir, "output", "results", "docking_results.dlg"), "w").close()
    assert m.previous_run(s, s.jobdir) == ([], True)
    s.num_poses = 99
    assert m.previous_run(s, s.jobdir)[0] == ["different settings"]
    s.num_poses = 10
    with open(os.path.join(s.jobdir, "my_protein.pdb"), "a") as fh:
        fh.write("REMARK changed\n")
    assert "a different my protein.pdb" in m.previous_run(s, s.jobdir)[0]


# ---- results -------------------------------------------------------------------------------------------------

LOG = """2026-01-01 10:00:00,000 - MetalDock - INFO - STARTING METALDOCK
2026-01-01 10:10:00,000 - MetalDock - INFO - DOCKING RESULTS:
2026-01-01 10:10:00,000 - MetalDock - INFO - Pose 1:
2026-01-01 10:10:00,000 - MetalDock - INFO - -------------
2026-01-01 10:10:00,001 - MetalDock - INFO - Binding Energy: -7.3412 kcal/mol
2026-01-01 10:10:00,001 - MetalDock - INFO - Ligand Efficiency: -0.2280 kcal/mol
2026-01-01 10:10:00,001 - MetalDock - INFO - Interacting Residues:
2026-01-01 10:10:00,001 - MetalDock - INFO - Residue: HIS, ID: 57
2026-01-01 10:10:00,001 - MetalDock - INFO - Residue: ASP, ID: 102
2026-01-01 10:10:00,002 - MetalDock - INFO - Pose 2:
2026-01-01 10:10:00,002 - MetalDock - INFO - Binding Energy: -6.1000 kcal/mol
2026-01-01 10:10:00,002 - MetalDock - INFO - Ligand Efficiency: -0.1900 kcal/mol
2026-01-01 10:10:00,003 - MetalDock - INFO - THE PRINTED POSES AND PROTEIN CAN BE FOUND IN THE RESULTS DIRECTORY
2026-01-01 10:10:00,004 - MetalDock - INFO - METALDOCK SUCCESFULLY COMPLETED
"""


def test_parse_results():
    poses, rmsd = m.parse_results(LOG)
    assert [p["pose"] for p in poses] == [1, 2]
    assert poses[0]["energy"] == -7.3412 and poses[0]["efficiency"] == -0.228
    assert poses[0]["residues"] == [("HIS", "57"), ("ASP", "102")]
    assert poses[1]["residues"] == []
    assert rmsd == []


def test_show_results_success(tmp_path, pdb_file, xyz_file, capsys):
    s = make_settings(pdb_file, xyz_file, jobdir=str(tmp_path))
    (tmp_path / "MetalDock.log").write_text(LOG)
    m.show_results(s)
    out = capsys.readouterr().out
    assert "-7.34" in out and "HIS57" in out and "pose_N" in out


def test_show_results_failure_shows_log_end(tmp_path, pdb_file, xyz_file, capsys):
    s = make_settings(pdb_file, xyz_file, jobdir=str(tmp_path))
    (tmp_path / "MetalDock.log").write_text(
        "2026-01-01 10:00:00,000 - MetalDock - INFO - STARTING METALDOCK\n"
        "2026-01-01 10:01:00,000 - MetalDock - INFO - GEOMETRY OPTIMIZATION NOT CONVERGED\n")
    m.show_results(s)
    out = capsys.readouterr().out
    assert "did not finish" in out and "NOT CONVERGED" in out and "resume" in out


def test_show_results_before_anything_ran(tmp_path, pdb_file, xyz_file, capsys):
    s = make_settings(pdb_file, xyz_file, jobdir=str(tmp_path))
    m.show_results(s)
    assert "did not finish" in capsys.readouterr().out


# ---- SLURM ---------------------------------------------------------------------------------------------------

def fake_sbatch(monkeypatch, returncode, text=""):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, "", text))


def test_check_submission(pdb_file, xyz_file, monkeypatch):
    s = make_settings(pdb_file, xyz_file, time_limit="1:00:00")
    fake_sbatch(monkeypatch, 0)
    assert m.check_submission(s).ok is True
    fake_sbatch(monkeypatch, 1, "sbatch: error: invalid partition specified: ncpu")
    result = m.check_submission(s)
    assert result.ok is False and "does not exist" in result.reason
    fake_sbatch(monkeypatch, 1, "sbatch: error: Batch job submission failed: Unable to contact slurm controller")
    assert m.check_submission(s).ok is None
    fake_sbatch(monkeypatch, 1, "sbatch: error: Memory specification can not be satisfied")
    assert "cores or this much memory" in m.check_submission(s).reason


def test_explain_submit_error():
    assert "time limit" in m.explain_submit_error("error: Requested time limit is invalid", "ncpu")
    assert "account" in m.explain_submit_error("error: Invalid account or account/partition combination", "ncpu")
    assert m.explain_submit_error("sbatch: error: something odd happened", "x") == "something odd happened"


def test_job_state_never_mistakes_an_error_for_a_finished_job(monkeypatch):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "squeue: error: Unable to contact slurm controller")
    monkeypatch.setattr(m.subprocess, "run", run)
    assert m.job_state("123")[0] == "error"
    monkeypatch.setattr(m.subprocess, "run", lambda c, **k: subprocess.CompletedProcess(c, 1, "", "Invalid job id specified"))
    assert m.job_state("123")[0] == "gone"
    monkeypatch.setattr(m.subprocess, "run", lambda c, **k: subprocess.CompletedProcess(c, 0, "RUNNING|1:02\n", ""))
    assert m.job_state("123") == ("state", ["RUNNING", "1:02"])
    monkeypatch.setattr(m.subprocess, "run", lambda c, **k: subprocess.CompletedProcess(c, 0, "", ""))
    assert m.job_state("123")[0] == "gone"


# ---- the whole wizard, with scripted answers -----------------------------------------------------------------

def script_input(monkeypatch, answers):
    answers = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise AssertionError(f"the wizard asked for more answers than scripted: {prompt!r}")
    monkeypatch.setattr(builtins, "input", fake_input)
    return answers


def no_sbatch(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(m.shutil, "which", lambda name: None if name == "sbatch" else real(name))


def test_wizard_save_mode(tmp_path, pdb_file, xyz_file, monkeypatch, capsys):
    """Residue-based site, ORCA quick, single point, save only. Checks the files that come out."""
    no_sbatch(monkeypatch)
    job = tmp_path / "wizjob"
    script_input(monkeypatch, [
        pdb_file, "", "",                    # protein, pH
        xyz_file, "0", "0",                  # complex, charge, spin
        "1", "2", "1",                       # quick level, single point, vacant site
        "1", "A10", "", "1", "",             # site: residue A10, default box, quick search, no expert options
        "2", "", "", str(job),               # save only (without sbatch: 1 = run, 2 = save), cores, memory, folder
        "", "", "",                          # time limit, job name, change partition? (no)
        "",                                  # confirm
    ])
    m.wizard()
    out = capsys.readouterr().out
    assert os.path.isfile(job / "input.ini")
    assert os.path.isfile(job / "my_protein.pdb")
    assert [f for f in os.listdir(job) if f.startswith("submit_")]
    config = configparser.ConfigParser(interpolation=None)
    config.read(job / "input.ini")
    assert config["QM"]["orcasimpleinput"] == "PBE def2-SVP"
    assert config["METAL_COMPLEX"]["geom_opt"] == "False"
    assert config["METAL_COMPLEX"]["vacant_site"] == "True"
    assert config["DOCKING"]["num_poses"] == "3"
    assert float(config["DOCKING"]["dock_x"]) == pytest.approx(6.4, abs=0.01)
    assert "Submit it later with" in out


def test_wizard_rejects_inconsistent_spin_then_accepts(tmp_path, pdb_file, xyz_file, monkeypatch, capsys):
    no_sbatch(monkeypatch)
    job = tmp_path / "wizjob2"
    script_input(monkeypatch, [
        pdb_file, "", "",
        xyz_file, "1", "0",                  # charge +1 with spin 0 is impossible (103 electrons)
        "no",                                # do not change the charge...
        "1",                                 # ...so try spin 1
        "1", "2", "2",                       # quick level, single point, full coordination sphere
        "2", "5 5 5", "", "1", "",           # typed coordinates (outside the protein: warns), quick search
        "2", "2", "", str(job),              # save only, 2 cores, memory, folder
        "", "", "",                          # time limit, job name, change partition? (no)
        "",                                  # confirm
    ])
    m.wizard()
    out = capsys.readouterr().out
    assert "103 electrons" in out and "must be an odd" in out
    assert "103 electrons, 1 unpaired: consistent" in out
    config = configparser.ConfigParser(interpolation=None)
    config.read(job / "input.ini")
    assert config["METAL_COMPLEX"]["spin"] == "1" and config["METAL_COMPLEX"]["charge"] == "1"
    assert config["DOCKING"]["dock_x"] == "5"


def test_wizard_cancel_runs_nothing(tmp_path, pdb_file, xyz_file, monkeypatch, capsys):
    no_sbatch(monkeypatch)
    job = tmp_path / "cancelled"
    script_input(monkeypatch, [
        pdb_file, "", "", xyz_file, "", "",
        "1", "2", "1",
        "1", "A10", "", "2", "",
        "2", "", "", str(job),               # save only, cores, memory, folder
        "", "", "",                          # time limit, job name, change partition? (no)
        "no",                                # do not save
    ])
    m.wizard()
    assert not job.exists()
    assert "Cancelled. Nothing was run." in capsys.readouterr().out


def test_wizard_ctrl_c_runs_nothing(tmp_path, pdb_file, monkeypatch, capsys):
    def interrupt(prompt=""):
        raise KeyboardInterrupt
    monkeypatch.setattr(builtins, "input", interrupt)
    with pytest.raises(SystemExit) as exc:
        m.wizard()
    assert exc.value.code == 130
    assert "Nothing was run" in capsys.readouterr().out


# ---- entry point ---------------------------------------------------------------------------------------------

def test_classic_arguments_go_to_the_original_program(monkeypatch):
    called = []
    import src.metal_dock.main as classic
    monkeypatch.setattr(classic, "main", lambda: called.append("classic"))
    monkeypatch.setattr(m, "wizard", lambda: called.append("wizard"))
    monkeypatch.setattr(m.sys, "argv", ["metaldock_cli.py", "-i", "input.ini", "-m", "dock"])
    m.main()
    assert called == ["classic"]
    called.clear()
    monkeypatch.setattr(m.sys, "argv", ["metaldock_cli.py"])
    m.main()
    assert called == ["wizard"]
