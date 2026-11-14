#!/usr/bin/env python3
"""Guided command-line interface for MetalDock.

Start it with:  ddsuite metaldock          (or, on its own, see README.md)

It asks for the protein and the metal complex, checks them, helps to choose the quantum-chemistry level and the
docking site, writes a self-contained job folder (input.ini plus copies of your files), and then submits the job to
SLURM, runs it in the terminal, or just saves the job script. It can watch a submitted job and summarise the poses
when it finishes.

Give it the classic arguments and it behaves exactly like the original program:
    ddsuite metaldock -i input.ini -m dock
"""
import argparse
import getpass
import glob
import hashlib
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

try:
    import readline
except ImportError:  # no Tab completion, everything else works
    readline = None

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

# The lab's launcher (ddsuite) sets DDSUITE_RUN_COMMAND, e.g. "ddsuite metaldock". It is used in messages that tell
# the user how to start this tool.
UNDER_DDSUITE = bool(os.environ.get("DDSUITE_RUN_COMMAND"))
RUN_COMMAND = os.environ.get("DDSUITE_RUN_COMMAND") or "metaldock"

MAIL_DOMAIN = "crick.ac.uk"
DEFAULT_PARTITION = "ncpu"
DEFAULT_ORCA_MODULES = "ORCA/6.1.0,OpenMPI/4.1.6-GCC-13.2.0"  # ORCA only runs in parallel with OpenMPI loaded
SUBMIT_CHECK_SECONDS = 30
POLL_SECONDS = 10
MAX_WATCH_ERRORS = 6
ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING"}
TOTAL_STEPS = 5

WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "TIP3", "SOL"}
NUCLEIC_NAMES = {"DA", "DC", "DG", "DT", "A", "C", "G", "U", "DU", "RA", "RC", "RG", "RU"}

# Metals MetalDock has docking parameters for. The first group has parameters fitted by the authors; the second
# uses AutoDock's own parameters; iridium is in the parameter file but has no fitted values (5.0 kcal/mol).
FITTED_METALS = ["V", "Cr", "Co", "Ni", "Cu", "Mo", "Ru", "Rh", "Pd", "Re", "Os", "Pt"]
AUTODOCK_METALS = ["Fe", "Zn", "Mn"]
UNFITTED_METALS = ["Ir"]
ALL_METALS = FITTED_METALS + AUTODOCK_METALS + UNFITTED_METALS

# AutoGrid cannot use a box with more than about 256 grid points per side; MetalDock uses 2.66 points per Angstrom.
MAX_BOX_SIZE = 90.0
MIN_BOX_SIZE = 5.0
LARGE_BOX_SIZE = 40.0

# (label, orcasimpleinput, explanation)
ORCA_PRESETS = [
    ("quick", "PBE def2-SVP",
     "Quick check: a small basis set. Minutes to an hour. Good for testing the workflow, not for results."),
    ("standard", "PBE def2-TZVP CPCM(Water)",
     "MetalDock's default: PBE with a triple-zeta basis and water as solvent. A sensible starting point."),
    ("hybrid", "B3LYP D3BJ def2-TZVP",
     "B3LYP with dispersion correction, as in the MetalDock examples. Slower, often hours for a geometry optimisation."),
]

# (name, number of docking runs = poses, maximum energy evaluations per run, explanation)
EFFORT_PRESETS = [
    ("quick", 3, 100000, "3 poses, short search. A few minutes. For testing."),
    ("standard", 10, 2500000, "10 poses, MetalDock's default search. Usually well under an hour."),
    ("thorough", 20, 5000000, "20 poses, twice as long a search. For difficult or large sites."),
]

# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
WARNINGS = []


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(text):
    return _c("1", text)


def dim(text):
    return _c("2", text)


def say(msg=""):
    print(msg)


def ok(msg):
    print(f"  {_c('32', '[ok]')} {msg}")


def warn(msg, record=True):
    """Print a warning. Recorded warnings are repeated in the final review."""
    if record:
        WARNINGS.append(msg)
    print(f"  {_c('33', '[!]')}  {msg}")


def fail(msg):
    print(f"  {_c('31', '[x]')}  {msg}")


def note(msg):
    print(f"       {dim(msg)}")


def heading(step, title):
    text = f"Step {step} of {TOTAL_STEPS}: {title}"
    print(f"\n{bold(text)}\n{'-' * len(text)}")


def table(rows, headers=None, indent="  "):
    rows = [[str(c) for c in r] for r in rows]
    if headers:
        rows = [list(headers)] + rows
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for n, r in enumerate(rows):
        print(indent + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if headers and n == 0:
            print(indent + "  ".join("-" * w for w in widths))


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
def _set_completer(func):
    if readline is None:
        return
    readline.set_completer(func)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def path_completer(text, state):
    text = os.path.expanduser(text)
    matches = glob.glob(text + "*")
    matches = [m + "/" if os.path.isdir(m) else m for m in matches]
    try:
        return sorted(matches)[state]
    except IndexError:
        return None


def choice_completer(choices):
    def completer(text, state):
        matches = [c for c in choices if c.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None
    return completer


def ask(label, default=None, validate=None, completer=None, required=True):
    """Prompt until the answer is valid. validate() may raise ValueError."""
    _set_completer(completer)
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        value = raw if raw else default
        if value in (None, ""):
            if required:
                print("       Please enter a value.")
                continue
            return None
        if validate is not None:
            try:
                value = validate(value)
            except ValueError as exc:
                fail(str(exc))
                continue
        return value


def ask_yes_no(label, default=False):
    def validate(text):
        t = str(text).lower()
        if t in ("y", "yes", "true"):
            return True
        if t in ("n", "no", "false"):
            return False
        raise ValueError("Please answer yes or no.")
    return ask(f"{label} (yes/no)", "yes" if default else "no", validate, choice_completer(["yes", "no"]))


def ask_number(label, default, cast=float, minimum=None, maximum=None, required=True):
    def validate(text):
        try:
            value = cast(text)
        except (TypeError, ValueError):
            raise ValueError("Please enter a number.")
        if minimum is not None and value < minimum:
            raise ValueError(f"Please enter a number of at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Please enter a number of at most {maximum}.")
        return value
    return ask(label, None if default is None else str(default), validate, required=required)


def ask_option(options, default=1, label="Choose a number"):
    """Show a numbered list of (key, text) pairs and return the key chosen."""
    for n, (_, text) in enumerate(options, 1):
        say(f"    {n}. {text}")

    def pick(text):
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1][0]
        raise ValueError(f"Please enter a number from 1 to {len(options)}.")
    return ask(label, str(default), pick)


def clean_path(text):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(text.strip().strip("'\""))))


def file_validator(extensions=()):
    def validate(text):
        path = clean_path(text)
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")
        if extensions and not path.lower().endswith(extensions):
            raise ValueError(f"Expected a {' or '.join(extensions)} file.")
        if os.path.getsize(path) == 0:
            raise ValueError("That file is empty.")
        return path
    return validate


def time_validator(text):
    if not re.fullmatch(r"(\d+-)?\d+(:\d{2}){0,2}", text):
        raise ValueError("Use hours:minutes:seconds, for example 12:00:00 (or days-hours:minutes:seconds).")
    return text


def fmt_minutes(minutes):
    minutes = int(math.ceil(minutes))
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00" if days else f"{hours}:{mins:02d}:00"


def safe_name(text, limit=40):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)[:limit] or "run"


# --------------------------------------------------------------------------
# The metal complex (XYZ file)
# --------------------------------------------------------------------------
@dataclass
class ComplexInfo:
    path: str
    atoms: list                                  # (symbol, x, y, z)
    comment: str = ""
    counts: dict = field(default_factory=dict)   # symbol -> number of atoms
    metals: list = field(default_factory=list)   # supported metals found, in the order of first appearance
    other_metals: list = field(default_factory=list)  # transition metals found that MetalDock cannot dock
    min_distance: float = 0.0
    has_hydrogen: bool = True
    electrons: int = 0                           # of the neutral complex

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def formula(self):
        order = sorted(self.counts, key=lambda s: (s != "C", s != "H", s))
        return "".join(f"{s}{self.counts[s] if self.counts[s] > 1 else ''}" for s in order)


def _normalise_symbol(text):
    # Some programs write "Ru1" or "RU" or a number; keep the leading letters, capitalised like a symbol.
    letters = re.match(r"[A-Za-z]+", text)
    if not letters:
        raise ValueError(f"cannot read the element '{text}'")
    word = letters.group(0)
    return word[0].upper() + word[1:2].lower() if len(word) > 1 else word.upper()


def read_xyz(path):
    """Read an XYZ file. Raises ValueError with a plain-language reason."""
    from ase.data import atomic_numbers
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 3:
        raise ValueError("The file has fewer than three lines. An XYZ file has the atom count, a comment line, "
                         "then one line per atom.")
    try:
        declared = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError("The first line must be the number of atoms. This does not look like an XYZ file.")
    atoms = []
    for number, line in enumerate(lines[2:], 3):
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Line {number} should be 'element x y z' but is: {line.strip()[:50]}")
        try:
            symbol = _normalise_symbol(parts[0])
            xyz = [float(v) for v in parts[1:4]]
        except ValueError as exc:
            raise ValueError(f"Line {number}: {exc}")
        if symbol not in atomic_numbers:
            raise ValueError(f"Line {number}: '{parts[0]}' is not a chemical element.")
        atoms.append((symbol, *xyz))
    if len(atoms) != declared:
        raise ValueError(f"The first line says {declared} atoms but the file has {len(atoms)}.")
    counts = {}
    for atom in atoms:
        counts[atom[0]] = counts.get(atom[0], 0) + 1
    info = ComplexInfo(path=path, atoms=atoms, comment=lines[1].strip(), counts=counts)
    info.metals = [m for m in ALL_METALS if m in counts]
    info.other_metals = [s for s in counts if atomic_numbers[s] > 20 and s not in ALL_METALS
                         and s not in ("Br", "Kr", "Rb", "Sr", "I", "Xe", "As", "Se", "Ge", "Ga")]
    info.has_hydrogen = "H" in counts
    info.electrons = sum(atomic_numbers[s] * n for s, n in counts.items())
    dmin = None
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = math.dist(atoms[i][1:], atoms[j][1:])
            dmin = d if dmin is None or d < dmin else dmin
    info.min_distance = dmin or 0.0
    return info


def spin_problem(info, charge, spin):
    """Why this charge and spin cannot describe the complex, or None if they are possible.
    `spin` is the number of unpaired electrons (alpha minus beta), as MetalDock defines it.
    """
    electrons = info.electrons - charge
    if electrons < 1:
        return f"A charge of {charge} would leave {electrons} electrons."
    if spin > electrons:
        return f"{spin} unpaired electrons is more than the {electrons} electrons the complex would have."
    if (electrons - spin) % 2:
        parity = "odd" if electrons % 2 else "even"
        want = "an odd" if electrons % 2 else "an even"
        return (f"With charge {charge} the complex has {electrons} electrons, an {parity} number, so the number "
                f"of unpaired electrons must be {want} number (for example {1 if electrons % 2 else 0}, "
                f"{3 if electrons % 2 else 2}). You gave {spin}.")
    return None


def report_complex(info):
    ok(f"{info.n_atoms} atoms, {info.formula}. Metal: "
       f"{', '.join(info.metals) if info.metals else 'none that MetalDock supports'}.")
    if not info.metals:
        fail("No metal that MetalDock can dock was found in this file. Supported: " + ", ".join(ALL_METALS) + ".")
    if info.other_metals:
        warn(f"The file also contains {', '.join(info.other_metals)}, which MetalDock has no parameters for.")
    if info.min_distance < 0.6:
        warn(f"Two atoms are only {info.min_distance:.2f} A apart. Check that the coordinates are in Angstrom "
             "and that no atoms are duplicated.")
    if not info.has_hydrogen:
        warn("The complex has no hydrogen atoms. If that is not intended, add them before docking: "
             "the quantum-chemistry calculation and the docking charges depend on them.")
    if info.n_atoms > 120:
        warn(f"{info.n_atoms} atoms is large for a DFT geometry optimisation. Consider a single point "
             "calculation on an optimised structure.")
    unfitted = [m for m in info.metals if m in UNFITTED_METALS]
    if unfitted:
        warn(f"{', '.join(unfitted)} has no fitted MetalDock parameters; a default of 5.0 kcal/mol is used for "
             "its interaction with N, O, S and H donors, which is only a rough guess.")


# --------------------------------------------------------------------------
# The protein (PDB file)
# --------------------------------------------------------------------------
@dataclass
class PdbInfo:
    path: str
    n_atoms: int = 0
    chains: list = field(default_factory=list)
    models: int = 1
    hetero: dict = field(default_factory=dict)       # residue name -> number of residues (waters excluded)
    waters: int = 0
    altlocs: bool = False
    hydrogens: bool = False
    nucleic: bool = False
    residues: dict = field(default_factory=dict)     # (chain, number) -> (name, [(x, y, z), ...])
    coords: list = field(default_factory=list)       # all ATOM coordinates

    @property
    def residue_count(self):
        return len(self.residues)


def read_pdb(path):
    info = PdbInfo(path=path)
    chains, waters, hetero_seen = [], set(), {}
    with open(path, errors="replace") as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "MODEL":
                info.models += 1 if info.n_atoms else 0
                continue
            if record not in ("ATOM", "HETATM"):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                raise ValueError(f"Cannot read the coordinates on this line:\n       {line.rstrip()[:70]}")
            name = line[17:20].strip()
            chain = line[21:22].strip() or "_"
            try:
                number = int(line[22:26])
            except ValueError:
                raise ValueError(f"Cannot read the residue number on this line:\n       {line.rstrip()[:70]}")
            insertion = line[26:27].strip()
            altloc = line[16:17].strip()
            element = line[76:78].strip().upper() or re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:1].upper()
            info.n_atoms += 1
            if altloc and altloc not in ("A", "1"):
                info.altlocs = True
            if element == "H":
                info.hydrogens = True
            if name in WATER_NAMES:
                waters.add((chain, number, insertion))
                continue
            key = (chain, number)
            entry = info.residues.setdefault(key, (name, []))
            entry[1].append(xyz)
            if record == "ATOM":
                info.coords.append(xyz)
                if chain not in chains:
                    chains.append(chain)
                if name in NUCLEIC_NAMES:
                    info.nucleic = True
            else:
                hetero_seen.setdefault((chain, number, insertion), name)
    if not info.coords:
        raise ValueError("The file has no ATOM records, so it holds no protein or nucleic acid. "
                         "Is it a PDB file?")
    info.chains = chains
    info.waters = len(waters)
    for name in hetero_seen.values():
        info.hetero[name] = info.hetero.get(name, 0) + 1
    return info


def report_pdb(info):
    kind = "nucleic acid" if info.nucleic else "protein"
    ok(f"{info.n_atoms} atoms, {len(info.chains)} chain(s) ({', '.join(info.chains)}), {kind}.")
    if info.models > 1:
        warn(f"The file holds {info.models} models (for example an NMR ensemble). Keep one model before docking.")
    if info.altlocs:
        warn("Some atoms have alternate locations. PDB2PQR keeps the first; check the result if it matters.")
    if info.hetero:
        listing = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(info.hetero.items()))
        note(f"Other molecules in the file (HETATM): {listing}.")
    if info.waters:
        note(f"{info.waters} water molecules.")
    if not info.hydrogens:
        note("No hydrogens in the file. That is fine: PDB2PQR adds them at the pH you choose.")


RESIDUE_PART = re.compile(r"^(?P<name>[A-Za-z]{3})?(?P<num>-?\d+)$")
RESIDUE_WITH_CHAIN = re.compile(r"^(?P<chain>[A-Za-z])(?P<name>[A-Za-z]{3})?(?P<num>-?\d+)$")


def parse_residue_token(token):
    """(chain or None, residue name or None, number) from 'A57', 'A:57', 'HIS57', 'A:HIS57' or '57'; None if it is
    none of these. A chain that is a digit must be written with a colon ('1:57')."""
    chain = None
    if ":" in token:
        chain, token = token.split(":", 1)
        if not re.fullmatch(r"[A-Za-z0-9]", chain):
            return None
    found = RESIDUE_PART.match(token)
    if found is None and chain is None:
        found = RESIDUE_WITH_CHAIN.match(token)
        if found:
            chain = found.group("chain")
    if found is None:
        return None
    return chain, found.group("name"), int(found.group("num"))


def find_residues(info, text):
    """Residues named in text (for example 'A:57, HIS 119, B210'). Raises ValueError listing what was not found."""
    found, missing = [], []
    for token in re.split(r"[,;]", text):
        token = token.strip().replace(" ", "")
        if not token:
            continue
        parsed = parse_residue_token(token)
        if parsed is None:
            raise ValueError(f"'{token}' is not a residue. Use a chain and number such as A57, or a name and "
                             "number such as HIS57; separate several with commas.")
        chain, name, number = parsed
        hits = [(key, val) for key, val in info.residues.items()
                if key[1] == number and (chain is None or key[0].upper() == chain.upper())
                and (name is None or val[0].upper() == name.upper())]
        if not hits:
            missing.append(token)
        found.extend(hits)
    if missing:
        chains = ", ".join(info.chains)
        raise ValueError(f"Not found in the protein: {', '.join(missing)}. Chains in the file: {chains}.")
    if not found:
        raise ValueError("Please name at least one residue.")
    return found


def centre_of(points):
    n = len(points)
    return tuple(round(sum(p[i] for p in points) / n, 3) for i in range(3))


def residue_centre(hits):
    return centre_of([xyz for _, (_, atoms) in hits for xyz in atoms])


def protein_extent(info):
    lo = [min(p[i] for p in info.coords) for i in range(3)]
    hi = [max(p[i] for p in info.coords) for i in range(3)]
    return lo, hi


def inside_protein_box(info, point, margin=5.0):
    lo, hi = protein_extent(info)
    return all(lo[i] - margin <= point[i] <= hi[i] + margin for i in range(3))


# --------------------------------------------------------------------------
# Settings, the input file and the job script
# --------------------------------------------------------------------------
@dataclass
class Settings:
    # files
    pdb_src: str = ""
    xyz_src: str = ""
    pdb: object = None
    cplx: object = None
    # complex
    metal: str = ""
    charge: int = 0
    spin: int = 0
    vacant_site: bool = True
    geom_opt: bool = True
    # protein
    clean_pdb: bool = True
    ph: float = 7.4
    # quantum chemistry
    engine: str = "ORCA"
    qm_level: str = "standard"
    orca_simple: str = "PBE def2-TZVP CPCM(Water)"
    orca_blocks: str = ""
    basis_set: str = "TZP"
    functional_type: str = "GGA"
    functional: str = "PBE"
    dispersion: str = ""
    solvent: str = ""
    relativity: str = ""
    # docking
    centre: object = None            # (x, y, z) or None: the metal position in the XYZ file
    centre_text: str = ""
    box: list = field(default_factory=lambda: [20.0, 20.0, 20.0])
    effort: str = "standard"
    num_poses: int = 10
    num_evals: int = 2500000
    random_pos: bool = True
    ini_parameters: bool = False
    energies: list = field(default_factory=lambda: [5.0, 5.0, 5.0, 5.0])  # e_NA, e_OA, e_SA, e_HD
    rmsd: bool = False
    # running
    mode: str = "submit"
    ncpu: int = 8
    memory: int = 2000
    jobname: str = ""
    jobdir: str = ""
    resume: bool = False
    time_limit: str = "24:00:00"
    partition: str = DEFAULT_PARTITION
    modules: str = DEFAULT_ORCA_MODULES
    pixi_dir: str = ""
    script_path: str = ""


def staged_name(path, suffix):
    """Name of the copy of an input file inside the job folder: safe characters and the extension MetalDock needs."""
    stem = safe_name(os.path.splitext(os.path.basename(path))[0], 60)
    return f"{stem}{suffix}"


def box_text(box):
    values = [f"{v:g}" for v in box]
    return values[0] if len(set(values)) == 1 else ",".join(values)


def effective_orca_blocks(s):
    """The ORCA blocks written to the input file: the user's blocks plus the memory per core."""
    blocks = s.orca_blocks.strip()
    if "%maxcore" not in blocks.lower():
        blocks = f"%maxcore {s.memory} {blocks}".strip()
    return blocks


def build_ini(s):
    lines = ["# Written by the MetalDock guided interface. Run it again with:",
             "#   metaldock -i input.ini -m dock",
             "[DEFAULT]",
             f"metal_symbol = {s.metal}",
             "method = dock",
             f"ncpu = {s.ncpu}",
             f"memory = {s.memory}",
             "",
             "[PROTEIN]",
             f"pdb_file = {staged_name(s.pdb_src, '.pdb')}",
             f"pH = {s.ph:g}",
             f"clean_pdb = {s.clean_pdb}",
             "",
             "[QM]",
             f"engine = {s.engine}"]
    if s.engine == "ORCA":
        lines += [f"orcasimpleinput = {s.orca_simple}", f"orcablocks = {effective_orca_blocks(s)}"]
    else:
        lines += [f"basis_set = {s.basis_set}", f"functional_type = {s.functional_type}",
                  f"functional = {s.functional}"]
        if s.dispersion:
            lines.append(f"dispersion = {s.dispersion}")
        if s.solvent:
            lines.append(f"solvent = {s.solvent}")
        if s.engine == "ADF" and s.relativity:
            lines.append(f"relativity = {s.relativity}")
    lines += ["",
              "[METAL_COMPLEX]",
              f"geom_opt = {s.geom_opt}",
              f"xyz_file = {staged_name(s.xyz_src, '.xyz')}",
              f"charge = {s.charge}",
              f"spin = {s.spin}",
              f"vacant_site = {s.vacant_site}",
              "",
              "[DOCKING]"]
    if s.centre is not None:
        lines += [f"dock_x = {s.centre[0]:g}", f"dock_y = {s.centre[1]:g}", f"dock_z = {s.centre[2]:g}"]
    lines += [f"box_size = {box_text(s.box)}",
              f"num_poses = {s.num_poses}",
              f"ga_dock_num_evals = {s.num_evals}",
              f"random_pos = {s.random_pos}",
              f"rmsd = {s.rmsd}"]
    if s.ini_parameters:
        lines += ["ini_parameters = True", f"e_NA = {s.energies[0]:g}", f"e_OA = {s.energies[1]:g}",
                  f"e_SA = {s.energies[2]:g}", f"e_HD = {s.energies[3]:g}"]
    return "\n".join(lines) + "\n"


def qm_is_quick(s):
    return s.engine == "ORCA" and s.qm_level == "quick"


def estimate_minutes(s):
    """A generous guess of the run time, in minutes. QM dominates; the docking itself takes minutes to an hour."""
    if s.geom_opt:
        qm = 6 * 60 if qm_is_quick(s) else 24 * 60
    else:
        qm = 60 if qm_is_quick(s) else 4 * 60
    dock = {"quick": 30, "standard": 120, "thorough": 240}.get(s.effort, 120)
    return min(qm + dock, 7 * 24 * 60)


def mail_address():
    try:
        user = getpass.getuser()
    except Exception:
        return None
    return f"{user}@{MAIL_DOMAIN}" if re.fullmatch(r"[A-Za-z0-9._-]+", user) else None


QM_ENVIRONMENT = {"ADF": ["AMSHOME", "AMSBIN", "AMSRESOURCES", "SCMLICENSE"], "Gaussian": ["g16root", "GAUSS_EXEDIR"]}


def command_line(s):
    """The command that runs MetalDock in the job folder."""
    pixi = shutil.which("pixi")
    manifest = os.path.join(REPO_DIR, "pixi.toml")
    if pixi:
        return (f"pixi run --manifest-path {shlex.quote(manifest)} --as-is python "
                f"{shlex.quote(os.path.join(REPO_DIR, 'metaldock'))} -i input.ini -m dock")
    return f"{shlex.quote(sys.executable)} {shlex.quote(os.path.join(REPO_DIR, 'metaldock'))} -i input.ini -m dock"


def build_script(s):
    mail = mail_address()
    lines = ["#!/bin/bash",
             f"#SBATCH --partition={s.partition}",
             "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={s.ncpu}",
             f"#SBATCH --mem-per-cpu={int(s.memory * 1.3)}M",
             f"#SBATCH --job-name={s.jobname}",
             f"#SBATCH --output={os.path.join(s.jobdir, s.jobname)}_%j.log",
             f"#SBATCH --time={s.time_limit}",
             *(["#SBATCH --mail-type=END,FAIL", f"#SBATCH --mail-user={mail}"] if mail else []),
             ""]
    for module in s.modules.split(","):
        if module.strip():
            lines.append(f"module load {module.strip()}")
    for name in QM_ENVIRONMENT.get(s.engine, []):
        if os.environ.get(name):
            lines.append(f"export {name}={shlex.quote(os.environ[name])}")
    pixi = shutil.which("pixi")
    if pixi:
        lines += ["", f'export PATH="{os.path.dirname(pixi)}:$PATH"']
    lines += ["", f"cd {shlex.quote(s.jobdir)}", "", command_line(s), ""]
    return "\n".join(lines)


def file_digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def same_file(a, b):
    return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)


def folder_only_holds(folder, files):
    """True when everything in the folder is one of the given files (a data folder the user points us at)."""
    names = {os.path.basename(f) for f in files}
    return all(entry in names for entry in os.listdir(folder))


def previous_run(s, folder):
    """None if the folder holds no earlier run of ours, else (differences, finished). Resuming needs identical input."""
    ini = os.path.join(folder, "input.ini")
    if not os.path.isfile(ini):
        return None
    differences = []
    with open(ini) as fh:
        if fh.read() != build_ini(s):
            differences.append("different settings")
    for src, suffix in ((s.pdb_src, ".pdb"), (s.xyz_src, ".xyz")):
        staged = os.path.join(folder, staged_name(src, suffix))
        if not os.path.isfile(staged) or file_digest(staged) != file_digest(src):
            differences.append(f"a different {os.path.basename(src)}")
    finished = os.path.isfile(os.path.join(folder, "output", "results", "docking_results.dlg"))
    return differences, finished


def validate_folder(path, protected):
    if os.path.isfile(path):
        return "That is a file, not a folder."
    for p in protected:
        if p and os.path.abspath(p) == path:
            return f"Please do not use {path} itself; choose a new folder inside it."
    parent = os.path.dirname(path)
    probe = parent
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not probe or not os.access(probe, os.W_OK):
        return f"You cannot write to {probe or parent}."
    if path.startswith(os.path.abspath(REPO_DIR) + os.sep):
        return "Please choose a folder outside the MetalDock installation."
    return None


def write_job_folder(s):
    os.makedirs(s.jobdir, exist_ok=True)
    for src, suffix in ((s.pdb_src, ".pdb"), (s.xyz_src, ".xyz")):
        target = os.path.join(s.jobdir, staged_name(src, suffix))
        if not same_file(src, target):
            shutil.copyfile(src, target)
    with open(os.path.join(s.jobdir, "input.ini"), "w") as fh:
        fh.write(build_ini(s))
    ok(f"Job folder ready: {s.jobdir}")
    note("It holds input.ini and copies of your two files; your originals are not touched.")


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------
def check_environment():
    missing = [name for name in ("obabel", "pdb2pqr30") if not os.path.exists(os.path.join(sys.prefix, "bin", name))]
    for rel in ("src/external/AutoDock/autodock4", "src/external/AutoDock/autogrid4"):
        if not os.access(os.path.join(REPO_DIR, rel), os.X_OK):
            missing.append(os.path.basename(rel))
    if missing:
        fail(f"This installation is incomplete (missing: {', '.join(missing)}).")
        say("  Ask a maintainer to run `ddsuite update metaldock`, or on your own copy run `pixi install`.")
        sys.exit(1)


def engine_status():
    """{engine: (usable here, note)}"""
    orca = shutil.which("orca")
    return {
        "ORCA": (bool(orca), "found on this computer" if orca else "free; the cluster job loads the ORCA module"),
        "ADF": (bool(os.environ.get("AMSBIN")), "set up in this shell" if os.environ.get("AMSBIN")
                else "needs AMSHOME, AMSBIN, AMSRESOURCES and SCMLICENSE (licensed)"),
        "Gaussian": (bool(os.environ.get("g16root")), "set up in this shell" if os.environ.get("g16root")
                     else "needs g16root (licensed)"),
    }


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def step_files(s):
    heading(1, "Your files")
    say("  MetalDock docks one metal complex into one protein (or DNA). You need:")
    say("    - the protein as a PDB file, and")
    say("    - the metal complex as an XYZ file (3D coordinates, in Angstrom).")
    say("  Press Tab to complete file names.\n")
    s.pdb_src = ask("Protein PDB file", None, file_validator((".pdb",)), path_completer)
    while True:
        try:
            s.pdb = read_pdb(s.pdb_src)
        except ValueError as exc:
            fail(str(exc))
            s.pdb_src = ask("Protein PDB file", None, file_validator((".pdb",)), path_completer)
            continue
        break
    report_pdb(s.pdb)
    if len(s.pdb.chains) > 1:
        note("MetalDock docks onto every chain in the file. Delete chains you do not want from the PDB first.")
    s.clean_pdb = True
    others = s.pdb.hetero
    if others:
        listing = ", ".join(sorted(others))
        say(f"\n  The PDB file also holds other molecules: {listing}.")
        say("  Normally these (and waters) are removed so only the protein is docked into.")
        if ask_yes_no("Keep them, for example a cofactor or a metal ion the complex should see?", False):
            s.clean_pdb = False
            warn("Other molecules are kept, so waters are not removed either. Delete waters and anything you do "
                 "not want from the PDB file yourself.")
    s.ph = ask_number("pH for protonating the protein (PDB2PQR + PROPKA)", 7.4, float, 0, 14)
    while True:
        s.xyz_src = ask("Metal complex XYZ file", None, file_validator((".xyz",)), path_completer)
        try:
            s.cplx = read_xyz(s.xyz_src)
        except ValueError as exc:
            fail(str(exc))
            continue
        report_complex(s.cplx)
        if not s.cplx.metals:
            continue
        break
    if len(s.cplx.metals) == 1:
        s.metal = s.cplx.metals[0]
    else:
        say(f"\n  The complex contains several metals: {', '.join(s.cplx.metals)}.")
        s.metal = ask("Which metal is docked (the centre MetalDock treats as the metal)", s.cplx.metals[0],
                      lambda t: _pick_metal(t, s.cplx.metals), choice_completer(s.cplx.metals))
    say("")
    s.charge = ask_number("Total charge of the complex", 0, int)
    note("Spin means the number of unpaired electrons: 0 for a closed-shell (singlet) complex, 1 for a doublet, "
         "2 for a triplet.")
    while True:
        s.spin = ask_number("Spin (unpaired electrons)", 0, int, 0)
        problem = spin_problem(s.cplx, s.charge, s.spin)
        if problem is None:
            ok(f"{s.cplx.electrons - s.charge} electrons, {s.spin} unpaired: consistent.")
            break
        fail(problem)
        if ask_yes_no("Change the charge instead?", False):
            s.charge = ask_number("Total charge of the complex", s.charge, int)
            if spin_problem(s.cplx, s.charge, s.spin) is None:
                ok("Consistent now.")
                break


def _pick_metal(text, metals):
    for m in metals:
        if m.lower() == text.lower():
            return m
    raise ValueError(f"Choose one of: {', '.join(metals)}")


def step_qm(s):
    heading(2, "Quantum chemistry")
    say("  MetalDock uses a quantum-chemistry program to get the complex's geometry and atomic charges. This is\n"
        "  the slow part of a run.\n")
    status = engine_status()
    options = [(name, f"{name}: {status[name][1]}") for name in ("ORCA", "ADF", "Gaussian")]
    say("  Which program?")
    s.engine = ask_option(options, 1)
    if s.engine == "ORCA":
        say("\n  Level of theory:")
        presets = [(p[0], f"{p[1]:<28} {p[2]}") for p in ORCA_PRESETS] + [("custom", "Something else (type it)")]
        choice = ask_option(presets, 2)
        if choice == "custom":
            s.qm_level = "custom"
            note("Type what you would write after the '!' in an ORCA input, for example: B3LYP def2-TZVP")
            s.orca_simple = ask("ORCA keywords", "PBE def2-TZVP", lambda t: t.lstrip("! ").strip())
            note("Optional ORCA blocks (the '%' parts), on one line. Do not give the number of cores; that is set "
                 "below. Press Enter for none.")
            s.orca_blocks = ask("ORCA blocks", "none", lambda t: "" if t.lower() == "none" else t.strip(),
                                required=False) or ""
            if "%pal" in s.orca_blocks.lower():
                warn("Your blocks contain %pal. MetalDock adds the number of cores itself; remove it.")
        else:
            s.qm_level = choice
            s.orca_simple = next(p[1] for p in ORCA_PRESETS if p[0] == choice)
            s.orca_blocks = ""
        s.modules = DEFAULT_ORCA_MODULES
    else:
        s.qm_level = "standard"
        s.modules = ""
        if not status[s.engine][0]:
            warn(f"{s.engine} does not seem to be set up in this shell. The run will fail unless the job can "
                 f"find it. {status[s.engine][1].capitalize()}.")
        s.basis_set = ask("Basis set", s.basis_set)
        s.functional_type = ask("Functional type (for example GGA, hybrid)", s.functional_type)
        s.functional = ask("Functional", s.functional)
        s.dispersion = ask("Dispersion correction (none for none)", "none", lambda t: "" if t.lower() == "none" else t,
                           required=False) or ""
        s.solvent = ask("Solvent model (none for none)", "none", lambda t: "" if t.lower() == "none" else t,
                        required=False) or ""
        if s.engine == "ADF":
            s.relativity = ask("Relativistic treatment (none for none, or ZORA)", "none",
                               lambda t: "" if t.lower() == "none" else t, required=False) or ""
    say("\n  Should the complex's geometry be optimised first?")
    options = [(True, "Yes: optimise it. Recommended unless your XYZ already comes from a DFT optimisation. "
                      "This can take many hours."),
               (False, "No: use the geometry as given and only calculate charges (a single point).")]
    s.geom_opt = ask_option(options, 1)
    say("\n  Does the metal have a free coordination site that could bind a protein atom directly?")
    options = [(True, "Yes: a vacant site. Choose this when you expect the metal to bind a residue "
                      "(for example histidine, cysteine, aspartate)."),
               (False, "No: the coordination sphere is full. The complex binds through its ligands, not the metal.")]
    s.vacant_site = ask_option(options, 1)


def step_docking(s):
    heading(3, "Where and how to dock")
    say("  Where on the protein should the complex be placed?\n")
    options = [("residue", "Around a residue I name (recommended when you know the site)"),
               ("coords", "At coordinates I type (x y z in Angstrom, from a viewer such as PyMOL)"),
               ("whole", "Search the whole protein (blind docking: slower and less precise)"),
               ("metal", "Where the metal already sits in the XYZ file (only if you placed the complex yourself)")]
    choice = ask_option(options, 1)
    s.box = [20.0, 20.0, 20.0]
    if choice == "residue":
        while True:
            text = ask("Residue(s), for example A57 or HIS57 or A:57,A:119")
            try:
                hits = find_residues(s.pdb, text)
            except ValueError as exc:
                fail(str(exc))
                continue
            break
        s.centre = residue_centre(hits)
        names = ", ".join(f"{val[0]}{key[1]}{'' if key[0] == '_' else ' chain ' + key[0]}" for key, val in hits[:6])
        s.centre_text = f"centre of {names}{' ...' if len(hits) > 6 else ''}"
        ok(f"Box centre ({s.centre[0]:g}, {s.centre[1]:g}, {s.centre[2]:g}) from {len(hits)} residue(s).")
    elif choice == "coords":
        def parse(text):
            parts = re.split(r"[,\s]+", text.strip())
            if len(parts) != 3:
                raise ValueError("Give three numbers: x y z.")
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                raise ValueError("Give three numbers: x y z.")
        s.centre = ask("Box centre x y z", None, parse)
        s.centre_text = "typed coordinates"
        if not inside_protein_box(s.pdb, s.centre):
            warn("That point is outside the protein. Check the coordinates are in the same frame as the PDB file.")
    elif choice == "whole":
        lo, hi = protein_extent(s.pdb)
        s.centre = centre_of(s.pdb.coords)
        s.centre_text = "centre of the whole protein"
        s.box = [min(MAX_BOX_SIZE, round(hi[i] - lo[i] + 10, 1)) for i in range(3)]
        ok(f"Box centred on the protein, {box_text(s.box)} A on each side.")
        if any(hi[i] - lo[i] + 10 > MAX_BOX_SIZE for i in range(3)):
            warn(f"The protein is larger than the largest box ({MAX_BOX_SIZE:g} A), so parts of it are not searched.")
        warn("Searching a whole protein needs far more sampling than a site. Use the 'thorough' search below and "
             "treat the poses as a first look.")
    else:
        s.centre = None
        s.centre_text = "metal position in the XYZ file"
        metal = next(a for a in s.cplx.atoms if a[0] == s.metal)
        if not inside_protein_box(s.pdb, metal[1:]):
            warn(f"The {s.metal} atom sits at ({metal[1]:g}, {metal[2]:g}, {metal[3]:g}), outside the protein. The "
                 "complex is probably not placed in the protein's frame, so the box would be empty. Choose "
                 "another way of giving the site.")
    if choice != "whole":
        say("\n  Size of the search box (the complex is placed and moved inside it).")
        note(f"One number, or three (x,y,z), in Angstrom, between {MIN_BOX_SIZE:g} and {MAX_BOX_SIZE:g}. 20 suits "
             "a small site.")

        def parse_box(text):
            try:
                values = [float(v) for v in re.split(r"[,\s]+", text.strip())]
            except ValueError:
                raise ValueError("Please enter a number, or three numbers.")
            if len(values) == 1:
                values *= 3
            if len(values) != 3:
                raise ValueError("Please enter one number or three.")
            if any(v < MIN_BOX_SIZE or v > MAX_BOX_SIZE for v in values):
                raise ValueError(f"Each side must be between {MIN_BOX_SIZE:g} and {MAX_BOX_SIZE:g} Angstrom.")
            return values
        s.box = ask("Box size", "20", parse_box)
        if max(s.box) > LARGE_BOX_SIZE:
            warn(f"A box over {LARGE_BOX_SIZE:g} A needs a longer search to be covered well.")
    say("\n  How thorough should the search be?")
    default = 3 if choice == "whole" else 2
    options = [(p[0], f"{p[0]:<9} {p[3]}") for p in EFFORT_PRESETS]
    s.effort = ask_option(options, default)
    preset = next(p for p in EFFORT_PRESETS if p[0] == s.effort)
    s.num_poses, s.num_evals = preset[1], preset[2]
    say("")
    if ask_yes_no("Change expert options (random start positions, metal-atom energies, RMSD)?", False):
        s.random_pos = ask_yes_no("Start every run from a random position in the box?", True)
        s.rmsd = ask_yes_no("Compare the poses with the input geometry (RMSD)? Useful when the XYZ is a known "
                            "bound structure", False)
        if ask_yes_no("Set the metal-atom well depths (kcal/mol) yourself instead of MetalDock's fitted values?",
                      False):
            s.ini_parameters = True
            names = ("metal-N acceptor (NA)", "metal-O acceptor (OA)", "metal-S acceptor (SA)",
                     "metal-H donor (HD)")
            s.energies = [ask_number(f"Well depth {n}", 5.0, float, 0) for n in names]


def step_run(s):
    heading(4, "How to run it")
    has_slurm = shutil.which("sbatch") is not None
    options = []
    if has_slurm:
        options.append(("submit", "Submit to the cluster queue (SLURM). Recommended."))
    options += [("run", "Run now in this terminal (blocks it; only sensible for a quick test on a compute node)"),
                ("save", "Only save the job folder and script; run nothing")]
    if not has_slurm:
        warn("SLURM (sbatch) was not found on this computer, so cluster submission is unavailable.")
    say("  Where should it run?")
    s.mode = ask_option(options, 1)
    stem = safe_name(f"metaldock_{os.path.splitext(os.path.basename(s.xyz_src))[0]}_"
                     f"{os.path.splitext(os.path.basename(s.pdb_src))[0]}", 60)
    s.jobname = stem[:40]
    say("\n  Computing resources. MetalDock runs one calculation at a time; the cores speed up the quantum "
        "chemistry.")
    s.ncpu = ask_number("Number of CPU cores", s.ncpu, int, 1, 128)
    s.memory = ask_number("Memory per core (MB)", s.memory, int, 500)
    if s.engine == "ORCA":
        note(f"Passed to ORCA as %maxcore {s.memory}; the job requests about 30% more, as ORCA can exceed it.")
    say("\n  Everything is kept in one job folder: your input copies, input.ini, the job script and the output.")
    protected = [s.pdb_src, s.xyz_src, os.path.dirname(s.pdb_src), os.path.dirname(s.xyz_src), os.getcwd(),
                 os.path.expanduser("~"), "/", REPO_DIR]
    default = os.path.join(os.getcwd(), stem)
    while True:
        folder = ask("Job folder", default, clean_path)
        problem = validate_folder(folder, protected)
        if problem:
            fail(problem)
            continue
        s.resume = False
        if os.path.isdir(folder) and os.listdir(folder):
            earlier = previous_run(s, folder)
            if earlier is None:
                if folder_only_holds(folder, [s.pdb_src, s.xyz_src]):
                    pass
                else:
                    fail(f"{folder} already holds other files. Choose a new or empty folder; nothing is deleted "
                         "by this program.")
                    default = folder.rstrip("/") + "_2"
                    continue
            else:
                differences, finished = earlier
                if differences:
                    fail(f"{folder} holds an earlier run made with different inputs ({', '.join(differences)}), so it "
                         "cannot be resumed.")
                    default = folder.rstrip("/") + "_2"
                    continue
                state = "It already finished." if finished else "It did not finish."
                say(f"       {folder} holds an earlier run with the same files and settings. {state}")
                say("       MetalDock skips steps whose results already exist (the protein preparation, finished "
                    "quantum-chemistry calculations), so resuming saves time.")
                if ask_yes_no("Resume it?", True):
                    s.resume = True
                else:
                    default = folder.rstrip("/") + "_2"
                    continue
        s.jobdir = folder
        break
    s.jobname = safe_name(os.path.basename(folder), 40)
    if s.mode == "run":
        return
    s.time_limit = fmt_minutes(estimate_minutes(s))
    say(f"\n  Time limit: a generous guess for {'a geometry optimisation' if s.geom_opt else 'a single point'} "
        "with this setup. A job that is killed at the limit loses its running calculation.")
    s.time_limit = ask("Time limit (hours:minutes:seconds)", s.time_limit, time_validator)
    s.jobname = ask("Job name", s.jobname, safe_name)
    pixi = shutil.which("pixi")
    s.pixi_dir = os.path.dirname(pixi) if pixi else ""
    say("\n  The job will use these cluster settings:")
    table([("Partition", s.partition), ("Cores", s.ncpu),
           ("Memory", f"{int(s.memory * 1.3)} MB per core ({int(s.memory * 1.3) * s.ncpu / 1000:.1f} GB total)"),
           ("Modules", s.modules or "none"), ("Email at the end", mail_address() or "none")], indent="    ")
    if ask_yes_no("Change the partition or modules?", False):
        s.partition = ask("Partition", s.partition)
        s.modules = ask("Modules to load (comma-separated, or none)", s.modules or "none",
                        lambda t: "" if t.lower() == "none" else t, required=False) or ""
    ensure_partition_access(s)


# --------------------------------------------------------------------------
# SLURM
# --------------------------------------------------------------------------
@dataclass
class SubmitCheck:
    ok: object          # True, False, or None when it could not be found out
    reason: str = ""


SCHEDULER_UNAVAILABLE = ("unable to contact", "socket timed out", "connection refused", "slurm controller",
                         "temporarily")


def explain_submit_error(text, partition):
    low = text.lower()
    if "invalid partition" in low:
        return f"the partition {partition} does not exist"
    if "invalid account" in low or "account/partition" in low:
        return f"your account cannot use the partition {partition}"
    if "invalid qos" in low or ("qos" in low and "not permitted" in low):
        return f"your account has no access to the quality of service that {partition} needs"
    if "time limit" in low:
        return f"the time limit is longer than the partition {partition} allows"
    if "node configuration is not available" in low or "memory" in low:
        return f"no node of the partition {partition} can give this many cores or this much memory"
    lines = [re.sub(r"^(sbatch: )?(error|allocation failure): ?", "", l.strip(), flags=re.I)
             for l in text.splitlines() if l.strip()]
    return lines[0] if lines else "the scheduler refused the job"


def check_submission(s):
    """Ask Slurm whether this job could be submitted (sbatch --test-only: nothing is queued)."""
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        return SubmitCheck(None, "sbatch was not found")
    command = [sbatch, "--test-only", f"--partition={s.partition}", f"--cpus-per-task={s.ncpu}",
               f"--mem-per-cpu={int(s.memory * 1.3)}M", f"--time={s.time_limit}", "--wrap=true"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SUBMIT_CHECK_SECONDS,
                                stdin=subprocess.DEVNULL, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return SubmitCheck(None, "the scheduler did not answer")
    if result.returncode == 0:
        return SubmitCheck(True)
    text = result.stdout + result.stderr
    if any(word in text.lower() for word in SCHEDULER_UNAVAILABLE):
        return SubmitCheck(None, "the scheduler could not be reached")
    return SubmitCheck(False, explain_submit_error(text, s.partition))


def ensure_partition_access(s):
    if s.mode == "run" or shutil.which("sbatch") is None:
        return
    while True:
        result = check_submission(s)
        if result.ok is True:
            return
        if result.ok is None:
            note(f"Could not check whether this job is accepted: {result.reason}.")
            return
        fail(f"Slurm would refuse this job: {result.reason}.")
        if ask_yes_no("Use these settings anyway?", False):
            warn(f"Slurm will refuse the job: {result.reason}.")
            return
        s.partition = ask("Partition", s.partition)
        s.ncpu = ask_number("Number of CPU cores", s.ncpu, int, 1, 128)
        s.memory = ask_number("Memory per core (MB)", s.memory, int, 500)
        s.time_limit = ask("Time limit (hours:minutes:seconds)", s.time_limit, time_validator)


def latest_job_id(jobname):
    out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-n", jobname, "-o", "%i|%V"],
                         capture_output=True, text=True).stdout
    jobs = [line.split("|") for line in out.splitlines() if "|" in line]
    return max(jobs, key=lambda j: j[1])[0] if jobs else None


def submit(s):
    say("")
    result = subprocess.run(["sbatch", s.script_path])
    if result.returncode != 0:
        fail("sbatch reported an error, so the job was not submitted.")
        why = check_submission(s)
        if why.ok is False:
            say(f"  Reason: {why.reason}. Run the assistant again with other settings.")
        return None
    job_id = latest_job_id(s.jobname)
    if job_id is None:
        warn("Could not find the job in the queue. If you declined at the submit prompt, it was not submitted.")
    return job_id


def job_state(job_id):
    """("state", [state, run time]), ("gone", None) or ("error", message). A failure to reach SLURM is never
    "gone", so a hiccup is not mistaken for a finished job."""
    try:
        result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T|%M"],
                                capture_output=True, text=True, check=False)
    except OSError as exc:
        return "error", str(exc)
    if result.returncode == 0:
        out = result.stdout.strip()
        return ("state", out.split("|")) if out else ("gone", None)
    if "invalid job id" in result.stderr.lower():
        return "gone", None
    return "error", result.stderr.strip() or f"squeue exited with code {result.returncode}"


def job_final_state(job_id):
    try:
        result = subprocess.run(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State,ExitCode"],
                                capture_output=True, text=True, check=False)
    except OSError:
        return None
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    return tuple(lines[0].split("|")[:2]) if lines and "|" in lines[0] else None


def watch(job_id, s):
    say(f"\n  Watching job {job_id}. Press Ctrl+C to stop watching (the job keeps running).\n")
    last_state, last_print, errors = None, 0.0, 0
    try:
        while True:
            kind, data = job_state(job_id)
            if kind == "error":
                errors += 1
                if errors >= MAX_WATCH_ERRORS:
                    warn(f"Could not reach SLURM to check the job ({data}). Stopped watching; "
                         "your job is not affected.", record=False)
                    show_follow_up(job_id, s)
                    return False
                time.sleep(POLL_SECONDS)
                continue
            errors = 0
            if kind == "state":
                now = time.time()
                if data[0] != last_state or now - last_print > 300:
                    label = {"PENDING": "waiting in the queue", "RUNNING": "running"}.get(data[0], data[0].lower())
                    say(f"  {time.strftime('%H:%M:%S')}  {label} (run time {data[1]})")
                    last_state, last_print = data[0], now
                time.sleep(POLL_SECONDS)
                continue
            final = job_final_state(job_id)
            if final and final[0].split()[0] in ACTIVE_STATES:
                time.sleep(POLL_SECONDS)
                continue
            break
    except KeyboardInterrupt:
        say("\n  Stopped watching. The job is still in the queue or running.")
        show_follow_up(job_id, s)
        return False
    say(f"\n  Job finished: {final[0]} (exit code {final[1]})" if final else "\n  Job finished.")
    return True


def show_follow_up(job_id, s):
    log = f"{os.path.join(s.jobdir, s.jobname)}_{job_id}.log"
    say("\n  Useful commands:")
    say(f"    squeue -u {getpass.getuser()}            show your queued and running jobs")
    say(f"    tail -f {os.path.join(s.jobdir, 'MetalDock.log')}   follow MetalDock's progress")
    say(f"    scancel {job_id}                         cancel the job")
    say(f"  If it stops early, resume with the same folder: run `{RUN_COMMAND}` again and choose {s.jobdir}.")
    if mail_address():
        say(f"\n  An email goes to {mail_address()} when the job ends or fails.")
    note(f"SLURM's own log: {log}")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
LOG_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ - \S+ - \w+ - ")


def parse_results(text):
    """Poses from MetalDock.log: [{'pose', 'energy', 'efficiency', 'residues': [(name, id)]}] and any RMSD lines."""
    poses, current, rmsd_lines, in_rmsd = [], None, [], False
    for raw in text.splitlines():
        line = LOG_PREFIX.sub("", raw).strip()
        m = re.match(r"Pose (\d+):", line)
        if m:
            current = {"pose": int(m.group(1)), "energy": None, "efficiency": None, "residues": []}
            poses.append(current)
            in_rmsd = False
            continue
        if current is None:
            continue
        m = re.match(r"Binding Energy:\s*(-?[\d.]+)", line)
        if m:
            current["energy"] = float(m.group(1))
            continue
        m = re.match(r"Ligand Efficiency:\s*(-?[\d.]+)", line)
        if m:
            current["efficiency"] = float(m.group(1))
            continue
        m = re.match(r"Residue:\s*(\S+),\s*ID:\s*(\S+)", line)
        if m:
            current["residues"].append((m.group(1), m.group(2)))
            continue
        if "RMSD" in line.upper() and not line.upper().startswith("THE PRINTED"):
            in_rmsd = True
        if in_rmsd and line and not line.startswith("THE PRINTED") and not line.startswith("#"):
            rmsd_lines.append(line)
        if line.startswith("THE PRINTED"):
            in_rmsd = False
    return poses, rmsd_lines


def log_tail(path, n=15):
    if not os.path.isfile(path):
        return []
    with open(path, errors="replace") as fh:
        lines = [LOG_PREFIX.sub("", l).rstrip() for l in fh.readlines()]
    return [l for l in lines if l.strip()][-n:]


def show_results(s, job_id=None):
    log = os.path.join(s.jobdir, "MetalDock.log")
    text = open(log, errors="replace").read() if os.path.isfile(log) else ""
    poses, rmsd_lines = parse_results(text)
    finished = "SUCCESFULLY COMPLETED" in text or "SUCCESSFULLY COMPLETED" in text
    results = os.path.join(s.jobdir, "output", "results")
    if poses and finished:
        say("\n  Poses (best first, as MetalDock ranks them):")
        rows = [(p["pose"], f"{p['energy']:.2f}" if p["energy"] is not None else "",
                 f"{p['efficiency']:.3f}" if p["efficiency"] is not None else "",
                 ", ".join(f"{n}{i}" for n, i in p["residues"][:8]) + (" ..." if len(p["residues"]) > 8 else ""))
                for p in poses]
        table(rows, headers=("Pose", "Energy (kcal/mol)", "Ligand efficiency", "Residues within 4 A"))
        if rmsd_lines:
            say("\n  RMSD to the input geometry:")
            for line in rmsd_lines[:12]:
                say(f"    {line}")
        energies = [p["energy"] for p in poses if p["energy"] is not None]
        if energies and min(energies) > 0:
            warn("Every pose has a positive energy, which means the complex clashes with the protein wherever it was "
                 "placed. The site is probably too tight or the wrong spot: try a larger box, another residue, or "
                 "check that the complex is not too big for the pocket.", record=False)
        elif energies and max(energies) > 0:
            note("Poses with a positive energy clash with the protein and can be ignored.")
        say(f"\n  Files in {results}:")
        say("    pose_N/                     the Nth pose (XYZ and PDB), for PyMOL or any viewer")
        say("    clean_<protein>.pdb         the protonated protein the poses belong to")
        say("    docking_results.dlg         AutoDock's full log")
        say(f"  The full log is {log}.")
        say("  Energies come from AutoDock with fitted metal parameters: use them to rank poses of one complex, "
            "and treat them as a guide, not as a measured affinity.")
    else:
        fail("MetalDock did not finish. The end of its log:")
        for line in log_tail(log):
            say(f"    {line}")
        if not text:
            say(f"    (no {log} yet)")
        qm_out = glob.glob(os.path.join(s.jobdir, "output", "QM", "*", "*.out"))
        if qm_out:
            say(f"\n  Quantum-chemistry output: {', '.join(os.path.relpath(p, s.jobdir) for p in sorted(qm_out))}")
        say(f"  After fixing the cause, run `{RUN_COMMAND}` again and choose the same folder to resume.")
        if job_id:
            say(f"  SLURM's log: {os.path.join(s.jobdir, s.jobname)}_{job_id}.log")


# --------------------------------------------------------------------------
# Review and execution
# --------------------------------------------------------------------------
def step_review(s):
    heading(5, "Review")
    qm = (f"ORCA: {s.orca_simple}" + (f" {s.orca_blocks}" if s.orca_blocks else "")) if s.engine == "ORCA" else \
        f"{s.engine}: {s.functional_type} {s.functional}/{s.basis_set}"
    complex_text = f"{s.cplx.formula}, {s.cplx.n_atoms} atoms, charge {s.charge}, {s.spin} unpaired electron(s)"
    rows = [("Protein", f"{s.pdb_src} ({s.pdb.n_atoms} atoms, chains {', '.join(s.pdb.chains)}, pH {s.ph:g})"),
            ("Metal complex", f"{s.xyz_src} ({complex_text})"),
            ("Metal", s.metal + (" (free coordination site)" if s.vacant_site else " (full coordination sphere)")),
            ("Quantum chemistry", qm),
            ("Geometry", "optimised first" if s.geom_opt else "used as given (single point)"),
            ("Docking site", s.centre_text + (f" ({s.centre[0]:g}, {s.centre[1]:g}, {s.centre[2]:g})" if s.centre else "")),
            ("Search box", f"{box_text(s.box)} A"),
            ("Search", f"{s.effort}: {s.num_poses} poses, {s.num_evals:,} evaluations each"),
            ("Job folder", s.jobdir + (" (resuming)" if s.resume else "")),
            ("Cores / memory", f"{s.ncpu} / {s.memory} MB per core")]
    if s.mode != "run":
        rows += [("Partition", s.partition), ("Time limit", s.time_limit), ("Job name", s.jobname)]
    if s.engine == "ORCA" and s.modules:
        rows.append(("Modules", s.modules))
    action = {"submit": "Submit the job to the queue", "run": "Run MetalDock now", "save": "Save the job"}[s.mode]
    rows.append(("Action", action))
    table(rows)
    if WARNINGS:
        say(f"\n  {_c('33', 'Warnings to consider:')}")
        for w in dict.fromkeys(WARNINGS):
            say(f"   - {w}")
    say("")
    return ask_yes_no(f"{action}?", True)


def write_script(s):
    write_job_folder(s)
    s.script_path = os.path.join(s.jobdir, f"submit_{s.jobname}.sh")
    with open(s.script_path, "w") as fh:
        fh.write(build_script(s))
    os.chmod(s.script_path, 0o755)
    ok(f"Job script saved: {s.script_path}")


def run_here(s):
    write_job_folder(s)
    if s.engine == "ORCA" and not shutil.which("orca"):
        fail("ORCA is not available in this shell. Run `module load ORCA/6.1.0 OpenMPI/4.1.6-GCC-13.2.0` and start "
             "again.")
        return None
    say("\n  Running... (Ctrl+C to stop)\n")
    try:
        result = subprocess.run([sys.executable, os.path.join(REPO_DIR, "metaldock"), "-i", "input.ini", "-m", "dock"],
                                cwd=s.jobdir)
    except KeyboardInterrupt:
        fail("Stopped.")
        return None
    return result.returncode == 0


def execute(s):
    if s.mode == "run":
        outcome = run_here(s)
        if outcome is None:
            sys.exit(1)
        show_results(s)
        if not outcome:
            sys.exit(1)
        return
    write_script(s)
    if s.mode == "save":
        say(f"\n  Submit it later with:  sbatch {s.script_path}")
        return
    job_id = submit(s)
    if job_id is None:
        return
    ok(f"Submitted job {job_id}.")
    show_follow_up(job_id, s)
    say("")
    if ask_yes_no("Watch the job until it finishes?", False):
        if watch(job_id, s):
            show_results(s, job_id)


def wizard():
    check_environment()
    s = Settings()
    say(bold("\nMetalDock"))
    say("Dock a metal-organic complex into a protein or DNA.")
    say("Answer each question, or press Enter to accept the suggestion in [brackets]. Ctrl+C quits.")
    say(f"Nothing is run until you confirm on the review screen. Experts can still use `{RUN_COMMAND} -i input.ini "
        "-m dock`.")
    started = False
    try:
        step_files(s)
        step_qm(s)
        step_docking(s)
        step_run(s)
        if not step_review(s):
            say("\nCancelled. Nothing was run.")
            return
        started = True
        execute(s)
    except (KeyboardInterrupt, EOFError):
        say("\n\nStopped." + ("" if started else " Nothing was run."))
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        prog=os.environ.get("DDSUITE_RUN_COMMAND"),
        description=f"Docking of metal-organic complexes. Run `{RUN_COMMAND}` with no arguments for the guided "
                    f"interface, or give an input file: `{RUN_COMMAND} -i input.ini -m dock`.")
    parser.add_argument("-i", "--input_file", help="input file in .ini format (skips the guided interface)")
    parser.add_argument("-m", "--method", choices=["dock", "mc"], help="method to use with an input file")
    args, _ = parser.parse_known_args()
    if args.input_file or args.method:
        from src.metal_dock.main import main as classic_main
        classic_main()
        return
    wizard()


if __name__ == "__main__":
    main()
