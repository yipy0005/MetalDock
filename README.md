
# MetalDock 

MetalDock is a versatile open-source software docking tool designed to facilitate the docking of metal-organic complexes for various transtions metals with proteins, DNA, or other biomolecules. For detailed instructions on how to utilize MetalDock, we refer you to our [documentation](https://metaldock.readthedocs.io/en/latest/) page.   

# Citation

If you decide to use MetalDock for any project, please cite the following paper: 

[https://doi.org/10.1021/acs.jcim.3c01582](https://doi.org/10.1021/acs.jcim.3c01582)

# Running MetalDock

MetalDock runs in the folder you start it from: it writes `output/` and `MetalDock.log` there.

## Guided interface (recommended if you are new to the command line)

```bash
module load ORCA/6.1.0 OpenMPI/4.1.6-GCC-13.2.0   # for engine = ORCA; needed only if you choose "run now"
ddsuite metaldock                                  # no arguments: the guided interface
```

It asks one question at a time, with a suggested answer in [brackets] (press Enter to accept it), and checks each
answer as you go:

1. **Your files.** Protein PDB and metal complex XYZ. It reads both and reports what it finds (chains, other molecules,
   atoms, metals), offers to keep cofactors or ions, and checks that **charge and spin are possible for the number of
   electrons** (an impossible combination makes the quantum-chemistry job fail). Spin is the number of unpaired electrons.
2. **Quantum chemistry.** ORCA with a quick, standard or hybrid preset (or your own keywords), ADF or Gaussian; whether to
   optimise the geometry; whether the metal has a free coordination site.
3. **Where and how to dock.** Around residues you name (for example `B1307` or `HIS57`), at typed coordinates, over
   the whole protein, or at the metal's position in the XYZ file. It warns if the site is outside the protein. Then a
   box size and a quick, standard or thorough search.
4. **How to run.** Submit to SLURM (default), run in the terminal, or only save. It asks for cores and memory, suggests
   a time limit, checks with SLURM that the job would be accepted, and keeps everything in one **job folder**.
5. **Review**, then confirm. Nothing runs before that.

The job folder holds copies of your two files, `input.ini` (exactly what MetalDock reads, so you can re-run or edit it
with `ddsuite metaldock -i input.ini -m dock`), the job script, and `output/`. Your original files are never changed
and nothing is ever deleted. If the folder already holds an earlier run with the same files and settings you can
**resume** it: MetalDock then skips the steps that already finished, such as the quantum-chemistry calculation.

When a submitted job finishes the interface can show the poses (energy, ligand efficiency, residues within 4 Å), or the
end of the log and the quantum-chemistry output if something went wrong.

## Command line (for experts and scripts)

```bash
ddsuite metaldock -i input.ini -m dock     # docking; use `-m mc` for Monte Carlo parameter optimisation
```

The options are described in the [documentation](https://metaldock.readthedocs.io/en/latest/). If `ASE_ORCA_COMMAND`
is not set, MetalDock uses the `orca` on your `PATH`, which `module load ORCA` provides. ADF needs `AMSHOME`, `AMSBIN`,
`AMSRESOURCES` and `SCMLICENSE`, and Gaussian needs `g16root`.

## On its own, without ddsuite

The environment is defined in `pixi.toml` (install [pixi](https://pixi.sh) first):

```bash
cd /path/to/MetalDock && pixi install
cd my_docking_run
pixi run --manifest-path /path/to/MetalDock/pixi.toml python /path/to/MetalDock/metaldock_cli.py           # guided
pixi run --manifest-path /path/to/MetalDock/pixi.toml python /path/to/MetalDock/metaldock_cli.py -i input.ini -m dock
```

## Tests

The guided interface and the parsing of AutoDock's output have automated tests that need no ORCA or SLURM:

```bash
pixi run -e dev test
```
