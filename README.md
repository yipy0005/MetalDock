
# MetalDock 

MetalDock is a versatile open-source software docking tool designed to facilitate the docking of metal-organic complexes for various transtions metals with proteins, DNA, or other biomolecules. For detailed instructions on how to utilize MetalDock, we refer you to our [documentation](https://metaldock.readthedocs.io/en/latest/) page.   

# Citation

If you decide to use MetalDock for any project, please cite the following paper: 

[https://doi.org/10.1021/acs.jcim.3c01582](https://doi.org/10.1021/acs.jcim.3c01582)

# Running MetalDock

MetalDock runs in the folder you start it from: it reads the `.ini` file you give it and writes `output/` and
`MetalDock.log` there.

## With the Drug Discovery Suite (lab users)

```bash
cd my_docking_run                          # contains input.ini, the protein .pdb and the complex .xyz
module load ORCA/6.1.0 OpenMPI/4.1.6-GCC-13.2.0   # only for engine = ORCA; ORCA needs OpenMPI when ncpu > 1
ddsuite metaldock -i input.ini -m dock     # docking; add `-m mc` for Monte Carlo parameter optimisation
```

`ddsuite metaldock --help` shows the options. If `ASE_ORCA_COMMAND` is not set, MetalDock uses the `orca` on your
`PATH`, which `module load ORCA` provides. ADF needs `AMSHOME`, `AMSBIN`, `AMSRESOURCES` and `SCMLICENSE`, and
Gaussian needs `g16root`, set as described in the [documentation](https://metaldock.readthedocs.io/en/latest/).

## On its own

The environment is defined in `pixi.toml` (install [pixi](https://pixi.sh) first):

```bash
cd /path/to/MetalDock && pixi install
cd my_docking_run
pixi run --manifest-path /path/to/MetalDock/pixi.toml python /path/to/MetalDock/metaldock -i input.ini -m dock
```
