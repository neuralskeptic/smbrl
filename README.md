# smbrl code

## install instructions
1. `conda env create -f smbrl.yml` # to create conda env and install needed packages
2. `pip install -e .` # to use src/ modules
3. `cd externals/quanser-robots; pip install -e .`
3. `cd externals/mushroom-rl; pip install --no-use-pep517 -e .[all,mujoco,plots]`
4. `pre-commit install` # for linter git hooks

## execute
- check in `scripts/`

## structure
- `src/` - models
- `scripts/` - main scripts to run models on data/simulator/real system
