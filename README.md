# smbrl code

## install instructions
1. `conda env create -f smbrl.yml` # to create conda env and install needed packages
2. `pip install -e .` # to use library/ modules
3. `cd clients; pip install -e .; cd ..` # to install quanser-clients
4. `pre-commit install` # for linter git hooks

## execute
- check in `scripts/`

## structure
- `library/` - models
- `experiments/` - simulator/real system
- `scripts/` - main scripts to run models on data/simulator/real system
