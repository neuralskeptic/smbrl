# smbrl code

## install instructions
1. `conda create -f smbrl.yml` # to create conda env and install needed packages
2. `pip install -e .` # to use library/ modules
3. `pre-commit install` # for linter git hooks

## execute
- check in `scripts/`

## structure
- `library/` - models
- `experiments/` - simulator/real system
- `scripts/` - main scripts to run models on data/simulator/real system
