# smbrl code

## install instructions
1. `conda env create -f smbrl.yml` # to create conda env and install needed packages
2. `pip install -e .` # to use src/ modules
3. `cd externals/quanser_robots; pip install -e .; cd ../..`
4. `cd externals/experiment_launcher; pip install -e .; cd ../..`
5. `cd externals/mushroom_rl; pip install --no-use-pep517 -e '.[all,mujoco,plots]'; cd ../..`
6. `cd externals/input-inference-for-control; pip install -r requirements.txt; pip install -e .; cd ../..`
7. `pre-commit install` # for linter git hooks

## execute
- check in `scripts/`

## structure
- `src/` - models
- `scripts/` - main scripts to run models on data/simulator/real system
