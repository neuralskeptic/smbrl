# Oneshell means all lines in a recipe run in the same shell
.ONESHELL:

# Need to specify bash in order for conda activate to work
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
# this needs to be run for every target, since they all get separate shells
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate smbrl

yml: FORCE
	$(CONDA_ACTIVATE)
	conda-env-export --conda-all --pip-all --reserve-duplicates --no-prefix -n smbrl
src: FORCE
	$(CONDA_ACTIVATE)
	pip install -e .
quanser_robots: FORCE
	$(CONDA_ACTIVATE)
	cd externals/quanser_robots; pip install -e .
mushroom_rl: FORCE
	$(CONDA_ACTIVATE)
	cd externals/mushroom_rl; pip install --no-use-pep517 -e .[all,mujoco,plots]
experiment_launcher: FORCE
	$(CONDA_ACTIVATE)
	cd externals/experiment_launcher; pip install -e .
i2c: FORCE
	$(CONDA_ACTIVATE)
	cd externals/input-inference-for-control; pip install -r requirements.txt; pip install -e .
externals: quanser_robots mushroom_rl experiment_launcher i2c FORCE

lint: FORCE
	$(CONDA_ACTIVATE)
	black . && isort --gitignore .

.PHONY: yml, src, lint, externals, quanser_robots, mushroom_rl, experiment_launcher, FORCE
FORCE:
