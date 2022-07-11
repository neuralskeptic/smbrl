yml:
	conda-env-export --conda-all --pip-all --reserve-duplicates --no-prefix -n smbrl
src: FORCE
	pip install -e .
externals: FORCE
	cd externals/quanser_robots; pip install -e .
	cd externals/mushroom_rl; pip install --no-use-pep517 -e .[all,mujoco,plots]
	cd externals/experiment_launcher; pip install -e .
lint:
	black . && isort --gitignore .


.PHONY: FORCE
FORCE:
