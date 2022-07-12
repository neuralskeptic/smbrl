yml: FORCE
	conda-env-export --conda-all --pip-all --reserve-duplicates --no-prefix -n smbrl
src: FORCE
	pip install -e .
quanser_robots: FORCE
	cd externals/quanser_robots; pip install -e .
mushroom_rl: FORCE
	cd externals/mushroom_rl; pip install --no-use-pep517 -e .[all,mujoco,plots]
experiment_launcher: FORCE
	cd externals/experiment_launcher; pip install -e .
externals: quanser_robots mushroom_rl experiment_launcher FORCE

lint: FORCE
	black . && isort --gitignore .


.PHONY: yml, src, lint, externals, quanser_robots, mushroom_rl, experiment_launcher, FORCE
FORCE:
