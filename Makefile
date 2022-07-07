yml:
	conda-env-export --conda-all --pip-all --reserve-duplicates --no-prefix -n smbrl
src: FORCE
	pip install -e .
externals: FORCE
	cd externals/quanser-robots; pip install -e .
	cd externals/mushroom-rl; pip install --no-use-pep517 -e .[all,mujoco,plots]
lint:
	black . && isort --gitignore .


.PHONY: FORCE
FORCE:
