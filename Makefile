.PHONY: fmt lint train eval package

fmt:
	python -m pip install ruff black
	ruff check --fix .
	black .

train:
	python scripts/train.py --config scripts/local_example.yaml

eval:
	python scripts/evaluate.py --config scripts/local_example.yaml

package:
	python scripts/package_model.py --artifacts out/model --output out/model.tar.gz
