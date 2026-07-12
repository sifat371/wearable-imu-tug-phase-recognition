.PHONY: install test tug spmt fogstar figures all

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

tug:
	tug-transfer tug --config configs/tug.yaml

spmt:
	tug-transfer spmt --config configs/spmt.yaml

fogstar:
	tug-transfer fogstar --config configs/fogstar.yaml

figures:
	tug-transfer figures --config configs/figures.yaml

all: tug spmt fogstar figures
