PYTHON ?= python3
.PHONY: all paper editorial figures test clean

all: paper editorial test

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp paper/main.pdf paper/inference_as_a_new_frontier.pdf

editorial: paper
	$(PYTHON) scripts/build_editorial.py

figures:
	$(PYTHON) paper/generate_authentic_figures.py
	$(PYTHON) paper/recreate_figures_7_9.py
	$(PYTHON) paper/generate_integrated_figures.py

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	cd paper && latexmk -c main.tex
