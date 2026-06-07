all:	paper supmat
paper:
	latexmk -pdf -shell-escape main
supmat:
	python generate_qualitative_figures.py
	latexmk -pdf -shell-escape supplementary_main
clean:
	latexmk -c
