FILE=value-distributions

all: $(FILE).pdf

clean:
	rm notation.aux
	rm $(FILE).{aux,bbl,blg,log,pdf,out}

touch:
	touch $(FILE).tex
	make

%.pdf: $(FILE).tex
	pdflatex $(FILE).tex
	bibtex $(FILE)
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex
