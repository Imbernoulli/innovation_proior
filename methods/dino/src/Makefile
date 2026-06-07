filename=main

pdf:
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}

clean:
	rm -f ${filename}.{ps,pdf,log,aux,out,dvi,brf,bbl,blg} 

all: pdf
