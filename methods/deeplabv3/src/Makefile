fnname1=top
fnname2=sup

all: camera-ready

tags:
	etags *.tex *.bib

camera-ready: *.tex *.bib
	pdflatex ${fnname1}; bibtex ${fnname1}; pdflatex ${fnname1}; pdflatex ${fnname1}; pdflatex ${fnname1};

clean:
	rm -f ${fnname1}.{out,aux,brf,blg,log,rip,toc,ps,tex.flc,dvi} missfont.log *~
	rm -f ${fnname2}.{out,aux,brf,blg,log,rip,toc,ps,tex.flc,dvi} missfont.log *~
