TARGET = iclr2019_conference

all: 
	pdflatex $(TARGET)
	bibtex $(TARGET)
	pdflatex $(TARGET)
	pdflatex $(TARGET)
	rm -rf $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).blg

