function savepdf( filename )
% Save and crop pdf using savefig and pdfcrop (uses Piotr's toolbox).
set(gcf,'renderer','painters'); set(gca,'OuterPosition',[0 0 1 1]);
[o,~]=system('pdfcrop'); tex=':/Library/TeX/texbin/:/usr/local/bin/';
if(o==127), setenv('PATH',[getenv('PATH') tex]); end
savefig(filename,1,'pdf','-r300','-fonts');
system(sprintf('pdfcrop %s.pdf %s.pdf',filename,filename));
end
