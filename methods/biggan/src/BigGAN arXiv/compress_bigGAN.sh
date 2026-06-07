rm -r images2;
cp -r images images2;
cd images2;
for folder in 1506681_GSR2_1 1519233_R1GP5 1535537 1576139_dropout0.2;
do
echo $folder;
cd $folder;
for i in `ls *.png`; do mogrify -format jpg -quality 65 $i; done; rm *.png;
cd ..;
done


for folder in neighbors;
do
echo $folder;
cd $folder;
for i in `ls *.png`; do mogrify -format jpg -quality 80 $i; done; rm *.png;
cd ..;
done;

for folder in trunc_figure2;
do
echo $folder;
cd $folder;
for i in `ls *.png`; do mogrify -format jpg -quality 90 $i; done; rm *.png;
cd ..;
done;

for folder in interps0;
do
echo $folder;
cd $folder;
for i in `ls *.png`; do mogrify -format jpg -quality 85 $i; done; rm *.png;
cd ..;
done;

for folder in samples0 samples1;
do
echo $folder;
cd $folder;
for i in `ls *.png`; do mogrify -format jpg -quality 95 $i; done; rm *.png;
cd ..;
done;



for i in `ls *.png`; do mogrify -format jpg -quality 90 $i; done; rm *.png
cd ..;
