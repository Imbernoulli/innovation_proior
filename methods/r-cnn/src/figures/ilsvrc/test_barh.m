function test_barh()

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 16);

addpath('export_fig');

methods = {'*R-CNN BB ', ...
           '*OverFeat (2) ', ...
           'UvA-Euvision ', ...
           '*NEC-MU ', ...
           '*OverFeat (1) ', ...
           'Toronto A ', ...
           'SYSU_Vision ', ...
           'GPU_UCLA ', ...
           'Delta ', ...
           'UIUC-IFP '};

mAPs = [
  31.4 
  24.3
  22.6
  20.9
  19.4
  11.5
  10.5
  9.8
  6.1
  1.0];

mAPs = mAPs(end:-1:1);
methods = methods(end:-1:1);

clf;
%set(gcf, 'Units', 'inches');
%set(gcf, 'Position', [12 0 3.65 2]);

barh(1:8, mAPs(1:8), 'FaceColor', [0 0 1]);
hold on;
barh(9:10, mAPs(9:10), 'FaceColor', [1 0 0]);
hold off;

set(gca, 'ytick', 1:10);
set(gca, 'yticklabel', methods);
%set(gca, 'DataAspectRatio', [1.1 0.4 1]);
xlabel('mean average precision (mAP) in %', 'FontSize', 18);

legend('competition result', 'post competition result', 'location', 'southeast');
title('ILSVRC2013 detection test set mAP', 'FontSize', 18);

xlim([0 100]);
ylim([0.5 10.5]);
grid on;
set(gca, 'ygrid', 'off');
for i = 1:length(mAPs)
  text(mAPs(i)+1, i, sprintf('%.1f%%', mAPs(i)));
end


%ylabel('wall time (min)', 'FontSize', 10);
%title('Data mining and convex optimization time', 'FontSize', 10);

set(gcf, 'Color', 'white');
export_fig ilsvrc13.pdf -pdf
%!open ilsvrc13.pdf
