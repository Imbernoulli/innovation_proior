clear all; close all; clc;

figure_id = 0;

save_folder = './effective_atrous';
mkdir(save_folder);

kernel_height = 3;
kernel_width = 3;
addpath('~/workspace/export_fig')

fontSize = 20;
lineWidth = 2;

%cmap = colormap('jet');
cols = [...    
    173 90 90; ...    
    165 165 82; ...
    92 173 173; ...
    128 128 192; ...
    ] / 255;

height = 65;
width = 65;

range = 1:65;

all_counts = zeros(length(range), kernel_height*kernel_width);

for atrous_rate = range

    if atrous_rate == 33;
        a= 1;
    end
    
kernel = cell(kernel_height, kernel_width);
for r = 1:kernel_height
    for c = 1:kernel_width
        kernel{r, c} = {-atrous_rate + (c-1) * atrous_rate, -atrous_rate + (r-1)*atrous_rate};
    end
end

output = zeros(height, width);

for r = 1:height
    for c = 1:width
        effective_weight = 0;
        for k_r = 1:kernel_height
            for k_c = 1:kernel_width
                if (r + kernel{k_r, k_c}{1} > 0 && r + kernel{k_r, k_c}{1} <= height && ...
                        c + kernel{k_r, k_c}{2} > 0 && c + kernel{k_r, k_c}{2} <= width)
                    effective_weight = effective_weight + 1;
                end
            end
        end
        output(r, c) = effective_weight;
    end
end

counts = hist(output(:), [1:kernel_height*kernel_width]);
all_counts(atrous_rate, :) = counts;

end

normalized_counts = all_counts / (height*width);

figure(1)

rates = 1:3:65;

plot(rates, normalized_counts(rates, 1), '-r^', 'LineWidth', lineWidth, 'color', cols(1, :))
hold on
%plot(normalized_counts(:, 2), '-rx', 'LineWidth', lineWidth, 'color', cmap(17, :))
plot(rates, normalized_counts(rates, 4), '-ro', 'LineWidth', lineWidth, 'color', cols(2, :))
%plot(normalized_counts(:, 6), '-rd', 'LineWidth', lineWidth, 'color', cmap(49, :))
plot(rates, normalized_counts(rates, 9), '-rs', 'LineWidth', lineWidth, 'color', cols(3, :))

ylabel('Normalized count', 'FontSize', fontSize);
xlabel('atrous rate', 'FontSize', fontSize);
%leg = legend('1 valid weight', '2 valid weights', '4 valid weights', '6 valid weights', '9 valid weights');
leg = legend('1 valid weight', '4 valid weights', '9 valid weights');

set(leg, 'FontSize', fontSize, 'Location', 'SouthEast')
set(gca, 'XGrid', 'on')
set(gca, 'YGrid', 'on');
set(gca,'FontSize', fontSize);
set(leg, 'FontSize', fontSize);
set(gcf, 'Color', [1 1 1]);

fn = fullfile(save_folder, sprintf('atrous_rate_vs_valid_weights.pdf', atrous_rate, height));
export_fig(fn)
