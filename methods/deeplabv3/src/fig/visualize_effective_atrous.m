clear all; close all; clc;

figure_id = 0;
cmap = colormap('jet');

save_folder = './effective_atrous';
mkdir(save_folder);

for atrous_rate = [10, 24, 40, 54]
figure_id = figure_id + 1;

kernel_height = 3;
kernel_width = 3;
addpath('~/workspace/export_fig')

kernel = cell(kernel_height, kernel_width);
for r = 1:kernel_height
    for c = 1:kernel_width
        kernel{r, c} = {-atrous_rate + (c-1) * atrous_rate, -atrous_rate + (r-1)*atrous_rate};
    end
end

height = 65;
width = 65;
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

scale_value = floor(64 / (kernel_height * kernel_width));
output = uint8(output * scale_value);

figure(figure_id)
imshow(output, cmap);
set(gcf, 'Color', [1, 1, 1]);

fn = fullfile(save_folder, sprintf('rate%d_size%d.pdf', atrous_rate, height));
export_fig(fn)

end
