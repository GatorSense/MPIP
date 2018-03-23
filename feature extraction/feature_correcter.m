% clear; close all; clc;
format long

pkg load image

basefolder = 'E:/work/Mizzou soybean 2017/Rollins Bottoms/';
pixel_length = 300.5 / 3840;
pixel_area = pixel_length * pixel_length;
pixel_volume = pixel_length * pixel_area;

fp = fopen('features.txt', 'r');
%fw = fopen('features_corrected.txt', 'w');
str = strrep(fgetl(fp), ',', ' ');

featurenames = {'Valid', 'Median no. of roots', 'Max. no. of roots', ...
                'Total root length', 'Depth', 'Max. width', ...
                'Width-to-depth ratio', 'Network area', 'Convex area', ...
                'Solidity', 'Perimeter', 'Average radius', 'Volume', ...
                'Surface area', 'Maximum radius', 'Lower Root Area', ...
                'Shallow Angle Freq.', 'Medium Angle Freq.', ...
                'Steep Angle Freq.', 'Shallowness Index', ...
                'Fine Radius Freq.', 'Medium Radius Freq.', ...
                'Coarse Radius Freq.', 'Fineness Index', ...
                'Holes', 'Computation'};

nfeatures = 26;
%fprintf(fw, 'File name,');
%for k = 1 : nfeatures
%    if k ~= nfeatures
%        fprintf(fw, '%s,', featurenames{k});
%    else
%        fprintf(fw, '%s\n', featurenames{k});
%    end
%end

while ~feof(fp)
    str = strrep(fgetl(fp), ',', ' ');
    [fn1, fn2, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, ...
     f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, cnt, msg] = ...
     sscanf(str, '%s%s%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g%g', "C");
    
    if numel(fn2) <= 2
        cnt = cnt + 1;
        f33 = f32; f32 = f31; f31 = f30; f30 = f29; f29 = f28; f28 = f27; f27 = f26; f26 = f25; f25 = f24; f24 = f23;
        f23 = f22; f22 = f21; f21 = f20; f20 = f19; f19 = f18; f18 = f17; f17 = f16; f16 = f15; f15 = f14; f14 = f13;
        f13 = f12; f12 = f11; f11 = f10; f10 = f9; f9 = f8; f8 = f7; f7 = f6; f6 = f5; f5 = f4; f4 = f3; f3 = f2; f2 = f1;
        f1 = str2double(fn2);
        filename = fullfile(basefolder, fn1);
        fn1(end-7:end-4)=[];
        filename2 = fn1;
    else
        filename = fullfile(basefolder, [fn1, ',', fn2]);
        fn2(end-7:end-4)=[];
        filename2 = [fn1, ',', fn2];
    end
    
    if numel(filename2) ~= 14
        fprintf(1, '%s\n', filename2);
    end
%    img = im2double(imread(filename));
%    img = img(:, :, 1);
%    
%    if filename(end-8) == '5' && sum(img(:, 1)) ~= 0
%        valid = 0;
%    else
%        valid = 1;
%    end
%    
%    % Conversion of features to physical units.
%    f3 = f3 * pixel_length;
%    f4 = f4 * pixel_length;
%    f5 = f5 * pixel_length;
%    f7 = f7 * pixel_area;
%    f8 = f8 * pixel_area;
%    f10 = f10 * pixel_length;
%    f11 = f11 * pixel_length;
%    f12 = f12 * pixel_volume;
%    f13 = f13 * pixel_area;
%    f14 = f14 * pixel_length;
%    f15 = f15 * pixel_area;
%    
%    featureset = [valid, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, ...
%        f28 + f29, f27 + f30, f26 + f31, ((f28 + f29)/(f26 + f31)), ...
%        f16 + f17 + f18, f19 + f20 + f21 + f22, f23 + f24 + f25, ((f16 + f17 + f18)/(f23 + f24 + f25)), ...
%        f32, f33];
%    
%    fprintf(fw, '%s,', filename2);
%    for j = 1 : nfeatures
%        if j ~= nfeatures
%            fprintf(fw, '%g,', featureset(j));
%        else
%            fprintf(fw, '%g\n', featureset(j));
%        end
%    end
end

fclose(fp);
%fclose(fw);
