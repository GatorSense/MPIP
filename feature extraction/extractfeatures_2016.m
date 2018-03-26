% Feature extraction program
% Authors:
%   Anand Seethepalli
%   Graduate Student,
%   Computer Engineering
%   University of Missouri, Columbia
%   Email: aseethepalli@noble.org
%          anand_seethepalli@yahoo.co.in
% 
%   Dr. Alina Zare
%   Associate Professor,
%   Electrical and Computer Engineering,
%   University of Florida, Gainesville
%   Email: azare@ufl.edu

clear; close all; clc;
format long

basefolder = 'D:/soybeanroots';
read_folder = basefolder;
filePattern = fullfile(read_folder, '*seg.png');
jpegFiles = dir(filePattern);

featurenames = {'Median no. of roots', 'Max. no. of roots', ...
                'Total root length', 'Depth', 'Max. width', ...
                'Width-to-depth ratio', 'Network area', 'Convex area', ...
                'Solidity', 'Perimeter', 'Average radius', 'Volume', ...
                'Surface area', 'Maximum radius', 'Lower Root Area', ...
                'Fine Radius Freq.', 'Medium Radius Freq.', ...
                'Coarse Radius Freq.', 'Fineness Index', ...
                'Shallow Angle Freq.', 'Medium Angle Freq.', ...
                'Steep Angle Freq.', 'Shallowness Index', ...
                'Holes', 'Computation'};

nfeatures = numel(featurenames);
filecnt = length(jpegFiles);

fp = fopen('features.txt', 'w');
fprintf(fp, 'File name,');
for k = 1 : nfeatures
    if k ~= nfeatures
        fprintf(fp, '%s,', featurenames{k});
    else
        fprintf(fp, '%s\n', featurenames{k});
    end
end

for k = 1:filecnt
    FileName = jpegFiles(k).name;
    binFileName = fullfile(read_folder, FileName);
    fprintf(1, 'Now reading (%g of %g) %s - %f\n', k, filecnt, binFileName, toc(start));
    fflush(1);
    pstart = tic;
    
    bin = im2double(imread(binFileName));
    bin = bin(:, :, 1);
    
    if sum(sum(bin)) == 0
        featureset = nan(1, nfeatures);
    else
        [MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist, holes]=feature_ext(bin);
        featureset = double([MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist(3:end-1), holes, 0]);
        featureset(nfeatures) = toc(pstart);
    end
    
    fprintf(fp, '%s,', FileName);
    for j = 1 : nfeatures
        if j ~= nfeatures
            fprintf(fp, '%g,', featureset(j));
        else
            fprintf(fp, '%g\n', featureset(j));
        end
    end
end

fclose(fp);



