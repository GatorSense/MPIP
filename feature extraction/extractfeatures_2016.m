% clear; close all; clc;
format long

pkg load image
warning('off', 'Octave:divide-by-zero')

% read_folder = 'E:\lexar\soyabean_roots\2015\work_day_45\';
% write_folder = 'E:\lexar\soyabean_roots\2015\work_day_45_seg_em\';
task = 'segcc';
% task = 'test';
basefolder = 'E:/work/Mizzou soybean 2017/Rollins Bottoms/';
% basefolder = 'C:\Users\asn5d\Desktop\mu_latex_thesis_template\thesispresentation\Figures\';

read_folder = basefolder; %[basefolder, task, '\'];
% write_folder = [basefolder, task, '2\'];
% featuretext = ['C:\Users\asn5d\Dropbox\bin_feat', task, '_features.txt'];
% feat = fopen(featuretext, 'w');

% read_folder = 'E:\work_day_45\';
% write_folder = 'E:\work_day_45_seg_em\';
% mkdir(write_folder);

filePattern = fullfile(read_folder, '*seg.png');
thinPattern = fullfile(read_folder, '*thin.png');
convPattern = fullfile(read_folder, '*convhull.png');

jpegFiles = dir(filePattern);
thinFiles = dir(thinPattern);
convFiles = dir(convPattern);
start = tic;
features = cell(5, 1);

UndistortImages = 0;

featurenames = {'Median no. of roots', 'Max. no. of roots', ...
                'Total root length', 'Depth', 'Max. width', ...
                'Width-to-depth ratio', 'Network area', 'Convex area', ...
                'Solidity', 'Perimeter', 'Average radius', 'Volume', ...
                'Surface area', 'Maximum radius', 'Lower Root Area', ...
                'radhist1', 'radhist2', 'radhist3', 'radhist4', 'radhist5', ...
                'radhist6', 'radhist7', 'radhist8', 'radhist9', 'radhist10', ...
                'orihist1', 'orihist2', 'orihist3', 'orihist4', 'orihist5', ...
                'orihist6', 'holes', ... %'radpca1', 'radpca2', 'radpca3', ...
                %'oripca1', 'oripca2', 'oripca3', ...
                'Computation'};

% nfeatures = numel(featurenames);
nfeatures = 33;
% Files = cell(length(jpegFiles), 1);

features{4, 1} = nan(length(jpegFiles), nfeatures);
% feats = nan(length(jpegFiles), nfeatures);
BadFiles = cell(length(jpegFiles), 1);
filecnt = length(jpegFiles);
nfails = 0;

prevtimes = cell(length(jpegFiles), 1);
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
    thinName = thinFiles(k).name;
    convName = convFiles(k).name;
    
    binFileName = fullfile(read_folder, FileName);
    thinFileName = fullfile(read_folder, thinName);
    convFileName = fullfile(read_folder, convName);
    fprintf(1, 'Now reading (%g of %g) %s - %f\n', k, filecnt, binFileName, toc(start));
    fflush(1);
%     ptask = getCurrentTask;
    pstart = tic;
    
    cno = str2double(FileName(end - 16));
    
    if cno ~= 1
        continue;
        bin = im2double(imread(binFileName));
        bin = bin(:, :, 1);
        thin = im2double(imread(thinFileName));
        thin = thin(:, :, 1);
        conv = im2double(imread(convFileName));
        conv = conv(:, :, 1);
        
        if sum(sum(bin)) == 0
            featureset = nan(1, 33);
        else
            [MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist, holes]=feature_ext(bin, thin, conv);
            featureset = double([MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist(3:end-1), holes, 0]);
            featureset(33) = toc(pstart);
        end
        
        fprintf(fp, '%s,', FileName);
        for j = 1 : nfeatures
            if j ~= nfeatures
                fprintf(fp, '%g,', featureset(j));
            else
                fprintf(fp, '%g\n', featureset(j));
            end
        end
        
        %feats(k, :) = featureset;
        %Files{k, 1} = FileName; %#ok<*AGROW>
    else
        bin = im2double(imread(binFileName));
        bin = bin(:, :, 1);
        thin = im2double(imread(thinFileName));
        thin = thin(:, :, 1);
        conv = im2double(imread(convFileName));
        conv = conv(:, :, 1);
        
        if sum(sum(bin)) == 0
            featureset = nan(1, 33);
        else
            [MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist, holes]=feature_ext(bin, thin, conv);
            featureset = double([MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist(3:end-1), holes, 0]);
            featureset(33) = toc(pstart);
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
end

fclose(fp);



