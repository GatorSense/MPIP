function [MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist, holes]=feature_ext(img2)
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

[ir, ic] = find(img2 == 1);
img2 = single(img2(min(ir) : max(ir), min(ic) : max(ic)));
img2_skel=bwmorph(img2,'skel',Inf); % skeletonize image
img2_perimeter = bwperim(img2);  % perimeter pixels
m=size(img2);

%% calculate median number of roots (MedR) and maximum number of roots (MaxR)

i1=[img2,zeros(m(1),1)];
i2=[zeros(m(1),1),img2];
i3=i1-i2;
i3(i3==-1)=0;
i3 = sum(i3, 2);
ii4 = [];
for ppp = numel(i3) : -1 : 1
    ii4(numel(i3) - ppp + 1) = i3(ppp);
end

 MedR=median(i3);
 MaxR=max(i3);
 clear i3;

 %% calculate total root length(TLength),depth and max width of root(maxw) and Width-to-depth ratio (wdr)
 
depth = max(ir) - min(ir) + 1;
TLength = sum(sum(img2_skel));
maxw = max(ic) - min(ic) + 1;
clear ic;
 
 wdr=maxw/depth;
 
 %% calculate Network area and Convex area (ConvA) and solidity(The fraction equal to the network area divided by the convex area.)

[row_root,~] = find(img2==1);
NA=length(ir);
clear ir;
DT = bwconvhull(img2);
ConvA = sum(sum(DT));
clear DT;
solidity=NA/ConvA;
Perim = sum(sum(img2_perimeter));
clear img2_perimeter;

%% calculate the radius, volume and surface area

%make a flip of background points and foreground points
fimg2=ones(m(1),m(2));
fimg2=fimg2-img2;

% calculate radius 
 r = bwdist(fimg2) .* img2_skel;
 clear img2_skel;
 clear fimg2;
 maxr = max (max(r));  % maximum radius
 [maxr_row,~] = find(r==maxr);
 pixel_below=find(row_root > maxr_row(1));
 LowerRootArea=length(pixel_below);
 [r_row,r_col] = find(r>0);
 AvRadius = sum(sum(r)) ./ TLength;
 Volume = sum(sum(pi .* (r .* r)));
 SA = sum(sum(2 .* pi .* r));
 
 eh = [0, 5.4, 10.8, 16.2, 21.6, 27, 32.4, 37.8, 43.2, 48.6, 54, 59.4];
 Nh = histcounts(r, eh);
 Nh = Nh(2 : end);
 rootszhistogram = Nh ./ sqrt(Nh * (Nh'));
 rootszhist = [sum(rootszhistogram(1:3)), sum(rootszhistogram(4:7)), ...
               sum(rootszhistogram(8:10)), sum(rootszhistogram(1:3)) / sum(rootszhistogram(8:10))];
 
 wsize = 20;
 nz = numel(r_row);
 oris = zeros(nz, 1);
 adds = [wsize * (-1), wsize * (-1), wsize, wsize];
 rhigh = m(1);
 chigh = m(2);
 
 for k = 1 : nz
    limits = [r_row(k), r_col(k), r_row(k), r_col(k)];
    limits = limits + adds;
    
    limits(limits < 1) = 1;
    
    if limits(1) < 0
        limits(1) = 0;
    end
    
    if limits(2) < 0
        limits(2) = 0;
    end
    
    if limits(3) > rhigh
        limits(3) = rhigh;
    end
    
    if limits(4) > chigh
        limits(4) = chigh;
    end
    
    plist = (r_row > limits(1) & r_row < limits(3) & r_col > limits(2) & r_col < limits(4));
    
    rvals = r_row(plist);
    cvals = r_col(plist);
    mcnt = numel(rvals);
    
    if isempty(rvals) || isempty(cvals)
        oris(k) = -1;
    elseif numel(rvals) == 1 || numel(cvals) == 1
        oris(k) = -1;
    else
        xc = bsxfun(@minus,[cvals, rvals],sum([cvals, rvals],1)/mcnt);  % Remove mean
        co = (xc' * xc) ./ (mcnt - 1);
        
        rt1 = (co(1,1) + co(2,2) + sqrt((co(1,1) - co(2,2)) * (co(1,1) - co(2,2)) + 4 * co(1,2) * co(1,2))) / 2;
        rt2 = (co(1,1) + co(2,2) - sqrt((co(1,1) - co(2,2)) * (co(1,1) - co(2,2)) + 4 * co(1,2) * co(1,2))) / 2;
        
        if rt1 < rt2
            rt1 = rt2;
        end
        
        v = [1; -(co(1,1) - rt1) / co(1,2)];
        if ~isnan(v(2))
            oris(k) = pi + atan(v(2));
        else
            oris(k) = pi;
        end
    end
 end
 
 er = 0 : 9;
 er = er .* (pi / 5);
 Nr = histcounts(oris, er);
 orihistogram = Nr ./ sqrt(Nr * (Nr'));
 orihistogram = orihistogram(3:end-1);
 orihist = [sum(orihistogram([3, 4])), sum(orihistogram([2, 5])), ...
               sum(orihistogram([1, 6])), sum(orihistogram([3, 4])) / sum(orihistogram([1, 6]))];
 
img2 = 1 - img2;
CC = bwconncomp(img2, 8);
holes = numel(CC.PixelIdxList);
end
