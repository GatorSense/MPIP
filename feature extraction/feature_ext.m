function [MedR,MaxR,TLength,depth,maxw,wdr,NA,ConvA,solidity,Perim,AvRadius,Volume,SA,maxr,LowerRootArea, rootszhist, orihist, holes]=feature_ext(img2, img2_skel, DT)
[ir, ic] = find(img2 == 1);
img2 = single(img2(min(ir) : max(ir), min(ic) : max(ic)));
%img2_skel=bwmorph(img2,'skel',Inf); % skeletonize image
img2_skel = img2_skel(min(ir) : max(ir), min(ic) : max(ic));
img2_perimeter = bwperim(img2);  % perimeter pixels
% BW2=img2_skel+img2_perimeter;
m=size(img2);

%% calculate median number of roots (MedR) and maximum number of roots (MaxR)

i1=[img2,zeros(m(1),1)];
i2=[zeros(m(1),1),img2];
i3=i1-i2;
i3(i3==-1)=0;
i3 = sum(i3, 2);
%  for u=1:m(1)
%     f=find(i3(u,:)==1);
%     nf=size(f);
%     nr(u,1)=nf(2);
%  end
ii4 = [];
for ppp = numel(i3) : -1 : 1
    ii4(numel(i3) - ppp + 1) = i3(ppp);
end

 MedR=median(i3);
 MaxR=max(i3);
 clear i3;
%  Bushiness=MaxR/MedR;

 %% calculate total root length(TLength),depth and max width of root(maxw) and Width-to-depth ratio (wdr)
 
%  depth=0;  % 
%  for v1=1:m(1)
%      row=find(img2(v1,:)==1);
%      nrow=size(row);
%      if nrow(2)>0
%          depth=depth+1; 
%      end
%  end
depth = max(ir) - min(ir) + 1;
TLength = sum(sum(img2_skel));
maxw = max(ic) - min(ic) + 1;
clear ic;
% maxw=0;
% for v2=1:m(2)
%     col=find(img2(:,v2)==1);
%     ncol=size(col);
%     if ncol(1)>0
%         maxw=maxw+1;
%     end
% end
 
 wdr=maxw/depth;
 
 %% calculate Network area and Convex area (ConvA) and solidity(The fraction equal to the network area divided by the convex area.)

[row_root,~] = find(img2==1);
% figure;scatter(col_root,row_root);
% root_pixel = [row_root,col_root];
NA=length(ir);
clear ir;
% DT = delaunayTriangulation(row_root,col_root);
% [~, ConvA] = convexHull(DT);
%DT = bwconvhull(img2);
ConvA = sum(sum(DT));
clear DT;
solidity=NA/ConvA;
Perim = sum(sum(img2_perimeter));
clear img2_perimeter;
% re_img2=zeros(m(1)+2,m(2)+2);   %reshape the img2 by adding 0 outside the orignal img2
% re_img2(2:m(1)+1,2:m(2)+1)= img2;
% [row_root_re,col_root_re] = find(re_img2==1);
% Perim=0;    %Perimeter (Perim)
% for j1=1: length(row_root_re)
%     if re_img2(row_root_re(j1)-1,col_root_re(j1)-1)+re_img2(row_root_re(j1)-1,col_root_re(j1))+re_img2(row_root_re(j1),col_root_re(j1)-1)+re_img2(row_root_re(j1)+1,col_root_re(j1)+1)+re_img2(row_root_re(j1),col_root_re(j1)+1)+re_img2(row_root_re(j1)+1,col_root_re(j1))+re_img2(row_root_re(j1)-1,col_root_re(j1)+1)+re_img2(row_root_re(j1)+1,col_root_re(j1)-1)<8
%         Perim=Perim+1;
%     end
% end


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
%  [row_root,col_root] = find(img2==1); % total pixel
 pixel_below=find(row_root > maxr_row(1));
 LowerRootArea=length(pixel_below);
 [r_row,r_col] = find(r>0);
 AvRadius = sum(sum(r)) ./ TLength;
 Volume = sum(sum(pi .* (r .* r)));
 SA = sum(sum(2 .* pi .* r));
 
 eh = [0, 5.4, 10.8, 16.2, 21.6, 27, 32.4, 37.8, 43.2, 48.6, 54, 59.4];
 Nh = histcounts_oct_opt(r, eh);
 Nh = Nh(2 : end);
 rootszhist = Nh ./ sqrt(Nh * (Nh'));
 
%  ori = @(x) orifunc(x, r_row, r_col);
%  rori = blockproc(r, [1, 1], ori);
 wsize = 20;
 nz = numel(r_row);
 oris = zeros(nz, 1);
 adds = [wsize * (-1), wsize * (-1), wsize, wsize];
 rhigh = m(1);
 chigh = m(2);
 
 for k = 1 : nz
    %fprintf(1, 'k=%g\n', k);
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
        % v = v1 ./ sqrt(v1' * v1);
        % [v, D] = eigs(co);
        if ~isnan(v(2))
            oris(k) = pi + atan(v(2));
        else
            oris(k) = pi;
        end
    end
 end
 
 er = 0 : 9;
 er = er .* (pi / 5);
%  er = [-1, er];
 Nr = histcounts_oct_opt(oris, er);
%  Nr = Nr(2 : end);
 orihist = Nr ./ sqrt(Nr * (Nr'));
 
img2 = 1 - img2;
CC = bwconncomp(img2, 8);
holes = numel(CC.PixelIdxList);
% r_matrix=[];
% for rno=1:length(r_row)
%     r_matrix(rno,1)=r_row(rno,1);
%     r_matrix(rno,2)=r_col(rno,1);
%     r_matrix(rno,3)=r(r_row(rno,1),r_col(rno,1));
% end
% % x=1:length(r_row);
% % y=r_matrix(:,3);
% % figure;plot(x,y);
end

% function [v] = orifunc(bs, r, c)
%     loc = bs.location;
%     isize = bs.imageSize;
%     wsize = 20;
%     
%     % [rmin, cmin, rmax, cmax]
%     adds = [wsize * (-1), wsize * (-1), wsize, wsize];
%     limits = [loc(1), loc(2), loc(1), loc(2)];
%     limits = limits + adds;
%     
%     limits(limits < 1) = 1;
%     
%     if limits(3) > isize(1)
%         limits(3) = isize(1);
%     end
%     
%     if limits(4) > isize(2)
%         limits(4) = isize(2);
%     end
%     
%     plist = (r > limits(1) & r < limits(3) & c > limits(2) & c < limits(4));
%     
%     rvals = r(plist);
%     cvals = c(plist);
%     
%     if isempty(rvals) || isempty(cvals)
%         v = -1;
%     elseif numel(rvals) == 1 || numel(cvals) == 1
%         v = -1;
%     else
%         co = cov([cvals, rvals]);
%         [v, ~] = eigs(co);
%         v = pi + atan2(v(2), v(1));
%     end
% end

function [histcnt] = histcounts_oct(rmat, edges)
    [r, c] = size(rmat);
    nelements = r * c;
    nedges = numel(edges) - 1;
    histcnt = zeros(1, nedges);
    
    for i = 1 : nelements
        if rmat(i) > edges(end) || rmat(i) < edges(1)
            continue;
        end
        
        hfilt = sum(edges(1 : end-1) <= rmat(i));
        histcnt(hfilt) = histcnt(hfilt) + 1;
    end
end

function [histcnt] = histcounts_oct_opt(rmat, edges)
    [r, c] = size(rmat);
    nelements = r * c;
    nedges = numel(edges) - 1;
    histcnt = zeros(1, nedges);
    
    for i = 1 : nedges
        histcnt(i) = sum(sum(rmat >= edges(i) & rmat < edges(i + 1)));
    end
end
