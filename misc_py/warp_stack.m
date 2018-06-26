function [stack] = warp_stack(stack_loc, transforms_file)

%%Get stack filenamess
load(transforms_file);

%Get series
name_format = strcat([char(stack_loc), 'img*.tif']);
series_files = dir(name_format);
L = numel(series_files);

filenames = [];
numbers = [];
for j=1:L
    filenames = [filenames, {strcat(series_files(j).folder, '\', series_files(j).name)}];
    numbers = [numbers, num_in_str(series_files(j).name)];
end
[~, numbers_order] = sort(numbers);
filenames = filenames(numbers_order);

%%Get transforms relative to middle image
mid = int32(L/2) + 1;

%Accumulate transformations
left_trans = [];
left_trans = [left_trans, inv(transforms(mid-1).T)];
for i = (mid-2):1
    left_trans = [left_trans(1)\transforms(i).T, left_trans];
end
right_trans = [];
right_trans = [right_trans, transforms(mid).T];
for i = (mid+1):(L-1)
    right_trans = [right_trans, right_trans(end)*transforms(i).T];
end

%Apply transformations
extreme_limits = crop_limits(left_trans(1));
for i = 2:(mid-1)
    limits = crop_limits(left_trans(i));
    if limits(1) > extreme_limits(1)
        extreme_limits(1) = limits(1);
    end
    if limits(2) > extreme_limits(2)
        extreme_limits(2) = limits(2);
    end
    if limits(3) < extreme_limits(3)
        extreme_limits(3) = limits(3);
    end
    if limits(4) < extreme_limits(4)
        extreme_limits(4) = limits(4);
    end
end
for i = mid:(L-1)
    j = i-mid+1;
    limits = crop_limits(left_trans(j));
    if limits(1) > extreme_limits(1)
        extreme_limits(1) = limits(1);
    end
    if limits(2) > extreme_limits(2)
        extreme_limits(2) = limits(2);
    end
    if limits(3) < extreme_limits(3)
        extreme_limits(3) = limits(3);
    end
    if limits(4) < extreme_limits(4)
        extreme_limits(4) = limits(4);
    end
end

%%Transform images to align them with the middle image, then crop

%Left images
for i = 1:(mid-1)
    name = filenames(i);
    img = imread(char(name(1)));
    img = img(:,1:3800);
    
    warped = imwarp(img, left_trans(i), 'OutputView', imref2d(size(img)));
end

%Middle image
name = filenames(mid);
img = imread(char(name(1)));
img = img(:,1:3800);

warped = imwarp(img, left_trans(i), 'OutputView', imref2d(size(img)));

%Right images
for i = mid:(L-1)
    name = filenames(i);
    img = imread(char(name(1)));
    img = img(:,1:3800);
    
    warped = imwarp(img, left_trans(i), 'OutputView', imref2d(size(img)));
end



stack = NaN;
end

function Num=num_in_str(A)
B = regexp(A,'\d*','Match');
for ii= 1:length(B)
  if ~isempty(B{ii})
      Num(ii,1)=str2num(B{ii});
  else
      Num(ii,1)=NaN;
  end
end
end

function limits = crop_limits(T)
% width and length of the input image
width = size(img_fixed,1);
height = size(img_fixed,2);

% transform the four corners of the image to find crop area
[x1,y1] = transformPointsForward(T,0,0);
[x2,y2] = transformPointsForward(T,width,0);
[x3,y3] = transformPointsForward(T,width,height);
[x4,y4] = transformPointsForward(T,0,height);

% find inner most borders for a rectangular crop
if max([x1,x4]) < 0
    x_left = 0;      
else
    x_left = ceil(max([x1,x4]));
end

if min([x2,x3]) > width
    x_right = width;     
else
    x_right = floor(min([x2,x3]));
end

if max([y1,y2]) < 0
    y_top = 0;   
else
    y_top = ceil(max([y1,y2])); 
end

if min([y3,y4]) > height
    y_bottom = height;      
else
    y_bottom = floor(min([y3,y4]));
end

limits = [x_left y_top x_right-x_left y_bottom-y_top];

end

% for i=14:L
%     name1 = filenames(i-1);
%     name2 = filenames(i);
%     img1 = imread(char(name1(1)));
%     img2 = imread(char(name2(1)));
%     img1 = img1(:,1:3800);
%     img2 = img2(:,1:3800);
% 
%     disp( strcat("Transform ", num2str(i-1), " of ", num2str(L-1), "...") );
% 
%     size(transforms)
%     tform = transforms(i-1);
%     %Save results
%     registered = imwarp(img2,tform, 'OutputView', imref2d(size(img1)));
%     figure
%     imshowpair(img1, registered, 'Scaling', 'joint')
% end