function [] = evolutionary_align(lb, ub)
    top_dir = 'X:/Jeffrey-Ede/focal-series/';
    %top_dir = '//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/series72/';
    save_loc = strcat(top_dir, 'transforms/transform');

    files = dir(top_dir);
    flags = [files.isdir] & ~strcmp({files.name},'.') & ~strcmp({files.name},'..') ...
        & ~strcmp({files.name},'transforms');
    stack_locs = files(flags);

    L = numel(stack_locs);
    filenames = [];
    numbers = [];
    for j=1:L
        filenames = [filenames, {strcat(stack_locs(j).folder, '\', ...
            stack_locs(j).name, '\')}];
        numbers = [numbers, num_in_str(stack_locs(j).name)];
    end
    [~, numbers_order] = sort(numbers);
    stack_locs = filenames(numbers_order);

    l_bound = 1+int32(lb*(L-1));
    u_bound = 1+int32(ub*(L-1));
    for i=l_bound:u_bound
        disp( strcat("Stack ", num2str(i), " of ", num2str(L), "...") );

        transforms = [];

        %Get series
        name_format = strcat([char(stack_locs(i)), 'img*.tif']);
        series_files = dir(name_format);
        L_series = numel(series_files);

        filenames = [];
        numbers = [];
        for j=1:L_series
            filenames = [filenames, {strcat(series_files(j).folder, '\', series_files(j).name)}];
            numbers = [numbers, num_in_str(series_files(j).name)];
        end
        [~, numbers_order] = sort(numbers);
        filenames = filenames(numbers_order);

        for j=2:L_series
            name1 = filenames(j-1);
            name2 = filenames(j);
            img1 = imread(char(name1(1)));
            img2 = imread(char(name2(1)));
            img1 = img1(:,1:3800);
            img2 = img2(:,1:3800);

            disp( strcat("Transform ", num2str(j-1), " of ", num2str(L_series-1), "...") );

            [optimizer, metric] = imregconfig('multimodal');
            optimizer.MaximumIterations = 1000;
            metric.UseAllPixels = 0;
            metric.NumberOfSpatialSamples = 250000;

            tform = imregtform(img2, img1, 'affine', optimizer, metric);

            %Save results
    %         registered = imwarp(img2,tform,'OutputView',imref2d(size(img1)));
    %         figure
    %         imshowpair(img1, registered, 'Scaling', 'joint')

            transforms = [transforms, tform];
        end

        save(strcat(save_loc, num2str(i), '.mat'), 'transforms');
    end
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