%Save focal series in subdirectories under this directory
outDirTop ='X:/Jeffrey-Ede/focal-series';
if outDirTop(end) ~= '/' 
    outDirTop = strcat(outDirTop, '/');
end

%Raw focal series
files = dir('X:/Jeffrey-Ede/focal-series*/*.dm*');

L = numel(files);
for i = 1:L
    
    disp( strcat("Stack ", num2str(i), " of ", num2str(L), "...") );
    
    name = strcat([files(i).folder, '/', files(i).name]);
    
    %Get stack
    [tags, imgs] = dmread(name);   

    %Normalize entire stack between 0.0 and 1.0
    minimum = min(min(min(imgs)));
    maximum = max(max(max(imgs)));

    imgs = (imgs-minimum) / (maximum-minimum);
            
    name_start = strcat([outDirTop, 'series', num2str(i)]);
    mkdir(name_start);
        
    try       
        stack_depth = size(imgs, 3);
        for j = 1:stack_depth
            img = single(imgs(:,:,j));

            %Save data to TIF
            img_name = strcat([name_start, '/img', num2str(j), '.tif']);

            t = Tiff(img_name, 'w'); 
            tagstruct.ImageLength = size(img, 1); 
            tagstruct.ImageWidth = size(img, 2); 
            tagstruct.Compression = Tiff.Compression.None; 
            tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 32; 
            tagstruct.SamplesPerPixel = 1; 
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
            t.setTag(tagstruct); 
            t.write(img); 
            t.close();
        end
    catch
        warning(num2str(i));
    end
end