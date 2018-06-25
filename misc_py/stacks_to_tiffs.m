%Save focal series in subdirectories under this directory
outDirTop ='X:/Jeffrey-Ede/focal-series';
if outDirTop(end) ~= '/' 
    outDirTop = strcat(outDirTop, '/');
end

%Raw focal series
files = dir('X:/Jeffrey-Ede/focal-series-**/**.dm*');

L = numel(files);
for i = 1:L
    
    disp( strcat("Stack ", num2str(i), " of ", num2str(L), "...") );
    
    name = strcat(files(i).folder, '/', files(i).name);
    
    try
        %Get stack
        evalc( '[tags, imgs] = dmread(name)' );   

        %Normalize entire stack between 0.0 and 1.0
        min = min(min(min(imgs)));
        max = max(max(max(imgs)));
        
        imgs = (imgs-min) / (max-min);
        
        stack_depth = size(img, 3);
        for j = 1:stack_depth
            img = single(img(:,:,(j-1):j));

            %Save data to TIF
            name = strcat([outDirTop, 'stack', num2str(i), ...
                '/img', num2str(j), '.tif']);

            t = Tiff(name, 'w'); 
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