%Harvest images from the microscopy data servers to train neural networks
%with

maxNoisetoSignalRatio = 0.02;

topDir = '\\flexo.ads.warwick.ac.uk\Shared39\EOL2100\2100\Users\';
if topDir(end) ~= '/' 
    topDir = strcat(topDir, '/');
end

outDirTop ='//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/2100';
if outDirTop(end) ~= '/' 
    outDirTop = strcat(outDirTop, '/');
end

statSavePeriod = 200; %Save stats every _ images
statsDir = '//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets';
if statsDir(end) ~= '/' 
    statsDir = strcat(statsDir, '/');
end

filesDir = '//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets';
if filesDir(end) ~= '/' 
    filesDir = strcat(filesDir, '/');
end

fprintf("Finding files...\n");

files = dir( strcat(topDir, '**/*.dm4') )

%Save file source locations
save(strcat(filesDir, 'files.mat'), 'files');

fprintf("Files saved...\n");

%%Harvest files
reaping = 1;
compendium = [];
L = numel(files);
for i = 1:L
    
    disp( strcat("Image ", num2str(i), " of ", num2str(L), "...") );
    
    if files(i).bytes/1024 < 50000
    
        try
            name = strcat(files(i).folder,'/',files(i).name);
            evalc( '[tags, img] = dmread(name)' );   

            if tags.InImageMode.Value == 1 && numel(size(img)) == 2
                
                %Get image stats and a cropped then resized 2048x2048 image 
                [stats, img2048] = img_params(img);
                
                %Store statistics
                compendium = [compendium, stats];
                
                disp(num2str(stats.ratio_of_meanNoise_to_mean));
                if stats.ratio_of_meanNoise_to_mean >=0 && stats.ratio_of_meanNoise_to_mean < maxNoisetoSignalRatio
                    
                    %Save data to TIF
                    name = strcat('reaping', num2str(reaping));
                    
                    t = Tiff(strcat(outDirTop, name, '.tif'), 'w'); 
                    tagstruct.ImageLength = size(img2048, 1); 
                    tagstruct.ImageWidth = size(img2048, 2); 
                    tagstruct.Compression = Tiff.Compression.None; 
                    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
                    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
                    tagstruct.BitsPerSample = 32; 
                    tagstruct.SamplesPerPixel = 1; 
                    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
                    t.setTag(tagstruct); 
                    t.write(img2048); 
                    t.close();
                
                    reaping = reaping+1;
                end
            end
        catch
            warning(num2str(i));
        end
    end
    
    %Leapfrog save the compendium every 100 images in case something goes wrong
    if mod(i, statSavePeriod) == 0
        save(strcat(statsDir, 'compendium1.mat'), 'compendium');
    end
    if mod(i, statSavePeriod)-idivide(int32(statSavePeriod), int32(2)) == 0
        save(strcat(statsDir, 'compendium2.mat'), 'compendium');
    end
end

%Save final statistics
save(strcat(statsDir, 'compendium.mat'), 'compendium');

disp('Finished!');