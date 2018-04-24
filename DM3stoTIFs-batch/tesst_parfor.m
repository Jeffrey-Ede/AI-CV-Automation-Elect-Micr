load('//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/compendium1')
load('//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/files')
a = compendium

reaping = 1;
compendium = [];
L = numel(files);
x=0;
for i = 1:L
    if files(i).bytes/1024 < 50000
       x = x+1;
    end
    if x == 8023
        x=i;
        break
    end
end
disp(x)