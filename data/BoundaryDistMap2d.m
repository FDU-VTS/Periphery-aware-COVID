%img_dir = 'E://DATA//visualization//mask_twotb//';
%save_dir = 'E://DATA//visualization//mask_twotb_dist//';

%layer=[142,130,160,140];
imgs = dir([img_dir '*.png']);
disp(imgs);

for i = 1: length(imgs)
    name = imgs(i).name;
    img = importdata([img_dir name]);
    %img = dlmread([img_dir name]);
  
    %img = permute(img, [2,3,1]);
    %figure
    %imshow(img * 255, []);
    %break;
    img = 1 - img;
    [dists, idx] = bwdist(img, 'euclidean');
    disp(name);
    disp([size(img) size(dists)]);
    %save([save_dir name], 'dists');
    figure
    imagesc(dists)
    colorbar;
    
    %disp(img(200,100,100));
    %disp(size(dists));
    %disp(dists(200,100,100));
    break;
end