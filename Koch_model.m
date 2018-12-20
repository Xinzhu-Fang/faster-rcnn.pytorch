package_path = 'C:\Program Files\MATLAB\R2018a\toolbox\signatureSal';
addpath(package_path)
% salMap = signatureSal('C:\Users\14868\Documents\GitHub\NN\project\images\gratings\0029.png');
image_path = 'C:\Users\14868\Documents\GitHub\NN\project\images\Capture2.png';
salMap = signatureSal(image_path);
imagesc(salMap)
% a = imread(image_path)