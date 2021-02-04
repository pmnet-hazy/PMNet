addpath ./matlab
%if exist('use_gpu', 'var') && use_gpu  
  caffe.set_mode_gpu();  
  gpu_id = 0;  % we will use the first gpu 
  caffe.set_device(gpu_id);  
%else  
%  caffe.set_mode_cpu();  
%end  



net_weights = '/home/mj/caffe/models/VGG/vgg19.caffemodel';
% 
% net_model_fc6 = '/media/bruce/research/trial_22/config/fc6.prototxt';
net_model_fc7 = '/home/mj/caffe/models/VGG/fc7.prototxt';
% net_model_fc8 = '/media/bruce/research/trial_22/config/fc8.prototxt';
% net_model_fc9 = '/media/bruce/research/trial_22/config/fc9.prototxt';
% net_model_fc10 = '/media/bruce/research/trial_22/config/fc10.prototxt';
% net_model_fc11 = '/media/bruce/research/trial_22/config/fc11.prototxt';

%net_model = '/media/bruce/research/trial_17/config/deploy.prototxt';

phase = 'test';
if ~exist(net_weights, 'file')  
  error('net not found');  
end
% 
% net_6 = caffe.Net(net_model_fc6, net_weights, phase);
net_7 = caffe.Net(net_model_fc7, net_weights, phase);
% net_8 = caffe.Net(net_model_fc8, net_weights, phase);
% net_9 = caffe.Net(net_model_fc9, net_weights, phase);
% net_10 = caffe.Net(net_model_fc10, net_weights, phase);
% net_11 = caffe.Net(net_model_fc11, net_weights, phase);

%---------------------------------------------------------------------
% read the image_name file to get all the image paths:
fid = fopen('/home/mj/caffe/data/notgood/notgood.txt');
tline = fgets(fid);
file_count = 0;
files = {};
while ischar(tline)
    file_count = file_count + 1;
    disp(tline);
    files{file_count,1} = tline;
    tline = fgets(fid);
end
fclose(fid);
fprintf('Found %d files\n', file_count);
%----------------------------------------------------------------------

%----------------------------------------------------------------------
% fed all the test images into caffe model and get the output features
% result_matrx_fc6 = [];
result_matrx_fc7 = [];
% result_matrx_fc8 = [];
% result_matrx_fc9 = [];
% result_matrx_fc10 = [];
% result_matrx_fc11 = [];
for i = 1:file_count
    
    [folder, name, ~] = fileparts(files{i});
    tic
    im = imread(['/home/mj/caffe/data/notgood/', folder, '/', name,'.jpg']);
 
    %check if the image is grey scale or rgb
    if size(im,3)==1
        im = cat(3,im,im,im);
    end
    
    cim = {prepare_image(im)};
    
%     scores_6 = net_6.forward(cim);
%     scores_6 = scores_6{1};
%     scores_6 = mean(scores_6,2);
%     result_matrx_fc6 = [result_matrx_fc6, scores_6];
    
    scores_7 = net_7.forward(cim);
    scores_7 = scores_7{1};
    scores_7 = mean(scores_7,2);
    result_matrx_fc7 = [result_matrx_fc7, scores_7];
    toc
%     scores_8 = net_8.forward(cim);
%     scores_8 = scores_8{1};
%     scores_8 = mean(scores_8,2);
%     result_matrx_fc8 = [result_matrx_fc8, scores_8];
%     
%     scores_9 = net_9.forward(cim);
%     scores_9 = scores_9{1};
%     scores_9 = mean(scores_9,2);
%     result_matrx_fc9 = [result_matrx_fc9, scores_9];
%     
%     scores_10 = net_10.forward(cim);
%     scores_10 = scores_10{1};
%     scores_10 = mean(scores_10,2);
%     result_matrx_fc10 = [result_matrx_fc10, scores_10];
%     
%     scores_11 = net_11.forward(cim);
%     scores_11 = scores_11{1};
%     scores_11 = mean(scores_11,2);
%     result_matrx_fc11 = [result_matrx_fc11, scores_11];
    
    disp(['process ' num2str(i) ' images']);
    
end
%switch the cow and rol
% result_matrx_fc6 = result_matrx_fc6';
result_matrx_fc7 = result_matrx_fc7';
% result_matrx_fc8 = result_matrx_fc8';
% result_matrx_fc9 = result_matrx_fc9';
% result_matrx_fc10 = result_matrx_fc10';
% result_matrx_fc11 = result_matrx_fc11';
% 
% save('feature_fc6.mat','result_matrx_fc6');

save('/home/mj/桌面/notgood.mat','result_matrx_fc7');
% save('feature_fc8.mat','result_matrx_fc8');
% save('feature_fc9.mat','result_matrx_fc9');
% save('feature_fc10.mat','result_matrx_fc10');
% save('feature_fc11.mat','result_matrx_fc11');


