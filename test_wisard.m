addpath('mnistHelper', 'wisard');


%% load dataset

MNIST_PATH = 'mnistData/';

images_train = loadMNISTImages([MNIST_PATH 'train-images.idx3-ubyte']);
labels_train = cellstr(num2str(loadMNISTLabels([MNIST_PATH 'train-labels.idx1-ubyte'])));

images_test = loadMNISTImages([MNIST_PATH 't10k-images.idx3-ubyte']);
labels_test = cellstr(num2str(loadMNISTLabels([MNIST_PATH 't10k-labels.idx1-ubyte'])));




%% binarize images
train_imagesb = WiSARD.binarizeData(images_train', 'thermometer', 15);
test_imagesb = WiSARD.binarizeData(images_test', 'thermometer', 15);

% % transpose image to match python
% for i=1:size(train_imagesb, 2)
%     tmp = reshape(train_imagesb(:, i), 28, 28)';
%     train_imagesb(:,i) = tmp(:);
% end
% 
% for i=1:size(test_imagesb, 2)
%     tmp = reshape(test_imagesb(:, i), 28, 28)';
%     test_imagesb(:,i) = tmp(:);
% end
% LEVELS = 6;
% 
% train_imagesb = [];
% for i = 1 : size(train_images, 2)
%     image = thermometerize(train_images(:, i), LEVELS);
%     train_imagesb(:, i) = image(:);
% end
% 
% test_imagesb = [];
% for i = 1 : size(test_images, 2)
%     image = thermometerize(test_images(:, i), LEVELS);
%     test_imagesb(:, i) = image(:);
% end


%% Evaluate

classes = sort(unique(labels_train));
input_size = size(train_imagesb, 2);
nbits = 4;

w = WiSARD( classes, input_size, nbits);
w.fit(train_imagesb, labels_train);

w.cleanZeros();

[y, results] = w.predict(test_imagesb);

accuracy = sum(strcmp(y, labels_test)) / length(y);
disp(accuracy);



