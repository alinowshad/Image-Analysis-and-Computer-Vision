clear all;
close all;
clc
%% Reading the Image
Image = imread('PalazzoTe.jpg');
figure(1);
imshow(Image);
title('Original image');

subplot(1, 2, 1);
imshow(Image);
title('Original Image');
Image = imrotate(Image, 270);
subplot(1, 2, 2);
imshow(Image);
title('Rotated Image');
%% Transfoming the rgb image to gray scale and negative transformation
grayImage = rgb2gray(Image);
negativeImage = 255 - Image;

figure(2);

subplot(1, 2, 1);
imshow(grayImage);
title('Gray Scale Image');

subplot(1, 2, 2);
imshow(negativeImage);
title('Negative Image');
%% Gray Scale Negative Transformation
grayNegativeImage = rgb2gray(negativeImage);
figure(3);
subplot(1, 2, 1);
imshow(Image);
title('Original Image');

subplot(1, 2, 2);
imshow(grayNegativeImage);
title('Negative Image');
%% Compute and display the histogram
subplot(2, 2, [3 4]);
figure(4);
imhist(grayImage);
title('Histogram Analysis');
xlabel('Pixel Intensity');
ylabel('Frequency');
%% Calculate Cumulative Distribution Function (CDF)
[counts, bins] = imhist(grayImage);
cdf = cumsum(counts) / sum(counts);

figure(5);
plot(bins, cdf);
title('Cumulative Distribution Function');
xlabel('Pixel Intensity');
ylabel('Cumulative Probability');

%% Performing Histogram Equalization
enhancedImage = histeq(grayImage);
% Showing the original and enhanced images side by side
figure(6);
subplot(2, 4, 1);
imshow(grayImage);
title('Original Image');

subplot(2, 4, 2);
imhist(grayImage);
title('Histogram Analysis');
xlabel('Pixel Intensity');
ylabel('Frequency');

subplot(2, 4, 3);
imshow(enhancedImage);
title('Enhanced Image');

subplot(2, 4, 4);
imhist(enhancedImage);
title('Histogram Analysis');
xlabel('Pixel Intensity');
ylabel('Frequency');
%% Performing Thersholding
% With above histogram method we can find an good threshold for the image
% The threshold separates the image into regions based on pixel intensity.
% Selecting  a threshold
threshold = 0.5;
% Threshold the image
binaryImage = grayImage > threshold;
% Showing the original and thresholded images side by side
figure (7);
subplot(1, 2, 1);
imshow(grayImage);
title('Original Image');

subplot(1, 2, 2);
imshow(binaryImage);
title('Thresholded Image');
%% Performing Thresholding on the Equalized Image

threshold = 0.5;
% Threshold the image
binaryImage = enhancedImage > threshold;
% Showing the original and thresholded images side by side
figure (8);
subplot(1, 2, 1);
imshow(enhancedImage);
title('Equalized Image');

subplot(1, 2, 2);
imshow(binaryImage);
title('Thresholded Image');

%% Perfoming a better approach for Thersholding
% As you saw in the above picture with this value our theresholded image
% is completely white which is shows that the value we selected it is not
% good
% Use Otsu's method to find an automatic threshold
level = graythresh(grayImage);
% Threshold the image
binaryImage = imbinarize(grayImage, level);
% Showing the original and thresholded images side by side
level = graythresh(enhancedImage);
binaryEqualizedImage = imbinarize(enhancedImage, level);
figure(9);
subplot(2, 4, 1);
imshow(grayImage);
title('Original Gray Image');

subplot(2, 4, 2);
imshow(binaryImage);
title('Thresholded Image');

subplot(2, 4, 3);
imshow(enhancedImage);
title('Equalized Image');

subplot(2, 4, 4);
imshow(binaryEqualizedImage);
title('Thresholded Equalized Image');

%% Color image analysis
% Spliting the color channels
redChannel = Image(:,:,1);
greenChannel = Image(:,:,2);
blueChannel = Image(:,:,3);


% Computing the histograms for each channel
[countsRed, binsRed] = imhist(redChannel);
[countsGreen, binsGreen] = imhist(greenChannel);
[countsBlue, binsBlue] = imhist(blueChannel);
% Showing the histograms
figure(10);
subplot(3, 1, 1);
bar(binsRed, countsRed, 'r');
title('Red Channel Histogram');

subplot(3, 1, 2);
bar(binsGreen, countsGreen, 'g');
title('Green Channel Histogram');

subplot(3, 1, 3);
bar(binsBlue, countsBlue, 'b');
title('Blue Channel Histogram');
%% Sobel Edge Detection
% Now we start with different edge detection methods, we start with sobel
sobelEdge = edge(grayImage, 'sobel', 0.03);
sobelEdge_eq = edge(enhancedImage, 'sobel', 0.03);
figure(11);
subplot(1, 2, 1);
imshow(sobelEdge);
title('Sobel Edge Detection Gray Scale Image');

subplot(1, 2, 2);
imshow(sobelEdge_eq);

title('Sobel Edge Detection Equalized Image');
% Label connected components
[labeledEdges, numEdges] = bwlabel(sobelEdge);

% Visualize the labeled edges
figure(12);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Prewitt Edge Detection
prewittEdge = edge(grayImage, 'prewitt', 0.02);
prewittEdge_eq = edge(enhancedImage, 'prewitt', 0.02);

figure(13);
imshow(prewittEdge);
title('Prewitt Edge Detection');

figure(14);
subplot(1, 2, 1);
imshow(prewittEdge);
title('Prewitt Edge Detection');

subplot(1, 2, 2);
imshow(prewittEdge_eq);
title('Prewitt Edge Detection Equalized Image');

% Label connected components
[labeledEdges, numEdges] = bwlabel(prewittEdge);

% Visualize the labeled edges
figure(15);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Canny Edge Detection
cannyEdge = edge(grayImage, 'canny',[.02 .3],15);
cannyEdge_eq = edge(enhancedImage, 'canny',[.02 .3],15);

figure(16);
imshow(cannyEdge);
title('Canny Edge Detection');

figure(17);
subplot(1, 2, 1);
imshow(cannyEdge);
title('Canny Edge Detection');

subplot(1, 2, 2);
imshow(cannyEdge_eq);

title('Canny Edge Detection Equalized Image');
% Label connected components
[labeledEdges, numEdges] = bwlabel(cannyEdge);

% Visualize the labeled edges
figure(18);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Laplacian of Gaussian (LoG)
h = fspecial('log');
logEdge = imfilter(grayImage, h);
logEdge_eq = imfilter(enhancedImage, h);
figure(19);
imshow(logEdge);
title('LoG Edge Detection');

figure(20);
subplot(1, 2, 1);
imshow(logEdge);
title('LoG Edge Detection');

subplot(1, 2, 2);
imshow(logEdge_eq);
title('LoG Edge Detection Equalized Image');
%% Roberts Edge Detection
robertsEdge = edge(grayImage, 'roberts', 0.03);
robertsEdge_eq = edge(enhancedImage, 'roberts', 0.03);

figure(21);
imshow(robertsEdge);
title('Roberts Edge Detection');

figure(22);
subplot(1, 2, 1);
imshow(robertsEdge);
title('Roberts Edge Detection');

subplot(1, 2, 2);
imshow(robertsEdge_eq);

title('Roberts Edge Detection Equalized Image');
[labeledEdges, numEdges] = bwlabel(robertsEdge);

% Visualize the labeled edges
figure(23);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Display all edge detection results together
figure(24);
subplot(2, 3, 1);
imshow(sobelEdge);
title('Sobel Edge Detection');

subplot(2, 3, 2);
imshow(prewittEdge);
title('Prewitt Edge Detection');

subplot(2, 3, 3);
imshow(cannyEdge);
title('Canny Edge Detection');

subplot(2, 3, 4);
imshow(logEdge);
title('LoG Edge Detection');

subplot(2, 3, 5);
imshow(robertsEdge);
title('Roberts Edge Detection');

subplot(2, 3, 6);
imshow(grayImage);
title('Original Image');


%% using the gray scale equalization
corners_eq = detectHarrisFeatures(enhancedImage, 'MinQuality', 0.02);
corners = detectHarrisFeatures(grayImage, 'MinQuality', 0.02);

figure(25)
subplot(1, 2, 1);
imshow(Image);
hold on;
plot(corners);
title('Harris Corner Detection');
hold off;

subplot(1, 2, 2);
imshow(Image);
hold on;
plot(corners_eq);
hold off;
title('Harris Corner Detection using equalized image');
%%  RGB to YCbCr transformation
% Convert the image to double precision for accurate calculations
rgbImage = im2double(Image);

% RGB to YCbCr transformation
Y = 0.299 * rgbImage(:, :, 1) + 0.587 * rgbImage(:, :, 2) + 0.114 * rgbImage(:, :, 3);
Cb = -0.1687 * rgbImage(:, :, 1) - 0.3313 * rgbImage(:, :, 2) + 0.5 * rgbImage(:, :, 3) + 128;
Cr = 0.5 * rgbImage(:, :, 1) - 0.4187 * rgbImage(:, :, 2) - 0.0813 * rgbImage(:, :, 3) + 128;

% Display the results
figure(26);
subplot(1, 4, 1); imshow(rgbImage); title('RGB Image');
subplot(1, 4, 2); imshow(Y, []); title('Luminance (Y)');
subplot(1, 4, 3); imshow(Cb, []); title('Chrominance Blue (Cb)');
subplot(1, 4, 4); imshow(Cr, []); title('Chrominance Red (Cr)');

%% Perform contrast stretching


% Convert the image to double precision for accurate calculations
originalImage = im2double(Image);

minIntensity = min(originalImage(:));
maxIntensity = max(originalImage(:));

% Scale the image using the full dynamic range [0, 1]
contrastStretchedImage = (originalImage - minIntensity) / (maxIntensity - minIntensity);

% Display the original and contrast-stretched images side by side
figure(27);
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(contrastStretchedImage);
title('Contrast Stretched Image');
%% Line Detection
% Use Hough transform for line detection
[H, T, R] = hough(prewittEdge);
peaks = houghpeaks(H, 200); % Adjust the number of peaks as needed
lines = houghlines(prewittEdge, T, R, peaks);

% Display the original image with detected lines
figure(28);
imshow(prewittEdge);
hold on;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'r');
end
title('Line Detection');
hold off;
%% Maskig RGB image
% Load the RGB image
rgbImage = Image;

% Display the image and allow the user to click on points
figure(29);
imshow(rgbImage);
title('Click on multiple pixels to get their RGB values');

% Get the number of points from the user
numPoints = input('Enter the number of points to select: ');

% Initialize arrays to store the coordinates and RGB values
xCoordinates = zeros(numPoints, 1);
yCoordinates = zeros(numPoints, 1);
rgbValues = zeros(numPoints, 3);

% Get user input for each point
for i = 1:numPoints
    fprintf('Click on point %d\n', i);
    
    % Get the user input (click on a pixel)
    [x, y] = ginput(1);
    
    % Round the coordinates to the nearest integers
    x = round(x);
    y = round(y);
    
    % Get the RGB values at the selected pixel
    pixelValue = rgbImage(y, x, :);
    
    % Store the coordinates and RGB values in arrays
    xCoordinates(i) = x;
    yCoordinates(i) = y;
    rgbValues(i, :) = pixelValue;
end

% Create a binary mask based on the selected RGB values
mask = false(size(rgbImage, 1), size(rgbImage, 2));

for i = 1:numPoints
    % Define a tolerance for RGB values (adjust as needed)
    tolerance = 20;
    
    % Create masks for each channel
    maskR = abs(double(rgbImage(:, :, 1)) - rgbValues(i, 1)) <= tolerance;
    maskG = abs(double(rgbImage(:, :, 2)) - rgbValues(i, 2)) <= tolerance;
    maskB = abs(double(rgbImage(:, :, 3)) - rgbValues(i, 3)) <= tolerance;
    
    % Combine the masks using logical AND
    colorMask = maskR & maskG & maskB;
    
    % Combine with the overall mask using logical OR
    mask = mask | colorMask;
end

% Convert the binary mask to a single channel image
maskedImage = bsxfun(@times, rgbImage, uint8(mask));

% Display the original image, the selected points, and the masked image


figure(29);
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(maskedImage);
title('Masked Image');
%% Masking Gray Scale Image

% Display the image and allow the user to click on points
figure(34);
imshow(grayImage);
title('Click on multiple pixels to get their intensity values');

% Get the number of points from the user
numPoints = input('Enter the number of points to select: ');

% Initialize arrays to store the coordinates and intensity values
xCoordinates = zeros(numPoints, 1);
yCoordinates = zeros(numPoints, 1);
intensityValues = zeros(numPoints, 1);

% Get user input for each point
for i = 1:numPoints
    fprintf('Click on point %d\n', i);
    
    % Get the user input (click on a pixel)
    [x, y] = ginput(1);
    
    % Round the coordinates to the nearest integers
    x = round(x);
    y = round(y);
    
    % Get the intensity value at the selected pixel
    intensityValue = grayImage(y, x);
    intensityValues(i) = intensityValue;
    
    % Store the coordinates
    xCoordinates(i) = x;
    yCoordinates(i) = y;
end

% Create a binary mask based on the selected intensity values
mask = false(size(grayImage));

for i = 1:numPoints
    % Define a tolerance for intensity values (adjust as needed)
    tolerance = 20;
    
    % Calculate the differences between pixel values and selected intensity values
    diffIntensity = abs(double(grayImage) - intensityValues(i));
    
    % Create the mask
    intensityMask = diffIntensity <= tolerance;
    
    % Combine with the overall mask using logical OR
    mask = mask | intensityMask;
end

% Convert the binary mask to a single channel image
maskedImage = bsxfun(@times, grayImage, uint8(mask));

% Display the original image, the selected points, and the masked image
figure(30);
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(maskedImage);
title('Masked Image');

cannyEdge = edge(maskedImage, 'canny', [.1 .2],6);

% Label connected components
[labeledEdges, numEdges] = bwlabel(cannyEdge);

% Visualize the labeled edges
figure(31);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Filtered Image Using Convolution

grayImage = double(grayImage);
% ---- Filters ---------------------- 
% filter to detect horizontal edges
gx = [1;0;-1]./sqrt(2);
% filter to detect vertical edges
gy = [1 0 -1]./sqrt(2);
% smoothing filter
F = [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1];
F = F./norm(F);
smooth_image = conv2(grayImage, F, 'same');
vert_ = conv2(smooth_image, gx, 'same');
hor_ = conv2(smooth_image, gy, 'same');
filteredImage = sqrt(vert_.^2 + hor_.^2);
filteredImage = uint8(filteredImage);
imshow(filteredImage)


cannyEdge = edge(filteredImage, 'canny', [.1 .2],6);

% Label connected components
[labeledEdges, numEdges] = bwlabel(cannyEdge);


figure(31);
subplot(1, 3, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 3, 2);
imshow(filteredImage)
title('Filtered Image');

subplot(1, 3, 3);
imshow(label2rgb(labeledEdges, 'jet', 'k', 'shuffle'));
title('Labeled Edges');
%% Create a prenormalized Scharr filter

% 3x3 gaussian
a = 0.003744;
b = 0.970049;
fkg = [a a a; a b a; a a a];

fk_scharr = [-3-3i, 0-10i, 3-3i; -10, 0, 10; -3+3i, 0+10i, 3+3i];
fk_scharr = fk_scharr / sum(abs(fk_scharr(:))); % Normalize the filter

% Assuming grayImage is the input grayscale image
smooth_image = conv2(im2single(grayImage), fkg, 'same'); % Convolve with Gaussian kernel

% Compute vertical and horizontal derivatives using the Scharr filter
vert_ = conv2(smooth_image, fk_scharr.', 'same'); % d/dx
hor_ = conv2(smooth_image, fk_scharr, 'same'); % d/dy

% Compute the gradient magnitude
scharrfilteredImage = sqrt(vert_.^2 + hor_.^2);

% Display the resulting image

figure(32);
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(scharrfilteredImage)
title('Scharr Filtered Image Edges');
%%


% Use Canny edge detection with adjusted thresholds
edges = edge(enhancedImage, 'canny', [.01 .1],30); % Adjust these values for higher sensitivity

% Apply Hough transform
[H,theta,rho] = hough(edges);

% Find peaks in the Hough transform matrix with an adjusted threshold
peaks = houghpeaks(H, 50000, 'threshold', ceil(0.1 * max(H(:)))); % Adjust this value for higher sensitivity

% Extract lines using Hough transform matrix and peaks
lines = houghlines(edges, theta, rho, peaks, 'FillGap', 30, 'MinLength', 10);

% Plot the lines on the original image
figure(33);
imshow(Image);
title("Line Detection with Canny Edge Detector and Hough Transform")
hold on;

for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', [0, 1, 0]);
end

hold off;

%% Finding the Vanishing Line in the Image
% Display the image

imshow(grayImage);
title('Geometry');

% Get the size (shape) of the image
imageSize = size(grayImage);

% Display the size
disp('Image Size:');
disp(imageSize);


hold on;
%{
% Select points interactively
[x, y] = getpts();
disp(x);
disp(y);

% Fit conic to five points
c = [x(1); y(1); 1];
d = [x(5); y(5); 1];
A = [x.^2, x.*y, y.^2, x, y, ones(size(x))];
N = null(A);
cc = N(:, 1);

[a1, b1, c1, d1, e1, f1] = deal(cc(1), cc(2), cc(3), cc(4), cc(5), cc(6));
C1 = [a1, b1/2, d1/2; b1/2, c1, e1/2; d1/2, e1/2, f1];

%save('C1.mat', 'C1');

% Fit another conic to five more points
[x, y] = getpts();
a = [x(1); y(1); 1];
b = [x(5); y(5); 1];
A = [x.^2, x.*y, y.^2, x, y, ones(size(x))];
N = null(A);
cc = N(:, 1);

[a2, b2, c2, d2, e2, f2] = deal(cc(1), cc(2), cc(3), cc(4), cc(5), cc(6));
C2 = [a2, b2/2, d2/2; b2/2, c2, e2/2; d2/2, e2/2, f2];
%save('C2.mat', 'C2');
%}
load('C1.mat');
load('C2.mat');
[a1, b1, c1, d1, e1, f1] = deal(C1(1), C1(2)*2, C1(5), C1(3)*2, C1(6)*2, C1(9));
[a2, b2, c2, d2, e2, f2] = deal(C2(1), C2(2)*2, C2(5), C2(3)*2, C2(6)*2, C2(9));


%save('a.mat', 'a');
%save('b.mat', 'b');
%save('c.mat', 'c');
%save('d.mat', 'd');
load("a.mat");
load('b.mat');
load('c.mat');
load('d.mat');

% Solve the equations for the conics
syms 'x' 'y';
eq1 = a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1;
eq2 = a2*x^2 + b2*x*y + c2*y^2 + d2*x + e2*y + f2;

eqns = [eq1 == 0, eq2 == 0];
S = solve(eqns, [x, y]);

% Get the intersection points
s1 = [double(S.x(1)); double(S.y(1)); 1];
s2 = [double(S.x(2)); double(S.y(2)); 1];
s3 = [double(S.x(3)); double(S.y(3)); 1];
s4 = [double(S.x(4)); double(S.y(4)); 1];

% Compute the horizon
II = s1;
JJ = s2;
%save('II.mat', 'II');
%save('JJ.mat', 'JJ');
load('II.mat');
load('JJ.mat');
%horizon = hcross(II, JJ);
%save('horizon.mat', 'horizon')
load('horizon.mat');
% Display the horizon
disp('Horizon:');
disp(horizon);

% Plot the selected points
scatter(subs(x, S), subs(y, S), 'g', 'filled');

% Draw the horizon line
x_vals = 1:imageSize(2);
y_vals = (-horizon(1)*x_vals - horizon(3))/horizon(2);
line(x_vals, y_vals, 'Color', 'r', 'LineWidth', 2);

c1_center=inv(C1)*horizon; %compute center of c1
c2_center=inv(C2)*horizon; %compute center of c2
%save('c1_center.mat', 'c1_center');
%save('c2_center.mat', 'c2_center');
load('c1_center.mat');
load('c2_center.mat');
lc1c2=hcross(c1_center,c2_center); % compute cone axis a
c1_center=c1_center./c1_center(3);
c2_center=c2_center./c2_center(3);
plot([c1_center(1), c2_center( 1)], [c1_center(2), c2_center(2)], 'b');
scatter(c1_center(1),c1_center(2),'filled',MarkerFaceColor='b');
scatter(c2_center(1),c2_center(2),'filled',MarkerFaceColor='b');
text(c1_center(1), c1_center(2), 'c1', 'FontSize', 20, 'Color', 'r');
text(c2_center(1), c2_center(2), 'c2', 'FontSize', 20, 'Color', 'r');

lab=hcross(a,b); 
lcd=hcross(c,d);
v1=hcross(lab,lcd); % compute vanishing point V1
%save('v1.mat', 'v1');
load('v1.mat');

% Plot points
plot([a(1), b(1), c(1), d(1)], [a(2), b(2), c(2), d(2)], 'ro');
text(a(1), a(2), 'A', 'FontSize', 12, 'Color', 'g');
text(b(1), b(2), 'B', 'FontSize', 12, 'Color', 'g');
text(c(1), c(2), 'C', 'FontSize', 12, 'Color', 'g');
text(d(1), d(2), 'D', 'FontSize', 12, 'Color', 'g');


plot([a(1), v1( 1)], [a(2), v1(2)], 'b');
plot([b(1), v1( 1)], [b(2), v1(2)], 'b');
plot([c(1), v1(1)], [c(2), v1(2)], 'b');
plot([d(1), v1(1)], [d(2), v1(2)], 'b');

text(v1(1), v1(2), 'v1', 'FontSize', 20, 'Color', 'r');

lac = hcross(a, c);
lbd = hcross(b, d);
v2 = hcross(lac, lbd); % compute vanishing point V2
%save('v2.mat', 'v2');
load('v2.mat');
disp("Vanishing Pont 1")
disp(v1);
disp("Vanishing Point 2")
disp(v2);

% Plot lines connecting points to vanishing point v2
plot([a(1), v2(1)], [a(2), v2(2)], 'g');
plot([c(1), v2(1)], [c(2), v2(2)], 'g');
plot([d(1), v2(1)], [d(2), v2(2)], 'g');
plot([b(1), v2(1)], [b(2), v2(2)], 'g');

% Plot vanishing point v2
text(v2(1), v2(2), 'V2', 'FontSize', 12, 'Color', 'm');
hold off;
%% Computing the calibration matrix
syms fx, syms fy, syms u0, syms v0;
K = [fx, 0, u0; ...
     0, fy, v0; ...
     0, 0, 1];
w_dual = K*K.';
w = inv(w_dual);
imDCCP = II*JJ' + JJ*II'; % image of coinc dual to circular points
imDCCP = imDCCP./norm(imDCCP);

[U,D,V] = svd(imDCCP);
D(3,3) = 1;
A = U*sqrt(D);
H_R=inv(A); % shape reconstruction homography
H=inv(H_R);
h1=H(:,1);
h2=H(:,2);
h3=H(:,3);
eqs = [ 
        h1.'*w* h2;
        %h2.'*w*h3;
        v2.'*w*h3;
        v2.'*w*h2;
        h1.'*w*h1-h2.'*w*h2;
        %v1.'*w*v2;
        %v1.'*w*v3;
        %v2.'*w*v3;
      ];

res = solve(eqs);
fx = real(double(res.fx)); fx = fx(fx > 0); fx = fx(1);
fy = real(double(res.fy)); fy = fy(fy > 0); fy = fy(1);
u0 = real(double(res.u0)); u0 = u0(1);
v0 = real(double(res.v0)); v0 = v0(1);

K = [fx, 0, u0; ...
     0, fy, v0; ...
     0, 0, 1]
%save('K.mat', 'K');
%load('K.mat');
%%
disp("K")
disp(K)
%% comptute roto-translation matrix
Q=inv(K)*H;
i_pi=Q(:,1);
j_pi=Q(:,2);
O_pi=Q(:,3);
k_pi=cross(i_pi,j_pi);
R_t=[i_pi j_pi k_pi O_pi;
     0     0    0     1]

c1_3d=R_t*[c1_center(1);c1_center(2);0;c1_center(3)]; % compute c1 center in camera reference
c2_3d=R_t*[c2_center(1);c2_center(2);0;c2_center(3)]; % compute c1 center in camera reference
c1_3d=c1_3d./c1_3d(4)
c2_3d=c2_3d./c2_3d(4)
figure(1);
scatter3(c1_3d(1),c1_3d(2),c1_3d(3),'filled')
text(c1_3d(1),c1_3d(2),c1_3d(3),'C1_center')
hold on
scatter3(c2_3d(1),c2_3d(2),c2_3d(3),'filled')
text(c2_3d(1),c2_3d(2),c2_3d(3),'C2_center')
scatter3(0,0,0,'r')
text(0,0,0,'camera')
plot3([c1_3d(1) c2_3d(1)],[c1_3d(2) c2_3d(2)],[c1_3d(3) c2_3d(3)],'color','b','LineWidth',2)
xlabel('X');ylabel("Y");zlabel("Z")


%save('c1_3d.mat', 'c1_3d');
%save('c2_3d.mat', 'c2_3d');
load('c1_3d.mat');
load('c2_3d.mat');
%%
par_geo_C1 = AtoG([C1(1), C1(2)*2, C1(5), C1(3)*2, C1(6)*2, C1(9)]);
par_geo_C2 = AtoG([C2(1), C2(2)*2, C2(5), C2(3)*2, C2(6)*2, C2(9)]);
c1_center = par_geo_C1(1:2);
c2_center = par_geo_C2(1:2);
radius1 = calculate_radius(c1_center, c);
radius2 = calculate_radius(c2_center, a);
distance = calculate_distance(c1_center, c2_center)
fprintf('The radius of the circle 1 is: %.2f\n', radius1);
fprintf('The radius of the circle 2 is: %.2f\n', radius2);
fprintf('The distance of the two cross section is: %.2f\n', distance);

C1_transformed = H' * C1 * H;
C2_transformed = H' * C2 * H;
c_transformed = H * c;
a_transformed = H * a;

par_geo_C1_rect = AtoG([C1_transformed(1), C1_transformed(2)*2, C1_transformed(5), C1_transformed(3)*2, C1_transformed(6)*2, C1_transformed(9)]);
par_geo_C2_rect = AtoG([C2_transformed(1), C2_transformed(2)*2, C2_transformed(5), C2_transformed(3)*2, C2_transformed(6)*2, C2_transformed(9)]);
c1_center_rec = par_geo_C1_rect(1:2);
c2_center_rec = par_geo_C2_rect(1:2);
radius1_rec = calculate_radius(c1_center_rec, c_transformed);
radius2_rec = calculate_radius(c2_center_rec, a_transformed);
distance_rec = calculate_distance(c1_center_rec, c2_center_rec)
fprintf('The radius of the circle is: %.2f\n', radius1_rec);
fprintf('The radius of the circle is: %.2f\n', radius2_rec);
fprintf('The distance of the two cross section is: %.2f\n', distance_rec);

% Calculate the ratio for original points
ratio_original = calculate_ratio(radius1, radius2, distance);
disp(ratio_original)
% Calculate the ratio for rectified points
ratio_rectified = calculate_ratio(radius1_rec, radius2_rec, distance_rec);
disp('The ratio of the radius to distance for rectified points is:');
disp(ratio_rectified);
disp('The ratio of the radius to distance for original points is:');
disp(ratio_original);
%%

im_err1 = zeros(size(grayImage, 1), size(grayImage, 2));
im_err2 = zeros(size(grayImage, 1), size(grayImage, 2));

% Calculate errors based on homography matrices C1 and C2
for i = 1:size(grayImage, 1)
    for j = 1:size(grayImage, 2)
        im_err1(i, j) = [j, i, 1] * C1 * [j; i; 1];
        im_err2(i, j) = [j, i, 1] * C2 * [j; i; 1];
    end
end

% Create a binary mask based on negative errors
msk = (im_err1 < 0) + (im_err2 < 0);

% Weighted combination of the original image and the mask
lambda = 0.5;
figure;
imshow(lambda * grayImage + (1 - lambda) * uint8(255 * msk));

%%
zminRadius = max(radius1);
zmaxRadius =max(radius2);
zValues = linspace(0.0, distance, 100);
thetaValues = linspace(0, 359, 360) * pi / 180;

% Build dummy cone data
tcount = length(thetaValues);
zcount = length(zValues);
coneData = rand(tcount, zcount);

% Initialization and Sorting
zmin = min(zValues);
zmax = max(zValues);
smallestRadius = min(zminRadius, zmaxRadius);
biggestRadius = max(zminRadius, zmaxRadius);

% Ensure thetaValues range in -pi;+pi;
thetaValues = mod(thetaValues + pi, 2*pi) - pi;
[thetaValues, si] = sort(thetaValues);
coneData = coneData(si, :);

% Intercept Theorem (Thales)
BC = biggestRadius - smallestRadius;
AE = zmax - zValues(:);
AC = (zmax - zmin);
DE = (BC * AE) / AC;
radiuses = smallestRadius + DE;

% Projection
xvalues = biggestRadius * thetaValues;
xcount = length(xvalues);
zcount = length(zValues);
planeData = zeros(xcount, zcount);

for zi = 1:zcount
    localX = radiuses(zi) * thetaValues;
    localValues = coneData(:, zi);
    planeData(:, zi) = interp1(localX, localValues, xvalues, 'linear', 0);
end

% Display in a single figure with subplots

% Create a figure with two subplots
figure;

% Subplot 1: 3D Plot of the Cylinder
subplot(1, 2, 1);
[Xcyl, Ycyl, Zcyl] = cylinder([biggestRadius, smallestRadius], 100);
Zcyl = Zcyl * (zmax - zmin) + zmin;
surf(Xcyl, Ycyl, Zcyl, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Plot of the Cylinder');

% Subplot 2: 2D Plane
subplot(1, 2, 2);
[X, Z] = ndgrid(xvalues, zValues);
pcolor(X, Z, planeData);
shading flat;
xlabel('X');
ylabel('Z');
title('2D Plane Corresponding to the Cylinder');
%%
function ratio = calculate_ratio(radius1, radius2, distance)
    % Calculate the ratio between the radius of the circular cross sections and their distance
    ratio = (radius1 + radius2) / distance;
end

function radius = calculate_radius(center, point_on_circle)
    % center and point_on_circle are 2-element vectors [x, y]
    radius = sqrt((point_on_circle(1) - center(1))^2 + (point_on_circle(2) - center(2))^2);
end


function distance = calculate_distance(point1, point2)
    x1 = point1(1);
    y1 = point1(2);
    x2 = point2(1);
    y2 = point2(2);
    % Calculate the distance between the two points using the distance formula
    distance = sqrt((x2 - x1)^2 + (y2 - y1)^2);
end

function c = hcross(a,b)
c = cross(a,b);
c = c/c(3);
end



function [ParG,code] = AtoG(ParA)
%  Conversion of algebraic parameters 
%  ParA = [A,B,C,D,E,F]'-  parameter vector of the conic:  Ax^2 + Bxy + Cy^2 +Dx + Ey + F = 0
%  to geometric parameters  ParG = [Center(1:2), Axes(1:2), Angle]'
%  code is:  1 - ellipse  
%            2 - hyperbola 
%            3 - parabola
%           -1 - degenerate cases  
%            0 - imaginary ellipse  
%            4 - imaginary parelle lines
%
%
%  Copyright 2011 Hui Ma
tolerance1 = 1.e-10;
tolerance2 = 1.e-20;
% ParA = ParA/norm(ParA);
if (abs(ParA(1)-ParA(3)) > tolerance1)
    Angle = atan(ParA(2)/(ParA(1)-ParA(3)))/2;
else
    Angle = pi/4;
end
c = cos(Angle);  s = sin(Angle);
Q = [c s; -s c];
M = [ParA(1)  ParA(2)/2;  ParA(2)/2  ParA(3)];
D = Q*M*Q';
N = Q*[ParA(4); ParA(5)];
O = ParA(6);
if (D(1,1) < 0) && (D(2,2) < 0)
    D = -D;
    N = -N;
    O = -O;
end
UVcenter = [-N(1,1)/2/D(1,1); -N(2,1)/2/D(2,2)];
free = O - UVcenter(1,1)*UVcenter(1,1)*D(1,1) - UVcenter(2,1)*UVcenter(2,1)*D(2,2);
% if the determinant of [A B/2 D/2;B/2 C E/2;D/2 E/2 F]is zero 
% and if K>0,then it is an empty set;
% otherwise the conic is degenerate
Deg =[ParA(1),ParA(2)/2,ParA(4)/2;...
     ParA(2)/2,ParA(3),ParA(5)/2;...
     ParA(4)/2,ParA(5)/2,ParA(6)];
K1=[ParA(1),ParA(4)/2;ParA(4)/2 ParA(6)];
K2=[ParA(3),ParA(5)/2;ParA(5)/2 ParA(6)];
K = det(K1)+ det(K2);
if (abs(det(Deg)) < tolerance2)
    if (abs(det(M))<tolerance2) &&(K > tolerance2)
        code = 4;  % empty set(imaginary parellel lines)
    else
        code = -1; % degenerate cases
    end
else
    if (D(1,1)*D(2,2) > tolerance1)
        if (free < 0)
            code = 1; % ellipse
        else
            code = 0; % empty set(imaginary ellipse)
        end
    elseif (D(1,1)*D(2,2) < - tolerance1)
        code = 2;  % hyperbola
    else
        code = 3;  % parabola
    end
end
XYcenter = Q'*UVcenter;
Axes = [sqrt(abs(free/D(1,1))); sqrt(abs(free/D(2,2)))];
if code == 1 && Axes(1)<Axes(2)
    AA = Axes(1); Axes(1) = Axes(2); Axes(2) = AA;
    Angle = Angle + pi/2;
end
if code == 2 && free*D(1,1)>0
    AA = Axes(1); Axes(1) = Axes(2); Axes(2) = AA;
    Angle = Angle + pi/2;
end
while Angle > pi
    Angle = Angle - pi;
end
while Angle < 0
    Angle = Angle + pi;
end
ParG = [XYcenter; Axes; Angle];
end