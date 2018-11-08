im = imread('whereswaldo.png');
filter = imread('waldo.png');

output = findWaldo(im, filter);