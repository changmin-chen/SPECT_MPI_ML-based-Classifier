function show(img, str)
% helper func1: display image
img = uint8(img);
figure, imagesc(img)
axis off
if size(img,3) == 1
    colormap gray
end
if str
    title(str)
end
end