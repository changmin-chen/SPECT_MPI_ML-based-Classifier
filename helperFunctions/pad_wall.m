function P = pad_wall(W)
% helper func 7: padding for 3D registration

P = zeros(89, 89, 16);
if size(W, 3) == 10
    P(:,:,4:13) = W;
else
    P = W;
end
end