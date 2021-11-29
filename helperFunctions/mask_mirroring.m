function mask_out = mask_mirroring(mask_in)
% input: mask_in
% about mask: ROI we want = True, ROI we don't want = False
mask_out = true(size(mask_in));

% column (left-right) neighbors mirroring
mask_in = ~mask_in;
for r = 1:8
    for c = 1:10
        valid = c-1>=1 && c+1<=10;
        if valid
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            lt_nei = mask_in((r-1)*89+1:r*89, (c-2)*89+1:(c-1)*89);
            rt_nei = mask_in((r-1)*89+1:r*89, c*89+1:(c+1)*89);
            mixed = (lt_nei | center) | rt_nei;         
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        elseif c-1 < 1 % left-most column
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            rt_nei = mask_in((r-1)*89+1:r*89, c*89+1:(c+1)*89);
            mixed = center | rt_nei;
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        elseif c+1 > 10 % right-most column
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            lt_nei = mask_in((r-1)*89+1:r*89, (c-2)*89+1:(c-1)*89);
            mixed = center | lt_nei;
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        end
    end
end

% row (up-down) neighbors mirroring
for r = 1: 2: 8
    st = mask_out((r-1)*89+1: r*89, :);
    rt = mask_out(r*89+1: (r+1)*89, :);
    mixed = st | rt;
    mask_out((r-1)*89+1: (r+1)*89, :) = repmat(mixed, [2,1]);
end

mask_out = ~mask_out;

end