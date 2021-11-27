function mask = get_mask(lvc, stress_centroids, rest_centroids)
% helper func 6: get half-circle mask

mask = true(size(lvc));
overlap_threshold = 0.75; % overlapping b/w area and circle-mask > 0.75
yaxis_threshold = 89/2; % distance to upper edge > distance to lower edge

% stress
count = 1;
for i = 1:2:7
    for j = 1:10
        lvc_p = lvc((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        radius = sqrt(sum(sum(lvc_p))/pi);
        tmp_mask = draw_circle(...
            stress_centroids(count,2)...
            , stress_centroids(count,1)...
            , radius);
        if (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) > overlap_threshold) && (stress_centroids(count,1)>=yaxis_threshold)
            tmp_mask(1:stress_centroids(count,1), :) = true; % we only need lower-half circle mask
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = tmp_mask;
        elseif (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) <= overlap_threshold) && find(sum(lvc_p, 2), 1,'last') >= 80
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = [true(74, 89); false(15, 89)];
        end
        count = count+1;
    end
end
% rest
count = 1;
for i = 2:2:8
    for j = 1:10
        lvc_p = lvc((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        radius = sqrt(sum(sum(lvc_p))/pi);
        tmp_mask = draw_circle(...
            rest_centroids(count,2)...
            , rest_centroids(count,1)...
            , radius);
        if (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) >overlap_threshold) && (rest_centroids(count,1)>=yaxis_threshold)
            tmp_mask(1:rest_centroids(count,1), :) = true; % we only need lower-half circle mask
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = tmp_mask;
        elseif (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) <= overlap_threshold) && find(sum(lvc_p, 2), 1,'last') >= 80
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = [true(74, 89); false(15, 89)];
        end
        count = count+1;
    end
end

function circle = draw_circle(x_center, y_center, radius)
% mask for a block element
[x, y] = meshgrid(1:89, 1:89);
distance = sqrt((x-x_center).^2 + (y-y_center).^2);
% draw circle mask
circle = true(89, 89);
circle(distance>radius) = false;
end

end