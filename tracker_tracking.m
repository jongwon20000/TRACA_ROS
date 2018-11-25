% function [trk, pos] = tracker_tracking(trk, im, bbox, opt)


% im = imread([BENCHMARK_PATH, scene_list(scene_idx).name, '/img/', img_files{frame_idx}]);

tic;

if(redetection > 0)
    patches_test = zeros(image_size(1), image_size(2), size(im, 3), 4, 'single');
else
    patches_test = zeros(image_size(1), image_size(2), size(im, 3), 3, 'single');
end


for iSc = -1:1
    patch = get_subwindow(im, pos, round(window_sz*(scale_ratio.^iSc)));
    patches_test(:,:,:,iSc+2) = single(imResample_fast(patch, image_size));
end

% figure(3);imshow(patch);drawnow;

if(redetection > 0)
    red_patch = get_subwindow(im, prev_pos(1:2), round(prev_pos(3:4)));
    patches_test(:,:,:,end) = single(imResample_fast(red_patch, image_size));
end

patch2 = get_subwindow(prev_im, pos, window_sz);
patches_update = single(imResample_fast(patch2, image_size));

patches = cat(4, patches_test, patches_update);

% feature extraction
if(gpus > 0)
    mbatch = gpuArray(patches);
else
    mbatch = patches;
end
mbatch = bsxfun(@minus, mbatch, feat_net.meta.normalization.averageImage);
feat_res = vl_simplenn_fast(feat_net, mbatch, [], feat_res);

mfeat = gather(feat_res(end).x);

xf = fft2(bsxfun(@times, cf_params.cos_window, mfeat));
xf_update = xf(:,:,:,end);
xf_test = xf(:,:,:,1:(end-1));


% correlation filter estimation
%         wf_curr = bsxfun(@times, cf_params.yf2, xf_update ./ (xf_update.*conj(xf_update) + cf_params.lambda));
wf_curr = cf_params.yf2.* xf_update ./ (xf_update.*conj(xf_update) + cf_params.lambda);
wf = (1-cf_params.gamma)*wf + cf_params.gamma*wf_curr;

if(redetection > 0)
    % response
    res = real(ifft2(bsxfun(@times, wf, conj(xf_test(:,:,:,1:3)))));
    
    % validate the CFs
    [r_max_v, r_max_idx] = max(res, [], 1);
    [~, c_max_idx] = max(r_max_v, [], 2);
    temp_ideal_y = bsxfun(@times, permute(yv(:, r_max_idx(c_max_idx)), [1,3,4,2]),...
        permute(yv(:, c_max_idx), [3,1,4,2]));
    ideal_y = reshape(temp_ideal_y, size(res));
    
    val = mean(mean((res - ideal_y).^2, 1), 2);
    response = permute(mean(bsxfun(@times, exp(-val_lambda*val), res), 3), [1,2,4,3]);
    
    % redetection response & validation
    red_res = real(ifft2(prev_wf.*conj(xf_test(:,:,:,end))));
    [red_r_max_v, red_r_max_idx] = max(red_res, [], 1);
    [~, red_c_max_idx] = max(red_r_max_v, [], 2);
    red_ideal_y = bsxfun(@times, permute(yv(:, red_r_max_idx(red_c_max_idx)), [1,3,2]),...
        permute(yv(:, red_c_max_idx), [3,1,2]));
    
    red_val = mean(mean((red_res - red_ideal_y).^2, 1), 2);
    red_response = mean(bsxfun(@times, exp(-val_lambda*red_val), red_res), 3);
    
else
    
    % response
    res = real(ifft2(bsxfun(@times, wf, conj(xf_test))));
    
    % validation
    [r_max_v, r_max_idx] = max(res, [], 1);
    [~, c_max_idx] = max(r_max_v, [], 2);
    temp_ideal_y = bsxfun(@times, permute(yv(:, r_max_idx(c_max_idx)), [1,3,4,2]),...
        permute(yv(:, c_max_idx), [3,1,4,2]));
    ideal_y = reshape(temp_ideal_y, size(res));
    
    val = mean(mean((res - ideal_y).^2, 1), 2);
    response = permute(mean(bsxfun(@times, exp(-val_lambda*val), res), 3), [1,2,4,3]);
    
end


% Validation score comparison
if(redetection > 0)
    cf_params.gamma = opt.gamma/10;%opt.redetect_gamma;
    redetect_success = max(red_response(:)) > max(response(:));
    redetection = redetection - 1;
    if(opt.visualization)
    end
else
    cf_params.gamma = opt.gamma;
    redetect_success = 0;
end

% redetection update
if(frame_idx > redetect_n_frame)
    if(max(response(:)) < max_res*redetect_eps && redetection == 0)
        redetection = redetect_n_frame;
        prev_wf = wf;
        prev_pos = [pos(1:2), window_sz];
    end
end
if(frame_idx == 2)
    max_res = max(response(:));
    if(max_res < 0.35)
        redetection = -1;
    end
else
    max_res = (1 - cf_params.gamma)*max_res + cf_params.gamma*max(response(:));
end

% redetection success
if(redetect_success)
    
    redetection = 0;
    
    %find the target position
    [vert_delta, horiz_delta] = find_peak(permute(red_response, [1,2,4,3]), feat_size, prev_pos(3:4));
    
    %exception
    if(isnan(vert_delta) || isnan(horiz_delta))
        vert_delta = 0; horiz_delta = 0;
    end
    
    wf = prev_wf;
    pos = prev_pos(1:2) - round([vert_delta, horiz_delta]);
    window_sz = round(prev_pos(3:4));
    
else
    
    %find the target position
    scale_delta = find(max(max(response,[],1),[],2) == max(response(:)),1);
    [vert_delta, horiz_delta] = find_peak(response(:,:,2), feat_size, window_sz);
    
    %exception
    if(isnan(vert_delta) || isnan(horiz_delta))
        vert_delta = 0; horiz_delta = 0; scale_delta = 2;
    elseif(length(scale_delta) > 1)
        scale_delta = 2;
    end
    
    if(init_window_sz/window_sz > 10 || init_window_sz/window_sz < 1/10)
        scale_delta = 2;
    end
    
    pos = pos - round([vert_delta, horiz_delta]);
    window_sz = round(window_sz * scale_ratio^(scale_delta-2));
    
end

% time check
timeh = toc;

% stack the position & target size
target_sz = round(window_sz / roi_resize_factor);

frame_idx = frame_idx + 1;



