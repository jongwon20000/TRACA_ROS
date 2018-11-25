% function trk = tracker_setTarget(trk, im, bbox, opt)

% [img_files, pos, target_sz, ground_truth, video_path]

window_sz = round(target_sz.*roi_resize_factor);
init_window_sz = window_sz;

% finetune augmentation
patch = get_subwindow(im, pos, window_sz);
patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 0.5));
patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 1.0));
patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 1.5));
patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 2.0));
patch = cat(4, patch, patch(:, end:-1:1, :, 1));
patch = cat(4, patch, patch(end:-1:1, :, :, 1));

patch = imresize(patch, image_size);


% dAE finetuning
mbatch = single(patch);

if(gpus > 0)
    mbatch = gpuArray(mbatch);
end

mbatch = bsxfun(@minus, mbatch, full_vggnet.meta.normalization.averageImage);

vgg_res = vl_simplenn(full_vggnet, mbatch);

% multi-daenet selection
sel_res = vl_simplenn(prior_net, vgg_res(end).x);
[~, dae_idx] = max(sel_res(end).x(:,:,:,1));
%     dae_idx = 3;

daenet = multi_daenet{dae_idx}{1};
if(gpus > 0)
    for jj = 1:size(daenet,1)
        daenet(jj,1) = vl_simplenn_move(daenet(jj,1), 'gpu');
        daenet(jj,2) = vl_simplenn_move(daenet(jj,2), 'gpu');
    end
end
dae_res = cell(size(daenet,1), 2);

mfeat = vgg_res(target_layers+1).x;

l2_error_stack = zeros(1, finetune_epoch);
orth_error_stack = zeros(1, finetune_epoch);
for epoch_idx = 1:finetune_epoch
    
    % multi-stage forward
    for jj = 1:size(daenet,1)
        dae_res{jj,1} = vl_simplenn(daenet(jj,1), mfeat, [], dae_res{jj,1});
        dae_res{jj,2} = vl_simplenn(daenet(jj,2), dae_res{jj,1}(end).x, [], dae_res{jj,2});
    end
    
    % multi-stage backward
    [dae_res, w] = multi_stage_backward_finetune(daenet, dae_res, mfeat, mfeat, orth_lambda, cf_params);
    
    % multi-stage gradient update
    % encoder update
    for jj = 1:size(daenet,1)
        for kk = 1:size(dae_res{jj,1}, 2)
            if(~isempty(dae_res{jj,1}(kk).dzdw))
                daenet(end,1).layers{kk}.weights{1} = ...
                    daenet(end,1).layers{kk}.weights{1} - learning_rate*dae_res{jj,1}(kk).dzdw{1};
                daenet(end,1).layers{kk}.weights{2} = ...
                    daenet(end,1).layers{kk}.weights{2} - learning_rate*dae_res{jj,1}(kk).dzdw{2};
            end
        end
        for kk = 1:size(daenet(jj,1).layers, 2)
            if(~isempty(daenet(jj,1).layers{kk}.weights))
                daenet(jj,1).layers{kk}.weights =  daenet(end,1).layers{kk}.weights;
            end
        end
    end
    
    % decoder update
    for jj = 1:size(daenet,1)
        for kk = 1:size(dae_res{jj,2}, 2)
            if(~isempty(dae_res{jj,2}(end-kk+1).dzdw))
                daenet(end,2).layers{end-kk+2}.weights{1} = ...
                    daenet(end,2).layers{end-kk+2}.weights{1} - learning_rate*dae_res{jj,2}(end-kk+1).dzdw{1};
                daenet(end,2).layers{end-kk+2}.weights{2} = ...
                    daenet(end,2).layers{end-kk+2}.weights{2} - learning_rate*dae_res{jj,2}(end-kk+1).dzdw{2};
            end
        end
        for kk = 1:size(daenet(jj,2).layers, 2)
            if(~isempty(daenet(jj,2).layers{end-kk+1}.weights))
                daenet(jj,2).layers{end-kk+1}.weights =  daenet(end,2).layers{end-kk+1}.weights;
            end
        end
        
    end
    
    % loss estimation
    l2_err = 0;
    orth_err = 0;
    for jj = 1:size(daenet,1)
        % 2norm
        l2_err = l2_err + mean((dae_res{jj,2}(end).x(:) - mfeat(:)).*(dae_res{jj,2}(end).x(:) - mfeat(:)));
        % orth loss
        if(orth_lambda > 0)
            ww = bsxfun(@times, sum(bsxfun(@times, permute(w{jj}, [1,2,4,3,5]), w{jj}),1), ...
                permute(1-eye(size(w{jj},3), size(w{jj},3)), [3,4,1,2]));
            wi2 = pagefun(@mtimes, permute(w{jj}, [2,1,3,4,5]), w{jj}) + epsilon;
            wk2 = permute(wi2, [1,2,4,3,5]);
            l2w = mean(vec(bsxfun(@times, bsxfun(@times, ww.*ww, 1./wi2),1./wk2)));
        else
            l2w = 0;
        end
        orth_err = orth_err + l2w;
    end
    epoch_l2_err = l2_err;
    epoch_orth_err = orth_err;
    
    % Loss visualization
    l2_error_stack(epoch_idx) = gather(epoch_l2_err);
    orth_error_stack(epoch_idx) = gather(epoch_orth_err);
    if(opt.visualization > 0)
        figure(1000);
        subplot(1,3,1); plot(l2_error_stack(1:epoch_idx)); title('l2 loss');
        subplot(1,3,2); plot(orth_error_stack(1:epoch_idx)); title('orth loss');
        subplot(1,3,3); plot(l2_error_stack(1:epoch_idx) + orth_lambda*orth_error_stack(1:epoch_idx)); title('entire loss');
        drawnow;
    end
    
end


%% tracking sequence
encoding_number = length(reduced_dim);
% positions = zeros(size(ground_truth,1), 4);
% positions(1,1:2) = pos;
% positions(1,3:4) = target_sz;

% for first frame
mfeat_origin = mfeat(:,:,:,1);
if(encoding_number < 1)
    x = mfeat_origin;
else
    curr_res = vl_simplenn(daenet(encoding_number,1), mfeat_origin);
    x = curr_res(end).x;
end
ocp_ratio = sum(sum(bsxfun(@times, abs(x), cf_params.bbox_bmap), 1), 2) ./ (sum(sum(abs(x), 1), 2) + epsilon);

% layer selecting
sorted_ocp = sort(ocp_ratio, 'descend');
sel_layers = find(ocp_ratio >= sorted_ocp(val_min));
if(numel(sel_layers) > val_min)
    [~, sort_idx] = sort(ocp_ratio, 'descend');
    sel_layers = sort_idx(1:val_min);
end

xf = fft2(bsxfun(@times, cf_params.cos_window, x(:,:,sel_layers)));
wf = gather(bsxfun(@times, cf_params.yf2, xf ./ (xf.*conj(xf) + cf_params.lambda)));

% integrate the feature network
feat_net = vggnet;
for ii = 1:numel(daenet(encoding_number, 1).layers)
    feat_net.layers(end + 1) = daenet(encoding_number, 1).layers(ii);
end
feat_net.layers{end-1}.weights{1} = feat_net.layers{end-1}.weights{1}(:,:,:,sel_layers);
feat_net.layers{end-1}.weights{2} = feat_net.layers{end-1}.weights{2}(1,sel_layers);

redetection = 0;

feat_res = [];
frame_idx = 2;

