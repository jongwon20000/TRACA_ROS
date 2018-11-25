% function trk = tracker_init(im_size, opt)

% external library load
run([MATCONV_PATH 'matlab/vl_setupnn.m']);
addpath('./tools');
addpath('./ae_train');
addpath('./cf_tracker');
addpath(genpath([PIOTR_PATH 'toolbox/']));

% parameters
target_layers = 7;

reduced_dim = [128, 64];

image_size = [224, 224];

feat_size = [26, 26];

orth_lambda = opt.orth_lambda;
finetune_epoch = opt.finetune_iter;
learning_rate = opt.finetune_rate;

val_min = opt.val_min;
val_lambda = opt.val_lambda;

roi_resize_factor = 2.5;
scale_ratio = opt.scale_ratio;


% cf parameters
output_sigma_factor = opt.output_sigma_factor;
lambda = opt.lambda;
gamma = opt.gamma;
epsilon = 0.00001;
fftw('planner', 'estimate');

% redetection parameters
redetect_n_frame = opt.redetect_n_frame;
redetect_eps = opt.redetect_eps;

gpus = 1;


% vgg network load
full_vggnet = load('./network/imagenet-vgg-m-2048');
full_vggnet.layers = full_vggnet.layers(1:(end-4)); % trimming the network
vggnet = full_vggnet;
vggnet.layers = vggnet.layers(1:max(target_layers)); % trimming the network

% pretrained dAE load
multi_daenet = load('./network/multi_daenet');
multi_daenet = multi_daenet.multi_dae;
prior_net = load('./network/prior_network');
prior_net = prior_net.prior_net;
prior_net.layers = prior_net.layers(1:(end-1));

% correlation filter initialization
output_sigma = sqrt(prod(ceil(feat_size/roi_resize_factor))) * output_sigma_factor;
bbox_bmap = zeros(feat_size);
bbox_bmap(ceil(feat_size(1)/2-feat_size(1)/2/roi_resize_factor):floor(feat_size(1)/2+feat_size(1)/2/roi_resize_factor) ...
    , ceil(feat_size(2)/2-feat_size(2)/2/roi_resize_factor):floor(feat_size(2)/2+feat_size(2)/2/roi_resize_factor)) = 1;
yv = single(zeros(feat_size(1), feat_size(1)));
yv_temp = gaussian_shaped_labels(output_sigma, feat_size);
yv_temp = yv_temp(1:end, 1);
for jj = 1:length(yv_temp)
    yv(:, jj) = single(circshift(yv_temp, [jj-1,0]));
end
yf = single(fft(vec(gaussian_shaped_labels(output_sigma, feat_size))));
yf2 = single(fft2(gaussian_shaped_labels(output_sigma, feat_size)));
cos_window = single(hann(feat_size(1)) * hann(feat_size(2))');
if(gpus > 0)
    yf = gpuArray(yf);
    bbox_bmap = gpuArray(bbox_bmap);
end
cf_params.yv = yv;
cf_params.yf = yf;
cf_params.yf2 = repmat(yf2, [1,1,val_min]);
cf_params.bbox_bmap = bbox_bmap;
cf_params.cos_window = cos_window;
cf_params.gamma = gamma;
cf_params.lambda = lambda;
cf_params.epsilon = epsilon;

if(gpus > 0)
    vggnet = vl_simplenn_move(vggnet, 'gpu');
    prior_net = vl_simplenn_move(prior_net, 'gpu');
    full_vggnet = vl_simplenn_move(full_vggnet, 'gpu');
end
