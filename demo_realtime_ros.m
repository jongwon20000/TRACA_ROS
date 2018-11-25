%% Overview
% This is a MATLAB script which allows to use TRACA in ROS.
% TRACA is a visual object tracking framework.

% This ROS package requires TRACA to be setup, see `TRACA_webcam.zip` and the readme.md
% on https://sites.google.com/site/jwchoivision/home/traca
% It also requires the MATLAB Robotics System Toolbox

% This script works as follows: it reads images from a sensor_msgs/Image topic,
% by default this is '/kinect2/hd/image_color_rect' (change below if needed).

% The target can be selected in two ways: 
% 1) Sending a sensor_msgs/RegionOfInterest message to the `init_tracking`
% topic. It allows to update the object of interest. However, be careful -
% it requires a re-initialization of the tracker which takes a few seconds.
% 2) By selecting from a MATLAB figure. This is disabled by default, set
% the `visualize_image_selection` ROS parameter to True if required. In
% this case, the target can only be set once!

% The output is twofold:
% 1) sensor_msgs/RegionOfInterest messages are published to the `/object_pos`
% topic. This contains the current position of the tracked object.
% 2) Visualization in a MATLAB figure. This can be disabled by setting the
% `visualize_output` ROS parameter to False.

% If you use this code, please cite the following paper:
% http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf
%
% @inproceedings{ChoiCVPR2018,
% author = {Jongwon Choi and Hyung Jin Chang and Tobias Fischer and Sangdoo Yun and Kyuewang Lee and Jiyeoup Jeong and Yiannis Demiris and Jin Young Choi},
% title = {{Context-aware Deep Feature Compression for High-speed Visual Tracking}},
% booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
% year = {2018},
% month = {June},
% pages = {479--488}
% }

% For ROS extension specific questions, please contact Tobias Fischer <t.fischer@imperial.ac.uk>
% For TRACA related questions, please contact Jongwon Choi <jwchoi.pil@gmail.com>

% This code is messy - sorry. Feel free to improve and contribute.

%%

%% ROS Setup
clear all;
close all;

MATCONV_PATH = '../matconvnet/';
PIOTR_PATH = './piotr_toolbox/';

if robotics.ros.internal.Global.isNodeActive == 0
    rosinit;
    disp('ROS Init done');
else
    disp('ROS Already initialized');
end

%% Tracker Initialization
% camera input
global im;
global init_roi_ros;
init_roi_ros = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cam = rossubscriber('/image');
global cam;
cam = rossubscriber('/kinect2/hd/image_color_rect');
% TODO: This should really be a service rather than a subscriber
roi_sub = rossubscriber('/init_tracking', 'sensor_msgs/RegionOfInterest', @init_callback);
roi_pub = rospublisher('/object_pos', 'sensor_msgs/RegionOfInterest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameter setting
opt.orth_lambda = 1000;
opt.finetune_iter = 10;
opt.finetune_rate = 0.000000001;

opt.scale_ratio = 1.2;
opt.val_min = 25;
opt.val_lambda = 50.0;

opt.output_sigma_factor = 0.05;
opt.lambda = 1.0;
opt.gamma = 0.025;

opt.redetect_n_frame = 50;
opt.redetect_eps = 0.7;
opt.redetect_gamma = 0.0025;

opt.visualization = 0;

% Tracker initialization
tracker_init;

try
    % watch out: if this is set to true, you can only set the target once!
    visualize_image_selection = rosparam("get", "visualize_image_selection");
catch
    visualize_image_selection = false;
end

try
    visualize_output = rosparam("get", "visualize_output");
catch
    visualize_output = true;
end

disp('Init done.');
disp('Publish ROI to ROS topic "/init_tracking".');
if visualize_image_selection
    disp('Alternatively, please select target bounding box by drawing a rectangle.');
    disp('WARNING: disable visualize_image_selection so that the bounding box of the target can be updated without restarting the script.');
end

%% Main tracking loop
% TODO: These loops are bad hacks - better get rid of them
while(1)
    [pos, target_sz] = get_roi(visualize_image_selection);
    init_roi_ros = [];
    % Target initialization for tracker
    fprintf("Initialize tracker\n");
    tic;
    tracker_setTarget;
    tElapsed = toc;
    fprintf("Initialization done after %.1fs\n\n", tElapsed);

    if visualize_output
        figure(2);
        H2 = uicontrol('Style', 'PushButton', ...
                            'String', 'Stop', ...
                            'Callback', 'delete(gcbf)');
    end

    %% Online Tracking
    % After target setting
    stime = [];
    while((visualize_output && ishandle(H2)) || (not(visualize_output)))
        % get image
        prev_im = im;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        im_msg = cam.LatestMessage;
        im = readImage(im_msg);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Tracking
        tracker_tracking();

        if visualize_output
            imshow(im);
            rectangle('Position', [round(pos(2) - target_sz(2)/2), round(pos(1) - target_sz(1)/2), target_sz(2), target_sz(1)],...
                'LineWidth', 5, 'EdgeColor', 'r');
            drawnow;
        end

        roi_msg = rosmessage(roi_pub);
        %TODO: Create Stamped ROI message
        %roi_msg.Header = im_msg.Header;
        roi_msg.XOffset = uint32(round(pos(2) - target_sz(2)/2));
        roi_msg.YOffset = uint32(round(pos(1) - target_sz(1)/2));
        roi_msg.Width = uint32(target_sz(2));
        roi_msg.Height = uint32(target_sz(1));
        send(roi_pub, roi_msg)

        for ii = 1:numel(stime)
            fprintf('\b');
        end
        stime = sprintf('running time: %.1f ms (%.1f Hz)', timeh*1000, 1/timeh);    
        fprintf(stime);
        pause(0.01);
        
        if(not(isempty(init_roi_ros)))
           break 
        end
    end

    if visualize_image_selection || (visualize_output && not(ishandle(H2)))
        close;
        break;
    end
    if visualize_output
        close;
    end
end

%% Cleanup
fprintf('\n\n');
rosshutdown;
fprintf('\nBye\n');

%% Callback to get ROI from ROS topic
function init_callback(~, roi_msg)
    global init_roi_ros
    fprintf('Received ROS bounding box\n\n');
    init_roi_ros = roi_msg;
end

%% Get ROI function - supports MATLAB figure and ROS topic
function [pos, target_sz] = get_roi(visualize_image_selection)
    global init_roi_ros
    global cam
    global im
    
    if visualize_image_selection
        %% Video Visualization
        % figure handling
        f = figure(1);
        H = uicontrol('Style', 'PushButton', ...
                            'String', 'Target Set', ...
                            'Callback', 'delete(gcbf)');
    end

    % Before target setting
    while((visualize_image_selection && ishandle(H) && isempty(init_roi_ros)) || ...
            (not(visualize_image_selection) && isempty(init_roi_ros)))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        im_msg = receive(cam,10);
        im = readImage(im_msg);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if visualize_image_selection
            imshow(im);
        end
        pause(0.01); %in seconds
    end

    if visualize_image_selection
        imshow(im);
    end
    
    %% Target Initialization
    % target bounding box input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(init_roi_ros)
        rect = getrect;
    else
        rect = single([init_roi_ros.XOffset, init_roi_ros.YOffset, init_roi_ros.Width, init_roi_ros.Height]);
    end

    pos = round([rect(2), rect(1)] + [rect(4), rect(3)]/2); % bbox center position
    target_sz = round([rect(4), rect(3)]); %bbox size

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if isequal(target_sz, [0,0])
        error('Do not select a single pixel, select a rectangle instead.')
    end
    
    if visualize_image_selection
        close;
    end
end
