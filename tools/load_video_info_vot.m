function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info_vot(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
		suffix = video(end-1:end);  %remember the suffix
		video = video(1:end-2);  %remove it from the video name
	else
		suffix = '';
	end

	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth' '.txt'];
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f,%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok, try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f %f %f %f %f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
	
	%set initial position and size
    top_left = [min([ground_truth(1,2), ground_truth(1,4), ground_truth(1,6), ground_truth(1,8)]) ...
        , min([ground_truth(1,1), ground_truth(1,3), ground_truth(1,5), ground_truth(1,7)])];
    bottom_right = [max([ground_truth(1,2), ground_truth(1,4), ground_truth(1,6), ground_truth(1,8)]) ...
        , max([ground_truth(1,1), ground_truth(1,3), ground_truth(1,5), ground_truth(1,7)])];
	target_sz = bottom_right - top_left + 1;
	pos = round((top_left + bottom_right)/2);
	
    X = [min([ground_truth(:,1), ground_truth(:,3), ground_truth(:,5), ground_truth(:,7)], [], 2), ...
        max([ground_truth(:,1), ground_truth(:,3), ground_truth(:,5), ground_truth(:,7)], [], 2)];
    Y = [min([ground_truth(:,2), ground_truth(:,4), ground_truth(:,6), ground_truth(:,8)], [], 2), ...
        max([ground_truth(:,2), ground_truth(:,4), ground_truth(:,6), ground_truth(:,8)], [], 2)];
    
	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
% 		ground_truth = [Y(:,1), X(:,1)] + [Y(:,2)-Y(:,1), X(:,2)-X(:,1)] / 2;
%         ground_truth = [Y(:,1), X(:,1), Y(:,2), X(:,2)];
        ground_truth = [Y(:,1), X(:,1)] + [Y(:,2)-Y(:,1), X(:,2)-X(:,1)]/2;
        ground_truth = [ground_truth, [Y(:,2)-Y(:,1), X(:,2)-X(:,1)]];
	end
	
				
    %general case, just list all images
    img_files = dir([video_path '*.png']);
    if isempty(img_files),
        img_files = dir([video_path '*.jpg']);
        assert(~isempty(img_files), 'No image files to load.')
    end
    img_files = sort({img_files.name});
	
end

