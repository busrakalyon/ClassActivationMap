function generate_mcg_topk_batch(job_json_path)
% Generate only the ranked top-k MCG masks for a JSON batch job.
% MCG's im2mcg(..., mode, 0) returns ranked labels without materializing
% every proposal mask. Each completed item is written in a Python-readable
% column-major binary intermediate format.

job = jsondecode(fileread(job_json_path));
original_dir = pwd;
cleanup_dir = onCleanup(@() cd(original_dir)); %#ok<NASGU>
mcg_root_path = char(job.mcg_root);
addpath(mcg_root_path);
run(fullfile(mcg_root_path, 'install.m'));

items = job.items;
for item_index = 1:numel(items)
    item = items(item_index);
    output_dir = char(item.intermediate_dir);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    completed_path = fullfile(output_dir, 'completed.json');
    if exist(completed_path, 'file')
        fprintf('MCG_SKIP_COMPLETED %s\n', char(item.sample_id));
        continue;
    end

    try
        image = imread(char(item.image_path));
        if strcmp(char(job.mode), 'accurate') && size(image, 1) * size(image, 2) > 2.5e6
            error('EdgeCAM:MCGAccurateSizeLimit', ...
                ['Official im2mcg silently forces fast mode above 2.5 megapixels; ' ...
                 'refusing to mislabel this cache as accurate.']);
        end
        [proposals, ~, ~] = im2mcg(image, char(job.mode), 0);
        proposal_count = numel(proposals.labels);
        top_k = double(job.top_k);
        if proposal_count < top_k
            error('EdgeCAM:MCGTooFewProposals', ...
                'MCG produced %d proposals, but top_k=%d', proposal_count, top_k);
        end

        height = size(image, 1);
        width = size(image, 2);
        masks = false(height, width, top_k);
        for rank_index = 1:top_k
            masks(:, :, rank_index) = ismember( ...
                proposals.superpixels, proposals.labels{rank_index});
        end
        scores = double(proposals.scores(1:top_k));

        write_binary(fullfile(output_dir, 'masks_uint8_f_order.bin'), uint8(masks), 'uint8');
        write_binary(fullfile(output_dir, 'scores_float64_le.bin'), scores, 'double');

        metadata = struct();
        metadata.sample_id = char(item.sample_id);
        metadata.image_path = char(item.image_path);
        metadata.height = height;
        metadata.width = width;
        metadata.proposal_count_total = proposal_count;
        metadata.saved_top_k = top_k;
        metadata.mcg_mode = char(job.mode);
        metadata.matlab_version = version;
        metadata.mcg_root = char(job.mcg_root);
        write_json(fullfile(output_dir, 'metadata.json'), metadata);
        write_json(completed_path, struct('status', 'completed'));
        fprintf('MCG_COMPLETED %s %d\n', char(item.sample_id), proposal_count);
    catch exception
        error_payload = struct();
        error_payload.sample_id = char(item.sample_id);
        error_payload.identifier = exception.identifier;
        error_payload.message = exception.message;
        error_payload.report = getReport(exception, 'extended', 'hyperlinks', 'off');
        write_json(fullfile(output_dir, 'error.json'), error_payload);
        fprintf(2, 'MCG_FAILED %s %s\n', char(item.sample_id), exception.message);
    end
end
end


function write_binary(path, values, precision)
file_id = fopen(path, 'w', 'ieee-le');
if file_id < 0
    error('EdgeCAM:MCGWriteFailure', 'Cannot open %s for writing', path);
end
cleanup_file = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fwrite(file_id, values, precision);
end


function write_json(path, payload)
file_id = fopen(path, 'w');
if file_id < 0
    error('EdgeCAM:MCGWriteFailure', 'Cannot open %s for writing', path);
end
cleanup_file = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fwrite(file_id, jsonencode(payload), 'char');
end
