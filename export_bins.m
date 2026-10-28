%% Export one session's binned spikes for umap_plots.py
% Bins every spike of the session, smooths each unit with a Gaussian along
% time, z-scores it, and saves the result with the bin times, the port each
% bin is at and its trial's reward, correctness, valuable patch and specific
% valuable patch labels to data/<session>/binned.mat -- the only file the
% Python side reads.
% Run from the project root.

data_file = 'reduced_np_data_17_05_2026.mat';
bin_size_ms = 100;
sigma_bins = 2;            % Gaussian width in bins

%% Load
[~, session] = fileparts(data_file);
raw = load(fullfile('data', 'raw', data_file));
trials = raw.np_data_reduced.trials;
units = raw.np_data_reduced.units;
tracking = raw.np_data_reduced.excel;
trial_durations = raw.np_data_reduced.metadata.SelectedTrialDurations;
clear raw

%% Bin spikes
session_length_sec = trials.TrialStartGlobalTime(end) + trial_durations(end);
n_bins = ceil(session_length_sec * 1000 / bin_size_ms);
bin_times_sec = ((1:n_bins) - 0.5) * bin_size_ms / 1000;   % bin centres

% the trial each bin falls in, NaN for bins in the gaps between trials
trial_starts = trials.TrialStartGlobalTime(:)';
trial_ends = trial_starts + trial_durations(:)';
trial_per_bin = discretize(bin_times_sec, [trial_starts, inf]);
has_trial = ~isnan(trial_per_bin);
past_end = false(size(trial_per_bin));
past_end(has_trial) = bin_times_sec(has_trial) > trial_ends(trial_per_bin(has_trial));
trial_per_bin(past_end) = NaN;
in_trial = ~isnan(trial_per_bin);

units = units(ismember(units.group, {'good', 'mua'}), :);
n_units = height(units);

% spike times are stored relative to each trial's start
spike_time = cell(height(trials), 1);
for t = 1:height(trials)
    spike_time{t} = trials.SpikeTime{t} + trials.TrialStartGlobalTime(t);
end
spike_time = vertcat(spike_time{:});
spike_identity = vertcat(trials.SpikeIdentity{:});

bin_idx = floor(spike_time * 1000 / bin_size_ms) + 1;
assert(all(bin_idx >= 1 & bin_idx <= n_bins), 'a spike fell outside the session');

% spikes from units dropped by the group filter get unit_idx 0 and are skipped
[~, unit_idx] = ismember(spike_identity, units.cluster_id);
keep = unit_idx > 0;
binned = accumarray([unit_idx(keep), bin_idx(keep)], 1, [n_units, n_bins]);

%% Smooth and z-score
radius = ceil(3 * sigma_bins);
kernel = exp(-(-radius:radius).^2 / (2 * sigma_bins^2));
kernel = kernel / sum(kernel);
smoothed = conv2(binned, kernel, 'same');

spikes = (smoothed - mean(smoothed, 2)) ./ std(smoothed, 0, 2);

% a unit with no variance comes out NaN, which UMAP's correlation metric can't take
usable = all(isfinite(spikes), 2);
fprintf('%s: %d units (%d dropped for no variance), %d bins of %d ms\n', ...
    session, sum(usable), sum(~usable), n_bins, bin_size_ms);
spikes = single(spikes(usable, :));

%% Head position per bin
% Video frames go onto the session clock like the spikes: trial start plus the
% frame's time in its trial. Frames past the last selected trial are dropped;
% an unmatched trial inside the range means the numberings disagree, so error.
[is_matched, trial_row] = ismember(tracking.trial_number, trials.SessionTrial);
unmatched = unique(tracking.trial_number(~is_matched));
assert(all(unmatched > max(trials.SessionTrial)), ...
    'frames reference unselected trials inside the session: %s', mat2str(unmatched'));

frames = tracking(is_matched, :);
frame_time = trials.TrialStartGlobalTime(trial_row(is_matched)) + frames.frame_time_in_trial;
assert(issorted(frame_time, 'strictascend'), 'frame times are not strictly increasing');

% badly tracked frames are dropped and the interpolation bridges the hole; no
% 'extrap', so a bin outside the tracked range is NaN rather than clamped
min_likelihood = 0.9;
tracked = frames.headstage_likelihood >= min_likelihood;
head_x = interp1(frame_time(tracked), frames.headstage_x(tracked), bin_times_sec, 'linear');
head_y = interp1(frame_time(tracked), frames.headstage_y(tracked), bin_times_sec, 'linear');

% gaps between trials have no frames, so interpolating across one would draw
% a straight line through places the mouse never was
head_x(~in_trial) = NaN;
head_y(~in_trial) = NaN;

%% Port per bin
% Port locations are recovered from behaviour: at reward time the mouse is at
% the port it chose, so each port's centre is the mean head position at reward
% over every trial at that port, rewarded or not. Per session, because each
% session's video puts the ports at different pixels.
n_ports = 8;
reward_time = trials.TrialStartGlobalTime + trials.RewardTimeInTrial;
reward_x = interp1(frame_time(tracked), frames.headstage_x(tracked), reward_time, 'linear');
reward_y = interp1(frame_time(tracked), frames.headstage_y(tracked), reward_time, 'linear');
reward_usable = ~isnan(reward_x) & ~isnan(reward_y);

port_positions = nan(n_ports, 2);
for port = 1:n_ports
    at_port = reward_usable & trials.Port == port;
    port_positions(port, :) = [mean(reward_x(at_port)), mean(reward_y(at_port))];
end

% nearest port within port_threshold_px of the bin's position, else 0. Port
% centres are 229+ px apart, so the threshold never confuses two ports; v5
% found 10 px keeps all but 16 of 410 trials' visits in the labels.
port_threshold_px = 10;
dx = head_x(:) - port_positions(:, 1)';
dy = head_y(:) - port_positions(:, 2)';
[nearest_px, port_per_bin] = min(sqrt(dx.^2 + dy.^2), [], 2);
port_per_bin(isnan(nearest_px) | nearest_px >= port_threshold_px) = 0;
port_per_bin = uint8(port_per_bin');
fprintf('%d of %d bins within %g px of a port\n', ...
    sum(port_per_bin > 0), n_bins, port_threshold_px);

%% Reward label per bin
% One code per trial, carried to every bin of that trial, 0 between trials:
%   1  rewarded, 30 ms
%   2  rewarded, 15 ms
%   3  unrewarded, right after a 15 ms trial at the same port -- licking on
%      after the port has depleted
%   4  unrewarded, anything else
reward_ms = round(trials.RewardDuration * 1000);
is_rewarded = trials.Rewarded == "rewarded";
assert(all(ismember(reward_ms(is_rewarded), [15, 30])), ...
    'rewarded trials with sizes other than 15 and 30 ms: %s', ...
    mat2str(unique(reward_ms(is_rewarded))'));

% the previous trial only counts if it is the session's actual previous trial,
% not one before a gap left by trial selection
prev_is_15_same_port = false(height(trials), 1);
prev_is_15_same_port(2:end) = diff(trials.SessionTrial) == 1 ...
    & is_rewarded(1:end-1) & reward_ms(1:end-1) == 15 ...
    & trials.Port(2:end) == trials.Port(1:end-1);

reward_per_trial = 4 * ones(height(trials), 1);
reward_per_trial(~is_rewarded & prev_is_15_same_port) = 3;
reward_per_trial(is_rewarded & reward_ms == 15) = 2;
reward_per_trial(is_rewarded & reward_ms == 30) = 1;
fprintf('reward trials: %d at 30 ms, %d at 15 ms, %d unrewarded after depletion, %d other unrewarded\n', ...
    histcounts(reward_per_trial, 0.5:1:4.5));

reward_per_bin = zeros(1, n_bins, 'uint8');
reward_per_bin(in_trial) = reward_per_trial(trial_per_bin(in_trial));

%% Correctness label per bin
% One code per trial, carried to every bin of that trial, 0 between trials:
%   1  rewarded, in the correct patch
%   2  unrewarded, in the correct patch
%   3  rewarded, in an incorrect patch
%   4  unrewarded, in an incorrect patch
% The correct patch is the block's rewarded pair: CorrectBlock 56 is ports 5
% and 6. Checked against the Correctness column, which should say the same.
is_correct = trials.Correctness == "correct";
block_patch = ceil(floor(trials.CorrectBlock / 10) / 2);
assert(isequal(is_correct, ceil(trials.Port / 2) == block_patch), ...
    'Correctness disagrees with the trial''s port being in CorrectBlock''s patch');

correct_per_trial = 2 * ~is_correct + 2 - is_rewarded;
fprintf('correctness trials: %d correct rewarded, %d correct unrewarded, %d incorrect rewarded, %d incorrect unrewarded\n', ...
    histcounts(correct_per_trial, 0.5:1:4.5));

correct_per_bin = zeros(1, n_bins, 'uint8');
correct_per_bin(in_trial) = correct_per_trial(trial_per_bin(in_trial));

%% Valuable patch label per bin
% One code per trial, carried to every bin of that trial, 0 between trials
% and for trials in none of the four groups. Rewarded or not never matters
% here, only for deciding when the patch was discovered:
%   1  pre discovery, not perseveration
%   2  perseveration: pre discovery, at the previous block's correct patch
%      (never the current block's own, even when the two share a patch)
%   3  post discovery, at the correct patch
%   4  post discovery, at an incorrect patch, with a later trial in the same
%      block back at the correct patch
% A block is a run of trials with the same CorrectBlock. Its patch counts as
% discovered at its third correct rewarded trial: that trial and every one
% before it in the block are pre, every later one is post.
block_per_trial = cumsum([true; diff(trials.CorrectBlock) ~= 0]);
port_patch = ceil(trials.Port / 2);
is_cr = is_correct & is_rewarded;

n_trials = height(trials);
cr_before = zeros(n_trials, 1);          % correct rewarded trials earlier in the block
prev_block_patch = nan(n_trials, 1);     % NaN through block 1, which has no previous block
returns_later = false(n_trials, 1);      % a later trial in the block is at the correct patch
for t = 2:n_trials
    if block_per_trial(t) == block_per_trial(t - 1)
        cr_before(t) = cr_before(t - 1) + is_cr(t - 1);
        prev_block_patch(t) = prev_block_patch(t - 1);
    else
        prev_block_patch(t) = block_patch(t - 1);
    end
end
for t = n_trials - 1:-1:1
    returns_later(t) = block_per_trial(t + 1) == block_per_trial(t) ...
        && (is_correct(t + 1) || returns_later(t + 1));
end

is_pre = cr_before < 3;
is_perseveration = is_pre & port_patch == prev_block_patch & port_patch ~= block_patch;

valuable_per_trial = zeros(n_trials, 1);
valuable_per_trial(is_pre & ~is_perseveration) = 1;
valuable_per_trial(is_perseveration) = 2;
valuable_per_trial(~is_pre & is_correct) = 3;
valuable_per_trial(~is_pre & ~is_correct & returns_later) = 4;
fprintf(['valuable patch trials: %d pre, %d perseveration, %d post correct, ' ...
    '%d post exploration with return, %d post exploration without return (unlabelled)\n'], ...
    histcounts(valuable_per_trial, 0.5:1:4.5), sum(valuable_per_trial == 0));

valuable_per_bin = zeros(1, n_bins, 'uint8');
valuable_per_bin(in_trial) = valuable_per_trial(trial_per_bin(in_trial));

%% Specific valuable patch label per bin
% One code per trial, carried to every bin of that trial, 0 between trials
% and for pre discovery trials that are not perseveration:
%   P      post discovery, in a block whose correct patch is P (1-4), wherever
%          the mouse is
%   4 + P  perseveration at patch P, the previous block's correct patch
specific_per_trial = zeros(n_trials, 1);
specific_per_trial(~is_pre) = block_patch(~is_pre);
specific_per_trial(is_perseveration) = 4 + prev_block_patch(is_perseveration);
fprintf('specific valuable patch trials, post by patch: %s, perseveration by patch: %s\n', ...
    mat2str(histcounts(specific_per_trial, 0.5:1:4.5)), ...
    mat2str(histcounts(specific_per_trial, 4.5:1:8.5)));

specific_per_bin = zeros(1, n_bins, 'uint8');
specific_per_bin(in_trial) = specific_per_trial(trial_per_bin(in_trial));

%% Save
out_dir = fullfile('data', session);
if ~isfolder(out_dir)
    mkdir(out_dir);
end
% -v7 rather than -v7.3 so scipy.io.loadmat can read it
save(fullfile(out_dir, 'binned.mat'), 'spikes', 'bin_times_sec', 'port_per_bin', 'reward_per_bin', 'correct_per_bin', 'valuable_per_bin', 'specific_per_bin', ...
    'bin_size_ms', 'sigma_bins', '-v7');
