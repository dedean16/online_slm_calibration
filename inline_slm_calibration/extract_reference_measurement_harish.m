% This analysis script is based on the measurement & analysis script virtual_TPM\SLM_generate_LUT\SLM_generate_LUT.m
% Instead of measuring it loads the data and performs the analysis
clear; close all; clc

%% Save settings
save_filepath = "C:\LocalData\slm_reference_phase_response.mat";
do_plot = true;

%% Load data
fprintf('\nLoading data... ')
load("\\ad.utwente.nl\TNW\BMPI\Data\Harish Sasikumar\Data\experimental_data\2021_10_Michelson(SLMCamera)+HBPC\2021_10_14_SLM_Camera\Without Software LUT loaded\linear\SLM_Camera 2021_10_12 16-13-04.mat")
fprintf('Done\n')

% Use first frame of measured data as missing measurement
frame = frames(:, :, 1, 1);

%% Experiment parameters
gray_values = (0:255)';    % gray values tested     
n_exp = 3;              % number of times experiment is performed
pause_time = 0.1;       % waiting time between every measurement (in sec)
excluded_from_fft = 5;  % first 'excluded_from_fft' sample is excluded in finding FFT maximum

frame_top = frame(1:end/4,:);
frame_bottom = frame(end*3/4+1:end, :);

% find spatial frequencies of fringes in the bottom and top part
frame_top_fft = fft2(frame_top-mean2(frame_top));
frame_bottom_fft = fft2(frame_bottom-mean2(frame_bottom));
max_freq_top = find_max_index(frame_top_fft, excluded_from_fft);
max_freq_bottom = find_max_index(frame_bottom_fft, excluded_from_fft);

% initialization
phase_response = zeros(length(gray_values),n_exp);
modulation_depth = zeros(length(gray_values),n_exp);


%% Analysis
for exp_i = 1:n_exp
    disp(['starting the analysis of experiment: ',num2str(exp_i),'/',num2str(n_exp)])
    for gray_i = 1:length(gray_values)
        frame=frames(:,:,gray_i,exp_i);

        frame_top = frame(1:end*1/4, :);
        frame_bottom = frame(end*3/4+1:end, :);
     
        % 2D Fourier transform bottom and top part of the frame
        frame_top_fft = fft2(frame_top);
        frame_bottom_fft = fft2(frame_bottom);
     
        % Compute phase difference between top and bottom
        phase_response(gray_i,exp_i) = angle(frame_top_fft(max_freq_top)) - angle(frame_bottom_fft(max_freq_bottom));
        modulation_depth(gray_i,exp_i) = min(abs(frame_top_fft(max_freq_top)),abs(frame_bottom_fft(max_freq_bottom)));
    end
    phase_response(:,exp_i) = unwrap(phase_response(:,exp_i) - phase_response(1,exp_i));
end

% compute mean and standard deviation of phase response
if phase_response(end) < 0 % make sure that phase is always increasing
    phase_mean = -mean(phase_response,2);
else
    phase_mean = mean(phase_response,2);
end
phase_std = std(phase_response,0,2);


%% retrieve field modulated field
E_s = mean(modulation_depth,2).*exp(1.0i*phase_mean);
E_s = E_s/max(abs(E_s(:)));


%% Save data
save(save_filepath, '-v7.3', "gray_values", "phase_mean", "phase_std", "phase_response", ...
    "modulation_depth", "E_s")

if do_plot
    %% calculate phase corresponding to 2pi (using linear fit)
    f = fit(gray_values,phase_mean,'a*x+b','Start',[0,0]);
    alpha = round((2*pi-f.b)/f.a);
    p_linear = f.a*gray_values+f.b;
    p_error = mean(abs(phase_mean-p_linear));
    
    %% plot phase response
    figure(1); clf; hold on;
    
    % plot data with errorbars
    errorbar(gray_values,phase_mean,phase_std,'s'); hold on;
    
    % plot linear fitted response
    plot(gray_values,p_linear,'--k','LineWidth',2);
    
    % figure layout
    legend('Measured','linear fit','Location','SouthEast');
    xlabel('Gray value');
    ylabel('Measured phase shift (rad)');
    title('SLM phase response');
    set(gca,'FontSize',14);
    xlim([gray_values(1), gray_values(end)]);
    
    % plot modulated field in complex plane
    figure(2); clf;
    plot(E_s,'-*b','LineWidth',2); hold on;
    plot(real(E_s(1)),imag(E_s(1)),'*r',real(E_s(end)),imag(E_s(end)),'*k','LineWidth',2);
    xlabel('Re\{E\}'); ylabel('Im\{E\}');
    title('Modulated field');
    set(gca,'FontSize',14);
    xlim([-1.1,1.1]); ylim([-1.1,1.1])
    axis square; grid on;
end


function index = find_max_index(fft_field, exclude_region)
%This function returns the index of the element in fft_field that has the
%largest absolute value. Points that are closer than 'exclude_region' to 
%the origin will be excluded.
    a = abs(fft_field);
    a(1:exclude_region, 1:exclude_region) = 0; %too close to zero order: remove
    a(1:exclude_region, (end-exclude_region):end) = 0;
    a((end-exclude_region):end, 1:exclude_region) = 0;
    a((end-exclude_region):end, (end-exclude_region):end) = 0;
    [~, index] = max(a(:));
end
