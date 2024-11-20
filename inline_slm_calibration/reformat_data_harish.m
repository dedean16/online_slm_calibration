feedback = zeros(9, 256);

gv_col = [0 32 64 96 128 160 192 224 256];
gv_row = 0:255;

load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=0 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=0 .mat')
feedback(1, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=32 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=32 .mat')
feedback(2, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=64 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=64 .mat')
feedback(3, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=96 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=96 .mat')
feedback(4, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=128 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=128 .mat')
feedback(5, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=160 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=160 .mat')
feedback(6, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=192 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=192 .mat')
feedback(7, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=224 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=224 .mat')
feedback(8, :) = slm_phase.total_intensity;
load('/home/dani/LocalData/2023_08_inline_slm_calibration/ppl=512 zoom=10 gray1=256 2023_08_10 15-06-28/00_ppl=512 zoom=10 gray1=256 .mat')
feedback(9, :) = slm_phase.total_intensity;

save('-v7.3', '/home/dani/LocalData/2023_08_inline_slm_calibration/harish_feedback.mat', 'feedback', 'gv_col', 'gv_row');
