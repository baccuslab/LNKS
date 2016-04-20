function [snlx,snl,LN_est] = SNL(linear_est,resp,num_bins)
l1 = length(linear_est);
l2 = length(resp);
N = min(l1,l2);
snl = zeros(num_bins,1);
snlx = zeros(num_bins,1);

linear_est = linear_est(1:N);
resp = resp(1:N);

bin_length = floor(N/num_bins);
last_bin = N - bin_length*(num_bins-1);

filter = ones(bin_length,1)/bin_length;

[lest_s,s_index] = sort(linear_est);
resp_s = resp(s_index);


pre_snlx = conv(lest_s,filter);
pre_snl = conv(resp_s,filter);

pre_snlx = pre_snlx(1:N);
pre_snl = pre_snl(1:N);

snlx(1:end-1) = pre_snlx(bin_length:bin_length:end-last_bin+1)';
snl(1:end-1) = pre_snl(bin_length:bin_length:end-last_bin+1)';

snlx(end) = mean(lest_s(end-last_bin+1:end));
snl(end) = mean(resp_s(end-last_bin+1:end));



snl = interp1(snlx,snl,linspace(min(snlx),max(snlx),num_bins));
snlx = linspace(min(snlx),max(snlx),num_bins);
LN_est = SNL_eval(linear_est,snlx,snl);

end