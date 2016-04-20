function [LN_est,linear_est,kernel,snlx,snl] = LN(stim,resp,kernel_length,num_bins)

l1 = length(stim);
l2 = length(resp);
N = min(l1,l2);

stim = stim(1:N);
resp = resp(1:N);

mean_stim = mean(stim);
stim = stim - mean_stim;
std_stim = std(stim);
% stim = stim/std(stim);


mean_resp = mean(resp);
resp = resp - mean_resp;

[kernel] = RC(stim,resp,kernel_length);

% kernel = kernel-mean(kernel(500:end));
kernel = kernel-mean(kernel(400:end));
% kernel = kernel-mean(kernel);
% linear_est = conv(stim,kernel(1:400));
linear_est = conv(stim,kernel);
linear_est = linear_est(1:N);


%kernel = kernel/abs(min(kernel(1:500)));
kernel = kernel/abs(std(linear_est)/std_stim*mean_stim);
kernel = kernel-mean(kernel(500:end));


linear_est = conv(stim,kernel);
linear_est = linear_est(1:N);


[snlx,snl,LN_est] = SNL(linear_est,resp,num_bins);

snl = snl + mean_resp;

end