function  LN_est = SNL_eval(linear_est,snlx,snl)

delta_bin = snlx(2)-snlx(1);
num_bins = length(snlx);
index1 = zeros(1,length(linear_est));
index2 = zeros(1,length(linear_est));

index1 = floor((linear_est-min(snlx))/delta_bin) + 1;
index1 = ((index1 >= 1).*(index1 <= num_bins)).*index1 + (index1 < 1) + (index1 > num_bins)*num_bins;

index2 = index1+((index1>1).*(index1<num_bins));

LN_est = snl(index1) + ((linear_est - snlx(index1))/delta_bin).*(snl(index2)-snl(index1));

clear index1 index2
end