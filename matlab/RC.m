 function [h1] = RC(X,Y,M)
 
offset = 100;
N = min(length(X),length(Y));

X = X(1:N);
Y = Y(1:N);

X = X - mean(X);
Y = Y - mean(Y);

num_pers = floor((N-M)/offset);


h1 = zeros(1,M);
fft_h1 = zeros(1,M);
cross_xy = fft_h1;
denom = cross_xy;

for k=1:num_pers

    x = X((k-1)*offset+1:(k-1)*offset+M);
    y = Y((k-1)*offset+1:(k-1)*offset+M);
    
    auto_x = abs(fft(x)).^2;
    auto_y = abs(fft(y)).^2;
    cross_xy = cross_xy + conj(fft(x)).*fft(y);
    denom = denom + (auto_x + mean(auto_y)*10);
    
end

fft_h1 = cross_xy./denom;

h1 = ifft(fft_h1);


end