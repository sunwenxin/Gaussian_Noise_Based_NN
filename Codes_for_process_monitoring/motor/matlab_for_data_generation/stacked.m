function [ x_s ] = stacked(x_o,s,Length,t_begin)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
% x_o: original;
%x_s: stack;
%s: future (s) data
%Length: the number of stacked data
%t_begin: the begin of time

[N_r,N_c]=size(x_o);
if N_r<N_c
    x_o=x_o;
elseif N_r>=N_c
    %fprintf('Check the dimensions')
     x_o=(x_o)';
end
[N_r,N_c]=size(x_o);
x_s=[];
for i=t_begin:(t_begin+Length-1)
    x_stack=[];
    for j=i:(i+s-1)
        x_stack=[x_stack; x_o(:,j)];
    end
x_s=[x_s x_stack];
end
   
end

