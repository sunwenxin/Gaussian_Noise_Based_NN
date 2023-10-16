%% first part: Stack original data
% s_p=3;
% Length=2e4;
% t_begin=20;
% s_f=2;
% U_p=stacked(U,s_p,Length,t_begin);
% Y_p=stacked(Y,s_p,Length,t_begin);
% U_f=stacked(U,s_f,Length,t_begin+s_p);
% Y_f=stacked(Y,s_f,Length,t_begin+s_p);
% 
% INN_u=[U_p;Y_p;U_f];
% INN_y=Y_f;

%% or second part: Stack faulty data

s_p=3;
Length=3e3;
t_begin=20;
s_f=2;
U_p_faulty=stacked(U_faulty,s_p,Length,t_begin);
Y_p_faulty=stacked(Y_faulty,s_p,Length,t_begin);
U_f_faulty=stacked(U_faulty,s_f,Length,t_begin+s_p);
Y_f_faulty=stacked(Y_faulty,s_f,Length,t_begin+s_p);

INN_u_faulty=[U_p_faulty;Y_p_faulty;U_f_faulty];
INN_y_faulty=Y_f_faulty;

