clear all
close all
clc

m_time = 32;

m_steeltime = 3;
m_idletime = 2;
m_load = 0;
m_ts = 0.01;

% System Parameters
Ks = 36;
Ts = 0.0017;
R = 0.58;
Ti = 0.0313;
Tm = 0.063;
Ce = 0.133;
alpha = 0.006666666666;
beta = 0.125;



% PI controller ASR
global ASR_d
ASR_c.K = 9.29;
ASR_c.Ti = 0.087;
ASR_c.Td = 0;
ASR_c.ts = m_ts;

ASR_d = mf_PI_c2d(ASR_c);
ASR_d.Cn = [0, alpha];



% PI controller ACR
global ACR_d
ACR_c.K = 0.545;
ACR_c.Ti = 0.0313;
ACR_c.Td = 0;
ACR_c.ts = m_ts;

ACR_d = mf_PI_c2d(ACR_c);
ACR_d.Ci = [beta, 0];


n_mean1 = 0;
n_var1 = 500; %5
n_seed1 = floor(1000*rand(1));

n_mean2 = 0;
n_var2 = 100;%1
n_seed2 = floor(1000*rand(1));

n_mean3 = 0;
n_var3 = 0.1;%0.1
n_seed3 = floor(1000*rand(1));

n_mean4 = 0;
n_var4 = 0.1;%0.1;
n_seed4 = floor(1000*rand(1));

d_mean1 = 0;
d_var1 = 0.01;
d_seed1 = floor(1000*rand(1));


% sin_amp = 0.5;
% sin_bias = 0;
% sin_freq = 2*pi*0.5;
% sin_phase = 0;

sin_amp = 0;
sin_bias = 0;
sin_freq = 0;
sin_phase = 0;

m_load1s_steptime = 10;
m_load1s_ini = 0;
m_load1s_final = m_load;
m_load1e_steptime = m_load1s_steptime+m_steeltime;
m_load1e_ini = 0;
m_load1e_final = -m_load;

m_load2s_steptime = m_load1e_steptime+m_idletime;
m_load2s_ini = 0;
m_load2s_final = m_load;
m_load2e_steptime = m_load2s_steptime+m_steeltime;
m_load2e_ini = 0;
m_load2e_final = -m_load;

m_load3s_steptime = m_load2e_steptime+m_idletime;
m_load3s_ini = 0;
m_load3s_final = m_load;
m_load3e_steptime = m_load3s_steptime+m_steeltime;
m_load3e_ini = 0;
m_load3e_final = -m_load;

m_load4s_steptime = m_load3e_steptime+m_idletime;
m_load4s_ini = 0;
m_load4s_final = m_load;
m_load4e_steptime = m_load4s_steptime+m_steeltime;
m_load4e_ini = 0;
m_load4e_final = -m_load;

m_load5s_steptime = m_load4e_steptime+m_idletime;
m_load5s_ini = 0;
m_load5s_final = m_load;
m_load5e_steptime = m_load5s_steptime+m_steeltime;
m_load5e_ini = 0;
m_load5e_final = -m_load;

m_load6s_steptime = m_load5e_steptime+m_idletime;
m_load6s_ini = 0;
m_load6s_final = m_load;
m_load6e_steptime = m_load6s_steptime+m_steeltime;
m_load6e_ini = 0;
m_load6e_final = -m_load;

m_load7s_steptime = m_load6e_steptime+m_idletime;
m_load7s_ini = 0;
m_load7s_final = m_load;
m_load7e_steptime = m_load7s_steptime+m_steeltime;
m_load7e_ini = 0;
m_load7e_final = -m_load;

m_load8s_steptime = m_load7e_steptime+m_idletime;
m_load8s_ini = 0;
m_load8s_final = m_load;
m_load8e_steptime = m_load8s_steptime+m_steeltime;
m_load8e_ini = 0;
m_load8e_final = -m_load;

m_load9s_steptime = m_load8e_steptime+m_idletime;
m_load9s_ini = 0;
m_load9s_final = m_load;
m_load9e_steptime = m_load9s_steptime+m_steeltime;
m_load9e_ini = 0;
m_load9e_final = -m_load;

m_load10s_steptime = m_load9e_steptime+m_idletime;
m_load10s_ini = 0;
m_load10s_final = m_load;
m_load10e_steptime = m_load10s_steptime+m_steeltime;
m_load10e_ini = 0;
m_load10e_final = -m_load;


m_load11s_steptime = m_load10e_steptime+m_idletime;
m_load11s_ini = 0;
m_load11s_final = m_load;
m_load11e_steptime = m_load11s_steptime+m_steeltime;
m_load11e_ini = 0;
m_load11e_final = -m_load;

% Controller after modification
global Cont
Cont.A = [ASR_d.A zeros(size(ASR_d.A,1),size(ACR_d.A,2));ACR_d.B*ASR_d.C ACR_d.A];
Cont.B = [zeros(size(ASR_d.B,1),size(ACR_d.B,2)) ASR_d.B;ACR_d.B ACR_d.B*ASR_d.D];
Cont.C = [ACR_d.D*ASR_d.C ACR_d.C];
Cont.D = [ACR_d.D ACR_d.D*ASR_d.D];





