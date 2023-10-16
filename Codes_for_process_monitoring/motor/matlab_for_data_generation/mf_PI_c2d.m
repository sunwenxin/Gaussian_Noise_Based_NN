function [ disc_PI ] = mf_PI_c2d( cont_PI )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


Kp = cont_PI.K - (cont_PI.K*cont_PI.ts)/(2*cont_PI.Ti);
Ki = (cont_PI.K*cont_PI.ts)/cont_PI.Ti;
Kd = (cont_PI.K*cont_PI.Td)/cont_PI.ts;

P1 = Kp + Ki + Kd;
P2 = -(Kp+2*Kd);
P3 = Kd;

disc_PI.num = [P1, P2, P3];
disc_PI.den = [1, -1, 0];

% disc_PI.A = [0, 0; 1, 1];
% disc_PI.B = [P3; (P1+P2)];
% disc_PI.C = [0, 1];
% disc_PI.D = P1;

disc_PI.A = [0, 1; 0, 1];
disc_PI.B = [0; 1];
disc_PI.C = [P3, (P1+P2)];
disc_PI.D = P1;
disc_PI.n = size(disc_PI.A,1);

end

