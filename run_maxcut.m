% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   April 24th, 2020
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

load Ybus_39
%load Ybus_118

% Set up MAXCUT problem
% Run experiment
[c,A,lb,ub] = genMAXCUT(Ybus,3);
[A,b,c,K] = genTheta(Ybus); lb = b; ub = b;
[A2,b2,c2,K2,info] = ctc(c,A,lb,ub);

% Solve using MOSEK
[r,res] = mosekopt('minimize info',sedumi2mosek(A2,b2,c2,K2));
[x,y,rp,rd] = moseksol2sedumi(res, A2,b2,c2,K2);

% Recover 
tt = tic;
[L, invD, p] = recover(y, info);

% Accuracy
pinf = rp / (1+norm(b2)); % |Ax - b| / (1 + |b|)
dinf = rd / (1+norm(c2)); % [Ay - c]_+ / (1 + |c|)
cx = c2'*x; by = b2'*y;
gap = abs(cx-by)/(1+abs(cx)+abs(by));
