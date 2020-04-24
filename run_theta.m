% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   April 24th, 2020
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

load Ybus_39
%load Ybus_118

% Set up MAXCUT problem
% Run experiment
[A,b,c,K] = genTheta(Ybus); lb = b; ub = b;
[A2,b2,c2,K2,info] = ctc(c,A,lb,ub);

% Solve SDP usign MOSEK
[r1,res1] = mosekopt('minimize info',sedumi2mosek(A,b,c,K));
[x1,y1,rp1,rd1] = moseksol2sedumi(res1, A,b,c,K);

% Solve CTC using MOSEK
[r2,res2] = mosekopt('minimize info',sedumi2mosek(A2,b2,c2,K2));
[x2,y2,rp2,rd2] = moseksol2sedumi(res2, A2,b2,c2,K2);

% Recover 
tt = tic;
[L, invD, p] = recover(y2, info);

% Accuracy
pinf = rp / (1+norm(b2)); % |Ax - b| / (1 + |b|)
dinf = rd / (1+norm(c2)); % [Ay - c]_+ / (1 + |c|)
cx = c2'*x; by = b2'*y;
gap = abs(cx-by)/(1+abs(cx)+abs(by));
