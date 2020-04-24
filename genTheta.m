function [ A,b,c,K ] = genTheta( Adj )
%GENTHETA Given the graph, generate sparse version of the Theta problem

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   August 8th, 2018
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

% Get graph
n = size(Adj,1);
assert(size(Adj,2) == n);
assert(issparse(Adj));

% Make weights positive
Adj = abs(Adj)/2;
Adj = Adj + Adj';

% Get graph edges
[i,j,~] = find(tril(Adj,-1));
i = i(:).'; j = j(:).'; 
m = numel(i);

% Generate constraints
A = [sparse((n+1)^2,1,1,(n+1)^2,1,1), ...
     sparse([(i-1)*(n+1) + j; (j-1)*(n+1) + i], repmat(1:m,2,1), 1, (n+1)^2, m, 2*m)];
b = zeros(m+1,1); b(1) = 1;
c = [speye(n), ones(n,1); ones(1,n), 0]; c = c(:);
K = struct; K.s = n+1; K.l = 0; K.q = 0;
end

