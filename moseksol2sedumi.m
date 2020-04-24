function [x,y,rp,rd] = moseksol2sedumi(res, At, b, c, K)
% Recover the sedumi style x and y variables from mosek
% Calling sequence 
%    [r,res] = mosekopt('minimize info',sedumi2mosek(A, b, C, K));
%    [x,y,rp,rd] = moseksol2sedumi(res, A, b, C, K); 
% is equivalent to
%    [x,y] = sedumi(A,b,C,K);

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   July 28th, 2018
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.


if ~isfield(K,'f'), K.f = 0; end

% recover barx
barx = res.sol.itr.barx;
Ks = K.s;
xsed = cell(numel(K.s),1);
top = 0;
for j = 1:numel(Ks)
    n = Ks(j);
    len = n*(n+1)/2;
    
    % Extract data
    mat = tril(ones(n));
    mat(mat>0) = barx(top+(1:len));
    mat = mat + tril(mat,-1)';
    
    % Save and move on
    xsed{j} = mat(:);
    top = top+len;
end
x  = [res.sol.itr.xx; cat(1,xsed{:})];
y  = res.sol.itr.y;

if nargout > 2
    % Compute residuals
    rp = norm(b-At'*x);
    rd = max([0;eigK(At*y-c,K)]);
end
end

function lab = eigK(x,K)
% lab = maxeigK(x,K)
%
% MAXEIGK  Computes the maximum eigenvalue of a vector x with respect to a 
%          self-dual homogenous cone K.
%
% See also sedumi, mat, vec, eyeK.

% New function by Michael C. Grant
% Copyright (C) 2013 Michael C. Grant.
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301, USA
%

% The existence of rsdpN is code for 'is this the internal format?'
if isfield(K,'f'), nf = K.f; else nf = 0; end
if isfield(K,'l'), nl = K.l; else nl = 0; end
if isfield(K,'q'), nq = length(K.q); else nq = 0; end
if isfield(K,'s'), ns = length(K.s); else ns = 0; end

% Free cone
lab = cell(4,1);
lab{1} = -inf(nf,1);
top = nf;

% linear cone
lab{2} = x(top+(1:nl));
top = top + nl;

% quadratic cone
if nq > 0
    labq = zeros(2,nq);
    tmp = sqrt(0.5);
    for i = 1:length(K.q)
        n = K.q(i);
        x0 = x(top+1);
        nrm = norm(x(top+2:top+n));
        labq(:,i) = tmp*[x0 + nrm; x0 - nrm];
        top = top + n;
    end
    lab{3} = labq(:);
end

% semidefinite cone
if ns > 0
    labs = cell(ns,1);
    for i = 1 : ns
        n = K.s(i); n2 = n*n;
        XX = x(top+(1:n2));
        XX = reshape(XX,n,n);
        XX = XX + XX';
        labs{i} = eig(XX)/2;
        top = top + n2;
    end
    lab{4} = cat(1,labs{:});
end

% Concatenate everything
lab = cat(1,lab{:});
end


