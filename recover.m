function [ L, invD, p] = recover(y, info )
%RECOVER Efficiently recover solution of chordal converted problem

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   August 8th, 2020
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

n = info.treeDecomp.n;
clique = info.treeDecomp.clique;
symbasis = info.symbasis;

% Generate submatrices
pnt = 0;
ell = numel(clique);
ii = cell(1,ell);
jj = cell(1,ell);
kk = cell(1,ell);
for v = 1:ell
    ord = numel(clique{v});
    ndof = size(symbasis{v},2);
    subval = symbasis{v}*y(pnt+(1:ndof));
    pnt = pnt + ndof;
    
    % where to put it
    idx = clique{v}; idx = idx(:);
    idx2 = bsxfun(@plus, idx, (idx-1)'*n);
    ii{v} = idx2(:)';
    jj{v} = v*ones(1,ord^2);
    kk{v} = subval(:).';
end
%assert(length(y) == pnt, 'mismatch in length of y. data is probably corrupt');
X = sparse([ii{:}],[jj{:}],[kk{:}],n^2,ell);

% Fuse together into a single matrix
Supp = sum(X~=0,2); 
X = sum(X,2);
X(Supp>0) = X(Supp>0) ./ Supp(Supp>0);
X = reshape(X,n,n);

% Solve matrix completion problem
p = info.perm;
[L,invD] = matComp(X, info.perm);

% Due to machine precision, invD can be slightly negative
invD = max(real(invD), 0);
end
    
function [L, invD] = matComp(S, p)
%MATCOMP Given sparse matrix S with Chordal sparsity pattern E, compute its
% maximum determinant matrix completion
%
%    inv(X) = maximize     logdet(Z)
%             subject to   Z(i,j) = S(i,j) 
%             for all      S(i,j) =/= 0.
%
% where X(p,p) = L*D*L'.

% Inputs: S - sparse matrix with chordal sparsity pattern
%         p - perfect elimination ordering for this sparsity pattern

% Ouputs: L,D - form X by X(p,p) = L*diag(D)*L'. Then inv(X) is the maximum
%               determinant completion of S.

n = size(S,1);
assert(size(S,2) == n, 'S must be square');
assert(issparse(S), 'S must be sparse');
assert(norm(S-S','fro') == 0, 'S must be perfectly symmetric');

S = S(p,p);

% Check for chordal sparsity
nr = symbfact(S);
if sum(nr) ~= (nnz(S)+n)/2
    error('S is not chordal!');
end

% Compute tree decomposition
[par, post] = etree(S);       % clique tree
ch = p2ch(par);
[row_ind, Sl] = compressCol(tril(S)); % cliques
mask = getmask(row_ind, par); 

% Andersen, Dahl, Vandenberghe -- Algorithm 4.2
L = cell(1,n);
invD = zeros(1,n);
V = cell(1,n); % Update matrices
for ind = n:-1:1 % reverse postorder = preorder
    j = post(ind);
    
    % j-th column of S
    % NB: S is lower-triangular, so Sj(1) = j by construction.
    Sj = Sl{j}; 
    Sjj = Sj(1); SIj = Sj(2:end);
    Jj = row_ind{j}; 

    % Run formula
    % [ Sjj, SIj' ] [ 1   ] = [ 1 / Dj ]
    % [ SIj, V{j} ] [ LIj ]   [ 0      ]
    SJJ = [Sjj, SIj'; SIj, V{j}]; 
    if numel(Jj) > 1 % Not a root node
        e1 = [1; zeros(numel(Jj)-1,1)];
        xvec = [e1, -SJJ(:,2:end)] \ SJJ(:,1);
        invD(j) = xvec(1);
        LIj = xvec(2:end);
    else
        invD(j) = Sjj;
        LIj = [];
    end

    % Store
    L{j} = [1; LIj];

    % Propagate to children
    V{j} = []; % Free this update matrix
    for i = ch{j}
        V{i} = SJJ(mask{i},mask{i});
    end
end
L = ccs2sparse(row_ind, L);
invD = sparse(invD(:));
end

function mask = getmask(row_ind, par)
% Precompute the masks such that 
%   M(mask{i}, mask{i}) = E_{JjIi}' * M * E_{JjIi} where j = par(i)
% and
%   E_{IJ}(i,j) = 1 if I(i) = J(j) and 0 otherwise,
%   Jj = row_ind{j}
%   Ii = row_ind{i}(2:end)
% 
n = numel(row_ind);
mask = cell(1,n);
for i = 1:n
    j = par(i);
    if j>0
        Jj = row_ind{j}; Ji = row_ind{i};
        mask{i} = ismember(Jj, Ji(2:end));
    end
end
end

function [row_ind, val] = compressCol(Mat)
% convert a sparse matrix into the compressed column storage format
row_ind = cell(1,size(Mat,2));
val = cell(1,size(Mat,2));
for j = 1:size(Mat,2)
    [row_ind{j}, ~, val{j}] = find(Mat(:,j));
end
end

function [Mat] = ccs2sparse(row_ind, val)
% Convert CCS back to sparse matrix
n = numel(row_ind);
col_ind = cell(1,n);
for j = 1:n
	col_ind{j} = j*ones(size(row_ind{j}));
end
row_ind = cat(1,row_ind{:});
col_ind = cat(1,col_ind{:});
val = cat(1,val{:});
Mat = sparse(row_ind,col_ind,val,n,n,numel(val));
end

function [ch, root] = p2ch(p)
% Convert the list of parent vectors into a cell array of children vectors.
n = length(p);
ch = cell(size(p));
root = [];
for i = 1:n
    par = p(i);
    if par > 0
        % Push child into parent
        ch{par} = [ch{par}, i];
    elseif par == 0
        root = [root, i];
    end
end
end
