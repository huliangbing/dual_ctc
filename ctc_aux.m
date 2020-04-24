function [Amat,bvec,cvec,Kcone,info] = ctc_aux( c, A, lb, ub)
%CTC_AUX Clique-tree conversion with auxillary variables.
% Conver the following problem into the following
%   min c'x s.t. lb <= A*x <= ub, x \in SDP(n)
% to
%   max -c2'*y1 s.t. [ AA ]*y1 + [ BB ]*y2 = [ b ]
%                    [ NN ]                = [ 0 ]
%                    y1 \in SDP(n1) x SDP(n2) x ... 
% Here y1 contain the split conic variables, while y2 contain the decoupled
% free variables.

% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   August 8th, 2020
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.



max_clique_size = 60; % Quit if clique is bigger than this
real_embedding  = true; 
verbose = 2;
tt = tic;

% Defaults
if size(A,2) > size(A,1), A = A.'; end
if nargin < 4 || isempty(ub)
    ub = lb;
elseif isempty(lb)
    lb = ub;
end
lb = lb(:); ub = ub(:); c = c(:);
assert(all(lb <= ub), 'All lower-bounds must be smaller than the upper-bounds!');
m = size(A,2);

%--------------------------------------------------------------------------
% Tree decomposition
%--------------------------------------------------------------------------
% Get adjacency pattern
Adj = any([A, c],2);
n = floor(sqrt(length(Adj)));
assert(length(Adj) == n^2, 'number of rows in A must be a perfect square');
Adj = reshape(any([A, c],2), n, n);

% Compute the tree decomposition
perm = amd(Adj);
[T] = treeDecomp(Adj, perm);
if max(T.nn) > max_clique_size
    error('Cliques are way too large!');
end
if verbose > 0
	fprintf('complete\n');
    fprintf(' Num. cliques: %d \n Max cliques size: %d\n', ...
        T.ell, T.omega);
end

%--------------------------------------------------------------------------
% Generate constraints
%--------------------------------------------------------------------------
if verbose > 0
	fprintf('Beginning constraint splitting....');
end

ell = T.ell; % Number of cliques
nn  = T.nn;  % Size of each clique

% Allocate objective
c2 = allocate(T, c(:));

% Allocate constraints and perform splitting
%    A(:,i)' * x == lb(i)
% if and only if there exists u1 such that
%    sum_j A2(i,j)' * xj + B2(i,j) * uj == lb2(i)
[A2,  B2] = deal(cell(m, 2));
[lb2, ub2] = deal(cell(m, 1));
blksiz_r1 = zeros(m,1);
for i = 1:m
    [Supp, Ai] = allocate(T, A(:,i));
    [Supp, Ai, Bi, ci] = split(T.parent, Supp, Ai);
    A2(i,:) = {Supp, Ai}; 
    B2(i,:) = {Supp, Bi};
    lb2{i} = ci * lb(i);
    ub2{i} = ci * ub(i);
    blksiz_r1(i) = numel(ci);
end
blksiz_c = nn.^2;

% Overlap constraints
doreal = isreal(A) && isreal(c);
[ R, blksiz_r2, V ] = overlaps( T, doreal);
P   = blkdiag(V{:});
blksiz_c2 = cellfun(@(X)size(X,2), V);

% Convert from blocks into sparse matrices
% A2 and B2 are transpose because we need to extract rows out of them, and
% it is much cheaper to index columns than rows.
obj = cell2sparse(c2, 1, blksiz_c)*P;
A2t = (cell2sparse(A2, blksiz_r1, blksiz_c)*P)';
B2t = cellblkdiag(B2, blksiz_r1, ell)';
R  = cell2sparse(R, blksiz_r2, blksiz_c2);
naux = size(B2t,1);

% Fuse the matrices and pad the data
A2t  = [A2t; B2t];
R    = [R,   sparse(size(R,1), naux)];
P    = [P,   sparse(size(P,1), naux)];
obj  = [obj, sparse(1, naux)];
lb = cat(1,lb2{:}); ub = cat(1,ub2{:}); 

if verbose > 0
	fprintf('complete\n');
end

%--------------------------------------------------------------------------
% Dualize and output
%  lb <= A2*P*y <= ub
%  |R * P *y|   <= 0 * x0
%  eig(-P*y)   <= 0
%--------------------------------------------------------------------------

% Assemble matrices
% Isolate equality constraints
do_eq = (ub-lb) < 1e-8; % Turn this into bound
do_lb = isfinite(lb) & ~do_eq;
do_ub = isfinite(ub) & ~do_eq;

% Apply diagonal preconditioning to AtA
D = diag(sparse(1./sqrt(sum(A2t.^2,1))));
ub = D*ub; lb = D*lb; A2t = A2t*D;

% LP cone
Amat_lp = [-A2t(:,do_lb)'; A2t(:,do_ub)'];
rhs_lp  = [-lb(do_lb); ub(do_ub)];
m_lp    = numel(rhs_lp);

% SOCP cone
m_over = size(R,1);
z_row   = sparse(1, size(A2t,1) );
Amat_eq = [A2t(:,do_eq)'; R];
rhs_eq  = [0.5*(lb(do_eq)+ ub(do_eq)); zeros(m_over,1)];
m_eq    = numel(rhs_eq);

Kcone   = struct;
Kcone.l = m_lp;
Kcone.q = 1 + m_eq;

% SDP cone
if doreal
    Amat_sdp = -P;
    Kcone.s = T.nn;
else
if real_embedding
    V2 = realembed(V);
    Amat_sdp = -blkdiag(V2{:});
    Amat_sdp = [Amat_sdp, sparse(size(Amat_sdp,1), naux)];
    Kcone.s = 2*T.nn;
else
    Amat_sdp = -P;
    Kcone.s = T.nn;
    Kcone.scomplex = 1:T.ell;
end
end
m_sdp = size(Amat_sdp,1);
rhs_sdp  = zeros(m_sdp,1);

% Put everything together
Amat = [Amat_lp; z_row; Amat_eq; Amat_sdp];
cvec = [rhs_lp;  0;     rhs_eq;  rhs_sdp];
bvec = -obj(:);

% Information needed for recovery
info.symbasis  = V;
info.treeDecomp = T;
info.perm = perm;
info.time = toc(tt);
end

function [V2] = realembed(V)
%Embedding the size-n hermitian PSD cone into the
%size-2n real symmetric PSD cone.
V2 = cell(size(V));
for j = 1:numel(V)
    V{j} = full(V{j});
    [n2, m] = size(V{j});
    V2{j} = zeros(4*n2, m);
    n = floor(sqrt(n2));
    for i = 1:m
        tmp = reshape(V{j}(:,i),n,n);
        tmp2 = [real(tmp), -imag(tmp); imag(tmp), real(tmp)];
        V2{j}(:,i) = tmp2(:);
    end
    V2{j} = sparse(V2{j});
end
end


function [Supp,Mp] = allocate(T, M)
%ALLOCATE 

% PREPROCESS
M = reshape(M, T.n, T.n);
M = (M+M')/2;

% ALLOCATION
% Get the smallest support that covers the nonzero elements in the
% constraint.
% 1. Get support that covers A
% 2. Trim nodes using leaf-removal. Traverse bottom-up, and at each
%    each node, check if the unique elements of that node are in the
%    support. If yes, cover that node; if not, ignore it.
%    The unique elements are exactly M( cliques{Jk}, super{Jk} )
Supp = unique(T.isuper(any(M,1)));
nn = numel(Supp); 
cover = false(1,nn);
Mp = cell(1,nn);
for v = 1:numel(Supp) 
    Jk = Supp(v); % This tree node
    % If A(:,Uj) is nonzero
    if nnz(M(T.clique{Jk},T.super{Jk})) > 0 
        % Must include in cover
        constr = M(T.clique{Jk}, T.clique{Jk});
        Mp{v} = constr(:)';
        
        % Has been convered
        cover(v) = true;
        M(T.clique{Jk}, T.clique{Jk}) = 0;
    end
end
Mp = Mp(cover); 
Supp = Supp(cover);

if nargout == 1
    Supp = {Supp, Mp};
end

end

function [ sparsemat ] = cellblkdiag(cellmat, blksiz_r, ell)
%CELLBLKDIAG Block-diagonalize a cell matrix
% some of the blocks may be empty. If so, then we will pad that block with
% a block with zero columns and the right amount of rows. 

% Example:
%       cellblkdiag({A,[],B}, [5,4,3]) 
%         = blkdiag(sparse(A), sparse(4,0), sparse(B));

% Basic input checks
assert(iscell(cellmat), 'must give cells');
m = size(cellmat,1);
assert(m == numel(blksiz_r), 'cellcolumn and blksiz_r must agree in dimensions');

% compute shifts
rr = [0; cumsum(blksiz_r(:))];

% Process nonzero elements
[ii,jj,kk] = deal(cell(ell,1));
naux = zeros(ell,1);
for i = 1:m
    % Set-up
    j = cellmat{i,1}; Elem = cellmat{i,2};
    numblks = numel(j); 
    assert(numel(Elem) == numblks, 'Supp Elem mismatch');
    
    % Retrive elements
    for blk = 1:numblks
        if isempty(Elem{blk}), continue; end
        
            % Find nonzeros
            [this_ii, this_jj, this_kk] = find(Elem{blk});
            this_j = j(blk);
            added_naux = size(Elem{blk},2);
            
            % Append with the appropriate shifts
            ii{this_j} = [ii{this_j}; this_ii + rr(i)];
            jj{this_j} = [jj{this_j}; this_jj + naux(this_j)];
            kk{this_j} = [kk{this_j}; this_kk];
            
            % Add padding
            naux(this_j) = naux(this_j) + added_naux;
    end
end
sparsemat = cell(1,ell);
for j = 1:ell
    sparsemat{j} = sparse(ii{j}, jj{j}, kk{j}, rr(end), naux(j));
end
sparsemat = [sparsemat{:}];
end

function [ sparsemat ] = cell2sparse( cellmat, blksiz_r, blksiz_c)
%CELL2SPARSE
% cellmat is a two-column cell matrix. First cell column is the (column)
% support, second cell column are the elements.
% (compressed row format)
% WARNING: DOES NOT CHECK BLOCK SIZES!! MAKE SURE THESE ARE CORRECT

% Basic input check
assert(numel(blksiz_r) == size(cellmat,1), 'number of block-rows mismatch');

% Compute shifts
rr = [0; cumsum(blksiz_r(:))];
cc = [0; cumsum(blksiz_c(:))];

% Process nonzero elements
m = size(cellmat,1);
[ii,jj,kk] = deal(cell(m,1));
for i = 1:m
    % Set-up
    j = cellmat{i,1}; Elem = cellmat{i,2};
    numblks = numel(j); 
    assert(numel(Elem) == numblks, 'Supp Elem mismatch');
    
    % Retrive elements
    [ii_i,jj_i,kk_i] = deal(cell(1,numblks));
    for blk = 1:numblks
        if isempty(Elem{blk}), continue; end
            % Find nonzeros and shift them
            [ii_i{blk}, jj_i{blk}, kk_i{blk}] = find(Elem{blk});
            ii_i{blk} = ii_i{blk} + rr(i);
            jj_i{blk} = jj_i{blk} + cc(j(blk));
            % Force into columns
            ii_i{blk} = ii_i{blk}(:);
            jj_i{blk} = jj_i{blk}(:);
            kk_i{blk} = kk_i{blk}(:);
    end

    % Fuse them together and make sparse matrix
    ii{i} = cat(1,ii_i{:});
    jj{i} = cat(1,jj_i{:});
    kk{i} = cat(1,kk_i{:});
end
ii = cat(1,ii{:});
jj = cat(1,jj{:});
kk = cat(1,kk{:});
sparsemat = sparse(ii,jj,kk,rr(end), cc(end));
end

function [ R, blksiz_r, V] = overlaps( T, doreal)
%OVERLAPS 

% Count the number of roots in order to determine the rank of this
% projection
root = find(T.parent == 0);
nr = numel(root);
R = cell(T.ell-nr,2);

% Set up bases
V = cell(1, T.ell);
for j = 1:T.ell
    V{j} = sparse(symbasis(T.nn(j), doreal));
end

% List of edges
v = 1:T.ell; v(root) = [];
blksiz_r = zeros(T.ell-nr,1);
for i = 1:(T.ell-nr)
    vi = v(i); vpi = T.parent(vi);
    
    % Compute overlaps and basis
    [tmp, mj] = Rblk(T.clique{vpi}, T.clique{vi});
    Vo = symbasis(mj, doreal);
    
    % Form the matrix
    R{i,1} = [vi, vpi]; % Support
    R{i,2} = {Vo'*(tmp*V{vi}), ... Values
             -Vo'*(Rblk(T.clique{vi}, T.clique{vpi}) * V{vpi})};
    blksiz_r(i) = size(Vo,2);
end
end

function [mat, mj] = Rblk(I,J)
% Generate the matrix that implements the operator
%   R_IJ(X_JJ) = X_WW, W = I \cap J
    mask = find(ismember(J(:), I));
    nj = numel(J); mj = numel(mask);
    mask2 = bsxfun(@plus, mask, nj*(mask-1)');
    mat = sparse(mask2(:),1:mj^2,1,nj^2,mj^2)';
end

function chkCone(K)
nflqr = 0;
if isfield(K,'f'), nflqr = nflqr + K.f; end
if isfield(K,'l'), nflqr = nflqr + K.l; end
if isfield(K,'q'), nflqr = nflqr + K.q; end
if isfield(K,'r'), nflqr = nflqr + K.r; end
assert(nflqr == 0, 'f, l, q, r, cones not yet supported!');
assert(isfield(K,'s'), 'Must provide s cone');
assert(numel(K.s)==1, 'Must provide a only single s cone');
end
