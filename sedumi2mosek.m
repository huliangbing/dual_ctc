function prob = sedumi2mosek(A, b, C, K)
% Convert SDP from SeDuMi format to MOSEK format
% Calling sequence 
%    [r,res] = mosekopt('minimize info',sedumi2mosek(A, b, C, K));
%    [x,y,rp,rd] = moseksol2sedumi(res, A, b, C, K); 
% is equivalent to
%    [x,y] = sedumi(A,b,C,K);

% Author: Adapted by Richard Y Zhang <ryz@illinois.edu>
%         from YALMIP's yalmip2SDPmosek by Johan Löfberg
% Date:   July 28th, 2018
% Reference: R.Y. Zhang, J. Lavaei, "Sparse Semidefinite Programs with 
%            Guaranteed Near-Linear Time Complexity via Dualized Clique 
%            Tree Conversion", https://arxiv.org/abs/1710.03475
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.


if ~isfield(K,'f'), K.f = 0; end

prob.a = A(1:(K.f+K.l+sum(K.q)),:)';
prob.c = C(1:(K.f+K.l+sum(K.q)));

prob.blx = [-inf(K.f,1);zeros(K.l,1);-inf(sum(K.q),1)];
prob.bux = [inf(K.f+K.l+sum(K.q),1)];

prob.bardim = K.s;
prob.blc = b;
prob.buc = b;

top = 1+K.f+K.l+sum(K.q);
prob.barc.subj = [];
prob.barc.subk = [];
prob.barc.subl = [];
prob.barc.val = [];
prob.bara.subi = [];
prob.bara.subj = [];
prob.bara.subk = [];
prob.bara.subl = [];
prob.bara.val = [];

tops = [1];
for j = 1:length(K.s)
    n = K.s(j);
    tops = [tops tops(end)+n^2];
end
[ii,jj,kk] = find(A(top:top + sum(K.s.^2)-1,:));
cols = zeros(length(ii),1);
rows = zeros(length(ii),1);
allcol = [];
allrow = [];
allcon = [];
allvar = [];
allval = [];
for j = 1:length(K.s)    
    ind = find(ii>=tops(j) & ii<=tops(j+1)-1);
    iilocal = ii(ind)-tops(j)+1;
    col = ceil(iilocal/K.s(j));
    row = iilocal - (col-1)*K.s(j);
    allcol = [allcol col(:)'];
    allrow = [allrow row(:)'];
    allvar = [allvar jj(ind(:))'];
    allval = [allval kk(ind(:))'];
    allcon = [allcon repmat(j,1,length(col))];
end
keep = find(allrow >= allcol);
allcol = allcol(keep);
allrow = allrow(keep);
allcon = allcon(keep);
allvar = allvar(keep);
allval = allval(keep);
%allvar = jj(keep);
%allval = kk(keep);
prob.bara.subi = [prob.bara.subi allvar];
prob.bara.subj = [prob.bara.subj allcon];
prob.bara.subk = [prob.bara.subk allrow];
prob.bara.subl = [prob.bara.subl allcol];
prob.bara.val = [prob.bara.val allval];

for j = 1:length(K.s)
    n = K.s(j);
    Ci = C(top:top+n^2-1);
    Ci = tril(reshape(Ci,n,n));
    [k,l,val] = find(Ci);
    prob.barc.subj = [prob.barc.subj j*ones(1,length(k))];
    prob.barc.subk = [prob.barc.subk k(:)'];
    prob.barc.subl = [prob.barc.subl l(:)'];
    prob.barc.val = [prob.barc.val val(:)'];     
    top = top + n^2;
end

if K.q(1)>0    
    prob.cones.type   = [repmat(0,1,length(K.q))];
    top = 1 + K.f + K.l;
    prob.cones.sub = [];
    prob.cones.subptr = [];
    for i = 1:length(K.q)
        prob.cones.subptr = [ prob.cones.subptr 1+length(prob.cones.sub)];
        prob.cones.sub = [prob.cones.sub top:top+K.q(i)-1];
        top = top + K.q(i);
    end
end
