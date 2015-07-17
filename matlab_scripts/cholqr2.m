function [q,r] = cholqr2(A)	
%CHOLQR2 numerically stable version of CholQR
%
% see also PRECHOLQR2, CHOLQR
%
% Jonghyun Harry Lee and Arvind Saibaba, 7/17/2015

    [Ap, r1] = cholqr(A);
	[q, r2] = cholqr(Ap);

    r = r2*r1;

end
