function [q,r] = cholqr(Y)
%CHOLQR2 Compute Y = QR
%
% see also PRECHOLQR2, CHOLQR2
 
% Jonghyun Harry Lee and Arvind Saibaba, 7/17/2015

	rtr = Y'*Y;
	r   = chol(rtr, 'upper');
    q   = (r'\Y')';		
end
