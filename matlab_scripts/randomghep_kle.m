function [v, l] = randomghep_kle(Qx, M, omega, k, p, twopass, verbose, para)
%RANDOMGHEP_KLE eigenvalues and eigenvectors of generalized Hermitian eigenvalue problem (GHEP) 
%               M*Q*M*x = lambda*M*x using a randomized approach based on 
%               Saibaba et. al., Numer. Linear Algebra Appl., 2015, in review
%
%   this function solves randomized GHEP for Karhunen-Loeve expansion;
%   the Fredholm integral equation of the second kind using 
%   a Galerkin projection with piecewise linear basis function gives 
%
%                 M*Q*M*x = lambda*M*x
%
%   [v,l] = RANDOMGHEP_KLE(Qx, M, omega, k, p, twopass, verbose, para) 
%
%   INPUTS:
%   Qx is a function handle (operator) that returns N x 1 vector Q*x;  
%   M is a N x N (hopefully sparse) matrix; 
%   omega is a N x N iid random matrix 
%   k is the desired rank
%   p is the oversampling parameter
%   twopass is 0 for singlepass, ~0 for twopass
%   verbose ~= 0 prints out PreCholQR accuracy; see precholqr2.m
%
%   OUTPUTS:
%   v is k x N eigenvectors 
%   l is k x 1 eigenvalues 
%
%   Note that in Saibaba et. al., M was created from FEniCS (http://fenicsproject.org/)     
%   The code is tested in MATLAB R2014b
%
%   Jonghyun Harry Lee and Arvind Saibaba, 7/17/2015
    
    r     = k + p;
    y	  = 0*omega;		

    if para
        parfor i = 1:r
            y(:,i)  = Qx(M*omega(:,i));
        end
    else
        for i = 1:r
            y(:,i) = Qx(M*omega(:,i));
        end
    end
    
    [q, Bq, ~] = precholqr2(M, y, verbose, para);
    
    if twopass
        Aq  = 0*q;
        parfor i = 1:r
            Aq(:,i) = M*Qx(M*q(:,i));
        end  
        T = q'*Aq;	
    else
        % need yh for single pass
        yh = M*y; 
        oao = omega'*yh;
        qtbo = Bq'*omega;
        
        T = qtbo'\(qtbo'\oao')';
    end
    
    [v,l] = eig(T);
    [l,i] = sort(diag(l),'descend');
    v     = q*v(:,i);
    
    %Return first k
    l = l(1:r-p);
    v = v(:,1:r-p);
    
end 
