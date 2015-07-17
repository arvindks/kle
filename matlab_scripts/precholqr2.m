function [q, Aq, r] = precholqr2(A, z, verbose, para)
%PRECHOLQR2 
% See Bradley R Lowery and Julien Langou. Stability analysis of qr factorization in an oblique inner product. 
% arXiv preprint arXiv:1401.5171, 2014.

% Jonghyun Harry Lee and Arvind Saibaba 7/17/2015

    [q, r] = cholqr2(z);

    Aq = A*q;

%   genearally explicit parallelization of A*q is slower than multi-threading computation of A*q in MATLAB  
%     if para
%         fprintf('compute Mq with parfor');
%         tStart = tic;
%         Aq = 0*q;
%         parfor i = 1:size(q,2)
%             Aq(:,i) = A*q(:,i);
%         end
%         tElapsed = toc(tStart);    
%         fprintf('in precholqr2 : %f\n',tElapsed);
%         
%         fprintf('compute Mq again');
%         tStart = tic;
%         Aq = A*q;
%         tElapsed = toc(tStart);    
%         fprintf('in precholqr2 : %f\n',tElapsed);
%         
%     else
%         Aq = A*q;
%     end   
    
    T = q'*Aq;
    
    R = chol(T, 'upper');
    
    Aq = Aq/R;
    q = q/R;
    r = R*r;  
    
    if verbose
        I = eye(size(q,2));
        fprintf('||Z -QR|| is %g \n', norm(z - q*r))
        fprintf('||Q^TAQ -I || is %g \n', norm(q'*Aq - I))
        fprintf('||Q^TAY -R || is %g \n', norm(Aq'*z - r))
        fprintf('||YR^{-1} -Q || is %g\n', norm((r'\z')' - q))
        fprintf('||Q^TAA^{-1}AQ -I || is %g \n', norm(Aq'*(A\Aq) - I))
    end  
    
end
