function Y = mse( views, d, k, r, max_iter, methods )
%MSE - Returns a multiview spectral embedding of the given data
%A MATLAB implementation of the Multiview Spectral Embedding method proposed by T. Xia, D. Tao, 
%T. Mei and Y. Zhang in "Multiview Spectral Embedding," in IEEE Transactions on Systems, 
%Man, and Cybernetics, Part B (Cybernetics), vol. 40, no. 6, pp. 1438-1446, Dec. 2010.
%
% Syntax:  [reduction] = mse(views,d,k,r,max_iter, methods)
%
% Inputs:
%    views - A cell array of NxM matrices
%    d - Final dimensionality
%    k - Neighborhood size
%    r - Regularization factor, r > 1
%    max_iter - Maximum number of iterations
%    methods - A cell array of methods for distance evaluation.
%	       All methods supported by pdist are allowed. 
%
% Outputs:
%    Y - Multiview embedding
%
% Example: 
%    mse({first_view, second_view},2,30,3,100,{'euclidean','jaccard'})
%
% Other m-files required: create_patch_alignment.m
%
% Author: Robert Ciszek 
% August 2014; Last revision: 22-July-2016

    if ~exist('methods', 'var')
        methods = repmat({'euclidean'},1,size(views,2));
    end

    n = size(views{1,1},1);
    m = size(views,2);
    L = create_patch_alignment(views, k, methods );
    alpha = ones(1,m) / m;
  
    Y = zeros(d,n);
    previous_Y = ones(d,n);
    iteration = 1;
      
    while previous_Y ~= Y
        
        disp(sprintf('Iteration: %i', iteration));
        iteration = iteration +1;        

        previous_Y = Y;
     
	%Calculate weighted L
	w_L =  sum(bsxfun(@times,L,reshape(alpha,[1 1 m])),3);
		        
	[V,D] = eig(w_L);
 
	f_w_L = w_L + eye(size(w_L))*(-2*min(eig(w_L)));
	[V,D] = eig(f_w_L);        
	e = diag(D);

        [B I] = sort(e);
        Y = transpose(V(:,I(1:d)));

        alpha = calculate_alpha( Y, L, m );
          
	disp(sprintf('Alpha: %f ',alpha ));
        if iteration >= max_iter
           break; 
        end
    end
    
    function alpha = calculate_alpha( Y, L, m )
        alpha = ones(1,m);
        den_a = 0;
        for m_i = 1:m
           den_a = den_a + (1/trace(Y*L(:,:,m_i)*transpose(Y)))^(1/(r-1)); 
        end
        for i=1:m 
          alpha(1,i) = ((1/trace(Y*L(:,:,i)*transpose(Y)))^(1/(r-1)))/den_a;
        end
    end

end

