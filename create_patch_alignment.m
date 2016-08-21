function alignment = create_patch_alignment( views, k, t, methods )
%CREATE_PATCH_ALIGNMENT - Creates 3D matrix composed of 2D neighborhood distance matrices.
%
% Syntax:  alignment = create_patch_alignment(views,k,methods)
%
% Inputs:
%    views - Cell array of feature matrices
%    k - Neighborhood size
%    t - Kernel parameter 
%    methods - Method for each view to be used in the distance calculations
%
% Outputs:
%    alignment - Patch alignment
%
% Example: 
%    create_patch_alignment({view1,view2},30,{'euclidean','euclidean']);
%
% Author: Robert Ciszek
% email: ciszek@uef.fi
% August 2014; Last revision: 21-August-2016

    if ~exist('methods', 'var')
	methods = repmat({'euclidean'},1,size(views,2));
    end 

    n = size(views{1,1},1);
    v = size(views,2);
    L = zeros(n,n,v);
    W = zeros(n,n,v); 
    D = zeros(n,n,v); 
    
    for v_i=1:v
	distances = squareform(pdist(views{1,v_i},char(methods(v_i)))).^2;
        distances(isinf(distances)) = 0;
        distances(isnan(distances)) = 0;
        distances  = exp(-distances/t);

	[sorted indexes] = sort(distances,2,'descend');
	for p=1:n
		W(p,indexes(p,2:2+k-1),v_i) = distances(p,indexes(p,2:2+k-1));
                W(indexes(p,2:2+k-1),p,v_i) = distances(p,indexes(p,2:2+k-1));		  
	end
        W(1+(n^2)*(v_i-1):n+1:v_i*(n^2)) = 0;
	D(1+(n^2)*(v_i-1):n+1:v_i*(n^2)) = sum(W(:,:,v_i),2);
        L(:,:,v_i) =  eye(n) - D(:,:,v_i)^(-1/2)*W(:,:,v_i)*D(:,:,v_i)^(-1/2);
    end       
    alignment = L; 
end
