function costFunction = cst(input, output, target, nodeLayers, func)
	if strcmp(func, 'quad') == 1
		costFunction = 1/(2*size(input, 2)) * sum(sum((0.5*(target - output{length(nodeLayers)}).^2)));
	elseif strcmp(func, 'cross') == 1
		costFunction = - 1 / length(input) .* sum(sum(target .* log(output{length(nodeLayers)}) + ...
                  (1 - target) .* log(1 - output{length(nodeLayers)})));
	elseif strcmp(func, 'log') == 1
		costFunction = sum(-log(max(output{length(nodeLayers)}))/size(input, 2));
	end
end
