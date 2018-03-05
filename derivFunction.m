function derivFunction = derF(z, func)
	if strcmp(func, 'relu') == 1
		derivFunction = double(z>0);
	elseif strcmp(func, 'softmax') == 1
		derivFunction = activFunction(z, func).* (1 - activFunction(z, func)); 
	elseif strcmp(func, 'tanh') == 1
		derivFunction = 1 - activFunction(z, func).^ 2;
	elseif strcmp(func, 'sigmoid') == 1
		derivFunction = activFunction(z,func).* (1 - activFunction(z,func));
	end
end
