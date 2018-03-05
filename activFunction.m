function activFunction = actF(z, func)
	if strcmp(func, 'relu') == 1
		activFunction = max(0,z);
	elseif strcmp(func, 'softmax') == 1
		activFunction = softmax(z);
	elseif strcmp(func, 'tanh') == 1
		activFunction = tanh(z);
	elseif strcmp(func, 'sigmoid') == 1
		activFunction = logsig(z);
	end
end
