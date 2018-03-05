function [ weights, biases, accuracy, cost ] = NeuralNetwork( inputs, targets, nodeLayers, numEpochs, batchSize, eta, trainSplit, testSplit, validSplit, savedNet, HidActivFunction, LastActivFunction, CostFunction, momentum, lambda, earlyStopEpochsPercent)
	
	% Training, testing and validating sets for the inputs and targets
	[train, valid, test] = dividerand(length(inputs),trainSplit, validSplit, testSplit);
	trainI = inputs(:, train);
	validI = inputs(:, valid);
	testI = inputs(:, test);
    
	trainT = targets(:, train);
	validT = targets(:, valid);
	testT = targets(:, test);
	
	% Check if we are loading a saved network
	if savedNet == 'No'
	
		% initialize weigths and biases
		weights = {};
		biases = {};
		
		% For each layer form the 2nd: Initialize weights and biases in the associated positions (normal distrib: avg = 0, sd = 1)
		for layer = 2 : length(nodeLayers)
			weights{layer} = normrnd(0, 1/sqrt(length(trainI)), nodeLayers(layer), nodeLayers(layer-1));
			biases{layer} = normrnd(0, 1, nodeLayers(layer), 1);
		end
		
	else
		% Retrieve node layers, weights and biases from the saved network
		nodeLayers = savedNet{0}
		weights = savedNet{1};
		biases = savedNet{2};
	end
  
	% Initialize cells for batches and targets and the batch counter
	batches = {};
	target_b = {};
	counter = 1;
	accuracy = {};
	cost = {};
	weightsD = {};
	biasesD = {};
	TrainCost = [];
	TestCost = [];
	ValidCost = [];
	
	% Check if there are enough instances for a whole batch
	for index = 1 : batchSize : length(trainI)
		%If True: 
		% Insert batch in batch cell
		% Insert target in target cell
		% Update the batch counter
		if batchSize > length(trainI) - index
			batches{counter} = trainI(:, index:end);
			target_b{counter} = trainT(:, index:end);
		%Else: 
		% Insert batch with the size of the remaining instances
		% Insert target in target cell
		else
			batches{counter} = trainI(:, index:index + batchSize -1);
			target_b{counter} = trainT(:, index:index+ batchSize-1);
			counter = counter+1;
		end 
	end 

	fprintf('\t|\t\t  TRAIN\t\t\t||\t\t\t  TEST\t\t\t||\t\t\t  VALIDATION\n');
	fprintf('---------------------------------------------------------------------------------------\n');
	fprintf('Ep\t|  Cost  |   Corr  |   Acc   ||   Cost |   Corr  |  Acc  ||  Cost  |  Corr  | Acc \n');
	fprintf('---------------------------------------------------------------------------------------\n');
	
	% For each batch of each epoch
	for epoch = 1 : numEpochs
		batch_c = 1;
		shuffle = randperm(length(batches));
		
		for batch = 1 : length(batches)
			% Initialize cells for activations, intermediate values and delta
			% First activation cell to the batch
			z = {};
			delta = {};
			activation = {};
			activation{1} = batches{shuffle(batch_c)};
			
			% For each layer (from the 2nd)
			% Calculate intermediate values
			% Calculate activation values and store them
			% ReLu in hidden layers
			% Softmax in last layer
			for layer = 2 : length(nodeLayers)
				z{layer} = bsxfun(@plus,(weights{layer} * activation{layer - 1}), biases{layer});
				if layer ~= length(nodeLayers)
					while (strcmp(HidActivFunction, 'relu') == 0 && strcmp(HidActivFunction, 'tanh') == 0 && strcmp(HidActivFunction, 'sigmoid') == 0)
						HidActivFunction = input('Activation function not available for HIDDEN LAYERS. Please write relu, tanh or sigmoid\n','s');
					end
					activation{layer} = activFunction(z{layer}, HidActivFunction);
				else
					while (strcmp(LastActivFunction, 'softmax') == 0 && strcmp(LastActivFunction, 'tanh') == 0 && strcmp(HidActivFunction, 'sigmoid') == 0)
						LastActivFunction = input('Activation function not available for OUTPUT LAYER. Please write softmax, tanh or sigmoid\n', 's');
					end
					activation{layer} = activFunction(z{layer}, LastActivFunction);
				end
			end
			
			% If we are using tanh we need to normalize values
			if strcmp(LastActivFunction, 'tanh') == 1 && size(targets, 1) > 1
				activation{length(nodeLayers)} = (activation{length(nodeLayers)} - min(min(activation{length(nodeLayers)}))) / (max(max(activation{length(nodeLayers)}))-min(min(activation{length(nodeLayers)})));
			end
			
			err = (activation{length(nodeLayers)} - target_b{shuffle(batch_c)});
			
			for layer = (length(nodeLayers)) : -1 : 2
				if layer == length(nodeLayers)
					% Last layer of backpropagation
					deriv = derivFunction(z{layer}, LastActivFunction);
					delta{layer} = err .* deriv;
				
				else
					delta{layer} = weights{layer+1}.' * delta{layer+1} .* derivFunction(z{layer}, HidActivFunction);
				end
			end
			
			% Momentum gradient descent with L2 regularization
			for layer = length(nodeLayers) : -1 : 2
				if epoch == 1 && batch == 1
					w = weights{layer} * (1-eta*lambda/length(batches{shuffle(batch_c)})) - eta/length(batches{shuffle(batch_c)}) * delta{layer} * activation{layer - 1}.';
					b = biases{layer} - eta / length(batches{shuffle(batch_c)}) * sum(delta{layer}, 2);
					weights{layer} = w;
					biases{layer} = b;
					weightsD{layer} = eta/length(batches{shuffle(batch_c)}) * delta{layer} * activation{layer - 1}.';
					biasesD{layer} = eta/length(batches{shuffle(batch_c)}) * sum(delta{layer}, 2);
                else
					weights{layer} = weights{layer} + weightsD{layer};
					biases{layer} = biases{layer} + biasesD{layer};
					weightsD{layer} = momentum .* weightsD{layer} * (1-eta*lambda/length(batches{shuffle(batch_c)})) - eta/length(batches{shuffle(batch_c)}) * delta{layer} * activation{layer - 1}.' ;
					biasesD{layer} = momentum .* biasesD{layer} - eta/length(batches{shuffle(batch_c)}) * sum(delta{layer}, 2);
				end
			end
			  
			batch_c = batch_c + 1;
		end
		
		% Final calculation of the outputs for the training, testing and validation datasets
		
		trainO = {};
		trainO{1} = trainI;
		testO = {};
		testO{1} = testI;
		validO = {};
		validO{1} = validI;
		
		for layer = 2 : length(nodeLayers)
			zTrain = bsxfun(@plus, (weights{layer} * trainO{layer-1}), biases{layer});
			zTest = bsxfun(@plus, (weights{layer} * testO{layer-1}), biases{layer});
			zValid = bsxfun(@plus, (weights{layer} * validO{layer-1}), biases{layer});
			
			if layer ~= length(nodeLayers)
				trainO{layer} = activFunction(zTrain, HidActivFunction);
				testO{layer} = activFunction(zTest, HidActivFunction);
				validO{layer} = activFunction(zValid, HidActivFunction);
			
			else
				trainO{layer} = activFunction(zTrain, LastActivFunction);
				testO{layer} = activFunction(zTest, LastActivFunction);
				validO{layer} = activFunction(zValid, LastActivFunction);
			end
		end
		
		% If not multiclass problem
		% Number or correct predictions and accuracy
		if size(targets, 1) == 1
			paTrain = trainT - round(trainO{length(nodeLayers)});
			paTest = testT - round(testO{length(nodeLayers)});
			paValid = validT - round(validO{length(nodeLayers)});

			trainCorrect = sum(paTrain(:)==0);
			testCorrect = sum(paTest(:)==0);
			validCorrect = sum(paValid(:)==0);
			
			trainAccu = trainCorrect / size(trainI,2);
			testAccu = testCorrect / size(testI,2);
			validAccu = validCorrect / size(validI,2);
		
		% If multiclass problem
		% Number or correct predictions and accuracy	
		else
			[maxTrT1, maxTrT2] = max(trainT);
			[maxTeT1, maxTeT2] = max(testT);
			[maxVaT1, maxVaT2] = max(validT);
			
			[maxTrO1, maxTrO2] = max(trainO{length(nodeLayers)});
			[maxTeO1, maxTeO2] = max(testO{length(nodeLayers)});
			[maxVaO1, maxVaO2] = max(validO{length(nodeLayers)});

			paTrain = maxTrT2 - maxTrO2;
			paTest = maxTeT2 - maxTeO2;
			paValid = maxVaT2 - maxVaO2;

			trainCorrect = sum(paTrain(:)==0);
			testCorrect = sum(paTest(:)==0);
			validCorrect = sum(paValid(:)==0);
			
			trainAccu = trainCorrect / length(trainI);
			testAccu = testCorrect / length(testI);
			validAccu = validCorrect / length(validI);	

		end
	
	
		% L2 regularization and costs

		while (strcmp(CostFunction, 'quad') == 0 && strcmp(CostFunction, 'cross') == 0 && strcmp(CostFunction, 'log') == 0)
			CostFunction = input('Cost function not available. Please write quad, cross or log\n', 's');
		end
		
		% L2 regularization
		totWeights = 0;
		for layer = 2: length(nodeLayers)
			totWeights = totWeights + sum(sum(weights{layer}.^2));
		end

		L2train = lambda / (2 * size(trainI,2)) * totWeights;
		L2test = lambda / (2 * size(testI,2)) * totWeights;
		L2valid = lambda / (2 * size(validI,2)) * totWeights;
		
		% Calculating costs
		TrainCost(epoch) = L2train + costFunction(trainI, trainO, trainT, nodeLayers, CostFunction);
		TestCost(epoch) = L2test + costFunction(testI, testO, testT, nodeLayers, CostFunction);
		ValidCost(epoch) = L2valid + costFunction(validI, validO, validT, nodeLayers, CostFunction);
		
		% Accuracies and costs
		accuracy{1}(epoch) = trainAccu;
		accuracy{2}(epoch) = testAccu;
		accuracy{3}(epoch) = validAccu;

		cost{1}(epoch) = TrainCost(epoch);
		cost{2}(epoch) = TestCost(epoch);
		cost{3}(epoch) = ValidCost(epoch);

		% If the number of epochs is lower than 100, print each iteration results
		if numEpochs <= 100
			fprintf('%d\t| %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f\n', ...
				epoch, TrainCost(epoch), trainCorrect, size(trainI, 2), trainAccu, ...
				TestCost(epoch), testCorrect, size(testI, 2), testAccu, ...
				ValidCost(epoch), validCorrect, size(validI, 2), validAccu);
            
		% Else print every 100 epochs.
		elseif mod(epoch, 100) == 0 && numEpochs > 100
			fprintf('%d\t| %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f\n', ...
				epoch, TrainCost(epoch), trainCorrect, size(trainI, 2), trainAccu, ...
				TestCost(epoch), testCorrect, size(testI, 2), testAccu, ...
				ValidCost(epoch), validCorrect, size(validI, 2), validAccu);		
		end
		
		% Check early stopping conditions
		if epoch >= round( earlyStopEpochsPercent/100 * numEpochs)
			if ValidCost(epoch) > ValidCost(epoch-1)
				if mod(numEpochs, 100) ~= 0 && numEpochs > 100
					fprintf('%d\t| %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f\n', ...
						epoch, TrainCost(epoch), trainCorrect, size(trainI, 2), trainAccu, ...
						TestCost(epoch), testCorrect, size(testI, 2), testAccu, ...
						ValidCost(epoch), validCorrect, size(validI, 2), validAccu);
				end
				fprintf('\nEarly Stopping: cost increased\n');
                subplot(1,2,1);
                plot(accuracy{1}) ; hold on; plot(accuracy{2}); hold on; plot(accuracy{3});
                title('Accuracy evolution per epoch'), ylabel('Accuracy'), xlabel('epoch');
                legend('Training', 'Test', 'Validation'); hold off;
                
                subplot(1,2,2);
                plot(cost{1}) ; hold on; plot(cost{2}); hold on; plot(cost{3});
                title('Cost evolution per epoch'), ylabel('Cost'), xlabel('epoch');
                legend('Training', 'Test', 'Validation'); hold off;
				break
            end
            
        end
             
    end
    
    fprintf('\nTraining finished\n');
    subplot(1,2,1);
    plot(accuracy{1}) ; hold on; plot(accuracy{2}); hold on; plot(accuracy{3});
    title('Accuracy evolution per epoch'), ylabel('Accuracy'), xlabel('epoch');
    legend('Training', 'Test', 'Validation'); hold off;
                
    subplot(1,2,2);
    plot(cost{1}) ; hold on; plot(cost{2}); hold on; plot(cost{3});
    title('Cost evolution per epoch'), ylabel('Cost'), xlabel('epoch');
    legend('Training', 'Test', 'Validation'); hold off;   
end
