# Neural Network (MultiLayer Perceptron) from Scratch using Matlab

INSTRUCTIONS

Main function:
----------------------------

**FILENAME: NeuralNetwork.m**

Function name: NeuralNetwork
Parameters:
-	inputs: matrix with features in the rows and instances in the columns
-	targets: matrix with outputs in the rows and instances in the columns
-	nodeLayers: vector with the number of neurons in each layer, including the input and output
-	NumEpochs: Number of epochs in the training
-	batchSize: number of examples for each batch
-	eta: learning rate
-	trainSplit: split correspondent for the train subset
-	testSplit: split correspondent for the test subset
-	validSplit: split correspondent for the validation subset
-	savedNet: previously saved network to reload
-	HidActivFunction: activation function to use in hidden layers
-	LastActivFunction: activation function to use in the output layer
-	CostFunction: cost function to be used (“cross”, “quad”, “log”)
-	Momentum: momentum parameter, it must be between 0 and 1
-	Lambda: parameter for L2 regularization.
-	earlyStopEpochsPercent: minimum percent of epochs run before checking early stop measures.

Output:
-	weights: weights of each connection
-	biases: biases of each node
-	accuracy: accuracies of training, test and validation sets.
-	cost: cost of training, test and validation sets

To call the function (example):
NeuralNetwork (input, target, [3 10 2], 50, 10, 0.2, 6, 2, 2, ‘No’, ‘sigmoid’, ‘sigmoid’, ‘cross’, 0.3, 3, 80) 

To save results:
[a, b, c, d] = NeuralNetwork( inputs, targets, nodeLayers, numEpochs, batchSize, eta, trainSplit, testSplit, validSplit, savedNet, HidActivFunction, LastActivFunction, CostFunction, momentum, L2lamb, earlyStopEpochsPercent)


Auxiliary functions
-------------------------------------------------------------------------------
**FILENAME: activFunction.m**

Function name: activFunction
Parameters:
-	z: value that will be processed.
-	func: function chosen to process z. It can be “relu”, “softmax”, “tanh” and “sigmoid”.
Output:
-	Processed value.
To call the function:
activFunction(z, func)

-------------------------------------------------------------------------------
**FILENAME: derivFunction.m**

Function name: derivFunction
Parameters:
-	z: value that will be processed.
-	func: function chosen to process z. It can be “relu”, “softmax”, “tanh” and “sigmoid”.
Output:
-	Processed value.
To call the function:
derivFunction(z, func)

-------------------------------------------------------------------------------
**FILENAME: costFunction.m**
Function name: costFunction
Parameters:
-	input: matrix with features in the rows and instances in the columns
-	output: matrix with calculated outputs in the rows and instances in the columns
-	target: matrix with actual outputs in the rows and instances in the columns
-	nodeLayers: vector with the number of neurons in each layer, including the input and output
-	func: function chosed for the cost calculation. It can be “cross”, “quad” and “log”.
Output:
-	Costs
To call the function:
costFunction(input, output, target, nodeLayers, func)
