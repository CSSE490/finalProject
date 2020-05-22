import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

//The code closely follows the algorithm in the Russel and Norvig, fig. 18.24
public class FeedForwardNetwork {
	
	private int inputLayerSize = 0;
	private int hiddenLayerSize = 0;
	private int numHiddenLayers = 0;
	private int outputLayerSize = 0;
	private int maxLayerSize = 0;

	
	public FeedForwardNetwork(int inputLayerSize, int hiddenLayerSize, int numHiddenLayers, int outputLayerSize){
		if (inputLayerSize <= 0) {
			System.out.println("Input layer must have at least one neuron.");
			System.exit(0);
		}
		this.inputLayerSize = inputLayerSize;
		
		if (hiddenLayerSize <= 0) {
			System.out.println("Hidden layer must have at least one neuron.");
			System.exit(0);
		}
		this.hiddenLayerSize = hiddenLayerSize;
		
		if (numHiddenLayers <= 0) {
			System.out.println("Must have at least one hidden layer.");
			System.exit(0);
		}
		this.numHiddenLayers = numHiddenLayers;
		
		if (outputLayerSize <= 0) {
			System.out.println("Output layer must have at least one neuron.");
			System.exit(0);
		}
		this.outputLayerSize = outputLayerSize;
		
		this.maxLayerSize = Math.max(Math.max(inputLayerSize, hiddenLayerSize), outputLayerSize);
	}

	private int trainingSetSize = 0;
	private double learningRate = 0;	
	private double inputs[][];
	private double desiredOutput[][];
	private double[][][] weights; // [layer][from][to]

	
	@SuppressWarnings("hiding")
	public void initNetwork(double[][] inputs, double[][] desiredOutput, double learningRate, double initialWeightOffset) {
		this.trainingSetSize = inputs.length;
		if (this.trainingSetSize == 0) {
			System.out.println("No training data.");
			System.exit(0);
		}
		if (inputs[0].length != this.inputLayerSize) {
			System.out.println("Mismatch between input layer size and training data length.");
			System.exit(0);
		}		
		this.inputs = inputs; 
		this.desiredOutput = desiredOutput;
		this.learningRate = learningRate;
		
		// To simplify the code, we generate a slightly oversized weight matrix. 
		// Not to worry though, we will only use what we need.
		this.weights = new double[this.numHiddenLayers+1][this.maxLayerSize][this.maxLayerSize]; 

		for (int l = 0; l < this.numHiddenLayers+1; l++) {
			for (int i = 0; i < this.hiddenLayerSize; i++){
				for (int j = 0; j < this.hiddenLayerSize; j++){
					this.weights[l][i][j] = Math.random() - initialWeightOffset; 	
				}
			}
		}
	}

	private static double sigmoidActivationFunction(double input){
		return 1.0 / (1 + Math.exp(-1.0 * input));
	}
	
	private static double sigmoidDerivative(double value){
		return value * (1 - value);
	}
	
	private void feedForward (double[][] activation, int fromLayerSize, int toLayerSize, int l){
		double inJ;
		for (int j = 0; j < toLayerSize; j++){
			inJ = 0;
			for (int i = 0; i < fromLayerSize; i++){
				inJ += this.weights[l][i][j] * activation[l][i];
			}
			// 0 is the first hidden layer
			activation[l+1][j] = sigmoidActivationFunction(inJ);
		}
	}
	
	public void trainNetwork(int iterations, boolean verbose){
		long start = System.currentTimeMillis();
		for (int k = 0; k < iterations; k++){
			double globalError = 0.0;
			//Run through entire training set once.
			for (int example = 0; example < this.trainingSetSize; example++){
			    // Used for progress bar. Uncomment if progress bar is desired. 
				//long startTime = System.currentTimeMillis();
				
				// We store the activation of each node (over all input and hidden layers) because we need that data during back propagation.
				// Please notice that in order to simplify the code, we store the activation of the input layer too.
				double[][] activation = new double[this.numHiddenLayers+2][this.maxLayerSize];
				// Reading in the activation from the training data example
				for (int i = 0; i < this.inputLayerSize; i++){
					activation[0][i] = this.inputs[example][i];
				}

				// There may be different sizes for the input, hidden and output layers, hence there are three different calls for feedForeard
				// input to first hidden layer
				feedForward(activation, this.inputLayerSize, this.hiddenLayerSize, 0);
				// hidden to hidden layers
				for (int l = 1; l < this.numHiddenLayers; l++){
					feedForward(activation, this.hiddenLayerSize, this.hiddenLayerSize, l);					
				}
				// last hidden layer to output layer
				feedForward(activation, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers);

				// calculating errors
				double[][] error = new double[this.numHiddenLayers+1][this.maxLayerSize];
				// Calculating errors at output layer
				for (int j = 0; j < this.outputLayerSize; j++) {					
					error[this.numHiddenLayers][j] = sigmoidDerivative(activation[this.numHiddenLayers+1][j]) * (this.desiredOutput[example][j] - activation[this.numHiddenLayers+1][j]);
					globalError += error[this.numHiddenLayers][j]*error[this.numHiddenLayers][j];
				}
				// Calculating error of last hidden layer. 
				calculateError(error, activation, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers-1);
				
				// Calculating error of remaining hidden layers. 
				for (int l = this.numHiddenLayers - 2; l >= 0; l--){				
					calculateError(error, activation, this.hiddenLayerSize, this.outputLayerSize, l);
				}				

				/** Adjusting weights **/ 
				
				//adjusting weights at output layer
				adjustWeights(activation, error, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers);
				// Adjusting weights at hidden layers.
				for (int l = this.numHiddenLayers-1; l > 0; l--){
					adjustWeights(activation, error, this.hiddenLayerSize, this.hiddenLayerSize, l);
				}
				// Adjusting weights at input layer.
				adjustWeights(activation, error, this.inputLayerSize, this.hiddenLayerSize, 0);
				// Uncomment the following line for a progress bar while training. Does not work in ecplise but does on the command line.
//				if (verbose) {
//					 printProgress(startTime, this.trainingSetSize, (example+1));
//				}
			}
			if (verbose) {
				System.out.println("Completed iteration " + (k+1) +" out of "+ iterations + " " + round((((double)(k+1)/iterations)), 4) + "% Complete");
				// calculating RMS
				System.out.println("Global error: " + Math.sqrt(globalError/(this.trainingSetSize*this.outputLayerSize)));
			}
		}
		System.out.println("training time: " + (System.currentTimeMillis() - start) / 60000.0 + "min");
		
	}
	
	private void calculateError(double[][] error, double[][] activation, int fromLayerSize, int toLayerSize, int l){
		double e;
		for (int i = 0; i < fromLayerSize; i++) {
			e = 0;
			for (int j = 0; j < toLayerSize; j++){
				e += this.weights[l+1][i][j] * error[l+1][j];
			}
			// Recall that activation is off by one, due to activation of input layer.
			error[l][i] = sigmoidDerivative(activation[l+1][i]) * e;
		}
	}
	
	private void adjustWeights(double[][] activation, double[][] error, int fromLayerSize, int toLayerSize, int l){
		for (int i = 0; i < fromLayerSize; i++){
			for (int j = 0; j < toLayerSize; j++){
				this.weights[l][i][j] += this.learningRate * activation[l][i] * error[l][j];
			}
		}		
	}


	@SuppressWarnings("boxing")
	public void printWeights(){
		System.out.println("Input layer.");		
		for (int i = 0; i < this.inputLayerSize; i++){
			for (int j = 0; j < this.hiddenLayerSize; j++){
				System.out.printf("Weight from input node %d to node %d is %f.\n", i, j, this.weights[0][i][j]);
			}
		}
		
		System.out.println("\nHidden layers.");
		for (int l = 0; l < this.numHiddenLayers-1; l++) {
			for (int i = 0; i < this.hiddenLayerSize; i++){
				for (int j = 0; j < this.hiddenLayerSize; j++){
					System.out.printf("Weight at hidden layer %d from node %d to node %d is %f.\n", l, i, j, this.weights[l][i][j]);
				}
			}
		}
		
		System.out.println("\nOutput layer.");
		for (int i = 0; i < this.hiddenLayerSize; i++){
			for (int j = 0; j < this.outputLayerSize; j++){
				System.out.printf("Weight from node %d to output node %d is %f.\n", i, j, this.weights[this.numHiddenLayers][i][j]);
			}
		}
}
	
	public void testNetwork(double[][] imageDatatest, double[][] labelDatatest){
		int count = 0;
		HashMap<Integer, HashMap<Integer, Integer>> confusion = new HashMap<>();
		for (int i = 0; i < 10; i++) {
			HashMap<Integer, Integer> map = new HashMap<>();
			for (int j = 0; j < 10; j++) {
				map.put(j, 0);
			}
			confusion.put(i, map);
		}
		//Run through entire training set once.
		for (int example = 0; example < imageDatatest.length; example++){
			// We store the activation of each node (over all input and hidden layers) because we need that data during back propagation.
			// Please notice that in order to simplify the code, we store the activation of the input layer too.
			double[][] activation = new double[this.numHiddenLayers+2][this.maxLayerSize];
			// Reading in the activation from the training data example
			for (int i = 0; i < this.inputLayerSize; i++){
				activation[0][i] = imageDatatest[example][i];
			}

			// There may be different sizes for the input, hidden and output layers, hence there are three different calls for feedForeard
			// input to first hidden layer
			feedForward(activation, this.inputLayerSize, this.hiddenLayerSize, 0);
			// hidden to hidden layers
			for (int l = 1; l < this.numHiddenLayers; l++){
				feedForward(activation, this.hiddenLayerSize, this.hiddenLayerSize, l);					
			}
			// last hidden layer to output layer
			feedForward(activation, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers);
			int numberIdentify = -1;
			double biggest = -1;
			double answer = -1;
			for (int j = 0; j < this.outputLayerSize; j++){
				//if (Math.abs(activation[this.numHiddenLayers+1][j] - labelDatatest[example][j]) > 0.1) {
					//System.out.println("Output neuron " + j + " has: " + activation[this.numHiddenLayers+1][j] + " should be: " + labelDatatest[example][j]);
					//count++;
				double out = activation[this.numHiddenLayers+1][j];
				if(out > biggest) {
					biggest = out;
					numberIdentify = j;
				}
				if(labelDatatest[example][j] == 1) {
					answer = j;
				}
			}
			System.out.print((int)numberIdentify  + ",");
			int temp = confusion.get((int)answer).get(numberIdentify) + 1;
			confusion.get((int)answer).put((int)answer, temp);
			
			if(numberIdentify != answer) {
//				System.out.println("***********************************  " + example + " expected answer:" + answer);
//				System.out.println("output:" + numberIdentify);
				count++;
			}
		}
		System.out.println();
		System.out.println(count + " errors out of " + imageDatatest.length);
		System.out.println("Done testing.");
		for (int i = 0; i < 10; i++) {
			System.out.print(i + ", ");
		}
		System.out.println("----------------------------");
		for (int i = 0; i < 10; i++) {
			System.out.println(i);
			for (int j = 0; j < 10; j++) {
				int result = confusion.get(i).get(j);
				System.out.print(result + ", ");
			}
			System.out.println("----------------------------");
			
		}
	}
	
	/***
	 * Allows a trained network to be tested on a supplied test batch.
	 *
	 * @param testingSetSize - size of testing set.
	 * @param test_inputs - the actual testing set.
	 * @param test_labels - the testing sets abels.
	 * @param verbose - displays information regarding failures.
	 */
	public void testNetworkBatch(int testingSetSize, double[][] test_inputs, double[][] test_labels, boolean verbose) {
		// Used for progress bar. Uncomment if progress bar is desired. 
		long startTime;
		int error_count = 0;
		//Run through entire training set once.
		for (int example = 0; example < testingSetSize; example++){
		    // Used for progress bar. Uncomment is progress bar is desired. 
			startTime = System.currentTimeMillis();
			
			// We store the activation of each node (over all input and hidden layers) because we need that data during back propagation.
			// Please notice that in order to simplify the code, we store the activation of the input layer too.
			double[][] activation = new double[this.numHiddenLayers+2][this.maxLayerSize];
			// Reading in the activation from the training data example
			for (int i = 0; i < this.inputLayerSize; i++){
				activation[0][i] = test_inputs[example][i];
			}
	
			// There may be different sizes for the input, hidden and output layers, hence there are three different calls for feedForeard
			// input to first hidden layer
			feedForward(activation, this.inputLayerSize, this.hiddenLayerSize, 0);
			// hidden to hidden layers
			for (int l = 1; l < this.numHiddenLayers; l++){
				feedForward(activation, this.hiddenLayerSize, this.hiddenLayerSize, l);					
			}
			// last hidden layer to output layer
			feedForward(activation, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers);
	
			for (int j = 0; j < this.outputLayerSize; j++){
				if (Math.abs(activation[this.numHiddenLayers+1][j] - test_labels[example][j]) > 0.1) {
					error_count++;
					if (verbose) {
						System.out.println("Output Layer: " + round(activation[this.numHiddenLayers+1][0], 2) + ", " + round(activation[this.numHiddenLayers+1][1], 2) + ", " +
							round(activation[this.numHiddenLayers+1][2], 2) + ", " + round(activation[this.numHiddenLayers+1][3], 2) + ", " +  round(activation[this.numHiddenLayers+1][4], 2) + ", " + 
							round(activation[this.numHiddenLayers+1][5], 2) + ", " + round(activation[this.numHiddenLayers+1][6], 2) + ", " + round(activation[this.numHiddenLayers+1][7], 2) + ", " + 
							round(activation[this.numHiddenLayers+1][8], 2) + ", " + round(activation[this.numHiddenLayers+1][9], 2));
						System.out.println("Label Layer: " + test_labels[example][0] + ", " + test_labels[example][1] + ", " +
							test_labels[example][2] + ", " + test_labels[example][3] + ", " +  test_labels[example][4] + ", " + 
							test_labels[example][5] + ", " + test_labels[example][6] + ", " + test_labels[example][7] + ", " + 
							test_labels[example][8] + ", " + test_labels[example][9]);
					}

				}
			}
			// Uncomment if progress bar is desired. 
			if (verbose) {
				printProgress(startTime, example, testingSetSize);
			}
		}
		System.out.println(error_count + " errors");
		System.out.println("Done testing.");
		System.out.println("Number correct: " + (testingSetSize - error_count) + " out of: " + testingSetSize);
		System.out.println("Overall accuracy: " +   round((((double)(testingSetSize-error_count)/testingSetSize)), 4));
	}
	
	public static double round(double value, int places) {
	    if (places < 0) {
	    	throw new IllegalArgumentException();
	    }
	    
	    BigDecimal bd = new BigDecimal(value);
	    bd = bd.setScale(places, RoundingMode.HALF_UP);
	    return bd.doubleValue();
	}
	// Method works well on console winows running through the command line
	// however this method does not work well with the Eclipse console window.
	@SuppressWarnings({ "boxing", "unused" })
	private static void printProgress(long startTime, long total, long current) {
	    long eta = current == 0 ? 0 : 
	        (total - current) * (System.currentTimeMillis() - startTime) / current;

	    String etaHms = current == 0 ? "N/A" : 
	            String.format("%02d:%02d:%02d", TimeUnit.MILLISECONDS.toHours(eta),
	                    TimeUnit.MILLISECONDS.toMinutes(eta) % TimeUnit.HOURS.toMinutes(1),
	                    TimeUnit.MILLISECONDS.toSeconds(eta) % TimeUnit.MINUTES.toSeconds(1));

	    StringBuilder string = new StringBuilder(140);   
	    int percent = (int) (current * 100 / total);
	    string
	        .append('\r')
	        .append(String.join("", Collections.nCopies(percent == 0 ? 2 : 2 - (int) (Math.log10(percent)), " ")))
	        .append(String.format(" %d%% [", percent))
	        .append(String.join("", Collections.nCopies(percent, "=")))
	        .append('>')
	        .append(String.join("", Collections.nCopies(100 - percent, " ")))
	        .append(']')
	        .append(String.join("", Collections.nCopies(current == 0 ? (int) (Math.log10(total)) : (int) (Math.log10(total)) - (int) (Math.log10(current)), " ")))
	        .append(String.format(" %d/%d, ETA: %s", current, total, etaHms));

	    System.out.print(string);
	    if (current == total) {
	    	System.out.println();
	    }
	}

}
