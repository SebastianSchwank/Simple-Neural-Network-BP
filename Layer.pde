//The following class implements a basic Neural-Network-Layer
//which is capable to perform feed forward function
//and error-feedback on n-Inputs and m-Outputs.
//The activation function is the sigmoid-function:
// f(x) = 1/(1+e^(-x))

class layer{
  float     learningRate,momentumRate;
  float[][] weights;
  float[][] momentum;
  
  
  layer(int numInputs,int numOutputs,float learningRate,float momentumRate){
    this.learningRate = learningRate;
    this.momentumRate = momentumRate;
      this.weights = new float[numInputs+1][numOutputs];
      this.momentum= new float[numInputs+1][numOutputs];
      //Init weights to random small numbers in range [-1.0..1.0]
      for(int i = 0; i < numInputs+1; i++)
        for(int j = 0; j < numOutputs; j++){
         this.weights[i][j] = random(-1.0,+1.0);
         this.momentum[i][j]= 0.0;
        }
  }
  
  void draw(){
    
      for(int i = 0; i < this.weights.length; i++)
        for(int j = 0; j < this.weights[0].length; j++){
          color c = color(this.weights[i][j]*255.0);
          set(i,j,c);
        }
    
  }
  
  void resetWeights(){
    //Init weights to random small numbers in range [-1.0..1.0]
      for(int i = 0; i < this.weights.length; i++)
        for(int j = 0; j < this.weights[0].length; j++){
         //this.weights[i][j] = random(-1.0,+1.0);
         this.momentum[i][j] += random(-0.1,0.0);
        }
  }
  
  //Perform the feed forward pass through the fully connected
  //artifical Neural Network with n x m weights, n inputs
  //and m outputs
  float[] feedForward(float[] inputs){
    float[] outputs = new float[this.weights[0].length];
    for(int j = 0; j < this.weights[0].length; j++){
      float sum = 0.0;
      for(int i = 0; i < inputs.length; i++)
        sum += inputs[i] * this.weights[i][j];
      sum += 1.0 * this.weights[inputs.length][j];
      outputs[j] = 1.0/(1.0+exp(-sum));
    }
    return outputs;
  }
  
  //Feedback a given error corresponding to it's input,
  //correct weights and output the backpropagated error.
                    //THIS want to be understooden:
  //Gradient descent of the sigmoid with:
  //feedbackError[i] = (sum[over all j's](errors[j] x weights[j]))[i] * (1.0-input[i]) * input[i];
  //and "at the same time": 
  //delta_Weight[i][j] = input[i] * learningRate * error[j]
  float[] feedbackError(float[] inputs, float[] errors, float[] output){
    
    for(int j = 0; j < output.length; j++){ errors[j] = errors[j]* (output[j]) * (1.0-output[j]);} //<>// //<>//
    
    float[] feedbackError = new float[inputs.length];
    for(int i = 0; i < inputs.length; i++){
      float errorSum = 0.0;
      for(int j = 0; j < this.weights[i].length; j++){
            errorSum           += errors[j] * weights[i][j];
            this.weights[i][j] += inputs[i] * errors[j] * 0.1 ;
            //print(inputs[i] * errors[j]);
      }
      feedbackError[i] = errorSum;
      //print(errorSum);
    }
    //Process Bias (? doesn't count to feedbackError:(!)
    
    int i = weights.length-1;
    float errorSum = 0.0;
    for(int j = 0; j < this.weights[i].length; j++){
          //this.momentum[i][j] = 1.0       * errors[j] + this.momentum[i][j] * this.momentumRate;
          this.weights[i][j] += 1.0       * errors[j] * 0.1;
    }
    
    return feedbackError;
  }
  
}
