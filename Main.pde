//The following Prototype is able to translate given
//sample-data, in this example synthesized by a frequency modulated
//sine function 1) , into a Model, here represented by a set of parameters
//for a fourjer-like-transformation 2).
//This is done by an ANN and the associated Simulator (in our case 2))
//1) sample synthesizing input function [0..4*PI]
//f(x) = 0.5+0.5*sin((x)*(1.0+sin(x))/2.0);
//2) target transformation
//t(x) = a0*(0.5+0.5*sin(f0*x+p0*2*PI)) + 
//       a1*(0.5+0.5*sin(f1*x+p1*2*PI)) +
//       ... 
//       aN*(0.5+0.5*sin(fN*x+pN*2*PI));
//       with N = 9 ( * 3 = 27 Parameters)

//TODO: make the learningRate adaptive to the squareSum of the errors
float learningRate = 0.123;
float momentum     = 0.1;
//
float jitter = 0.0;
FloatList errors;
//SimpleDeepFeedForwardArticialNeuralNetwork
ArrayList<layer> SDFFANN;
//int numOfLayers = 3;

//Number of Parameters
int numParameters = 8;
//Number of Sample Points
int numSamplePoints = 16;

float[] loopbackParameters;
float[] best;

//Train only if error is minimal
float errLess = 0;
boolean learnReset = false;

int frequencyNode = 0;

void setup(){
  size(1024,512);
  
  SDFFANN = new ArrayList<layer>();
  
  //Inialize Neural Network's Layer
  SDFFANN.add(new layer(numSamplePoints,numSamplePoints,learningRate,momentum));
  SDFFANN.add(new layer(numSamplePoints,numParameters,learningRate,momentum));
  SDFFANN.add(new layer(numSamplePoints,numParameters,learningRate,momentum));
  
  SDFFANN.add(new layer(numParameters,numSamplePoints,learningRate,momentum));
  SDFFANN.add(new layer(numSamplePoints*2,numSamplePoints,learningRate,momentum));
  SDFFANN.add(new layer(numSamplePoints,numParameters,learningRate,momentum));
}

void draw(){
    
  background(255);
  
  frequencyNode = (frequencyNode+1)%numParameters;
  
  //The input Sample Array
  float[] inputSamples = vectorizedInputFunction(numSamplePoints,0,2.0*PI,1.0*frequencyNode/64);
  
  //for(int i = 0; i < numSamplePoints; i++)
    //inputSamples[i] = 0.0;
    
  //inputSamples[frequencyNode] = 1.0;
  
  //Draw input-example-function
  stroke(0,0,255);
  for(int i = 1; i < numSamplePoints; i++)
    line(i-1,32+128*inputSamples[i-1],i,32+128*inputSamples[i]);
    
  float[] frequencyNodes = new float[numParameters];;
  for(int i = 0; i < numParameters; i++)
    frequencyNodes[i] = 0.0;
    
  frequencyNodes[frequencyNode] = 1.0;
  
  float[] output0 = SDFFANN.get(0).feedForward(inputSamples);
  float[] output1 = SDFFANN.get(1).feedForward(output0);
  //float[] output2 = SDFFANN.get(2).feedForward(output1);
  
  stroke(0,0,255);
  for(int i = 1; i < numParameters; i++)
    line(i-1,64+32*output1[i-1],i,64+32*output1[i]);
  
  float sumError = 0.0;
  float[] error = new float[numParameters];;
  for(int i = 0; i < numParameters; i++){
    error[i] = frequencyNodes[i]-output1[i];
    sumError += error[i]*error[i];
    //if(error[i] > 0.1) print(i+" "+error[i]);
  }
  
  //println(sumError);
  
  //float[] error0 = SDFFANN.get(2).feedbackError(output1,error,output2); //<>//
  float[] error1 = SDFFANN.get(1).feedbackError(output0,error,output1);
  float[] error2 = SDFFANN.get(0).feedbackError(inputSamples,error1,output0);
  
  errLess += sumError;
  if(frequencyNode == 7){ println(errLess);errLess = 0.0; }
  //println(frequencyNode);
  //SDFFANN.get(1).feedbackError(inputSamples,frequencyNodes);
  //SDFFANN.get(2).feedbackError(inputSamples,frequencyNodes);
  //SDFFANN.get(0).draw();
  
}
