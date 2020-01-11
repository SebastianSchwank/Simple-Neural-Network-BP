//In this section we define the input-giving-function
//and the target-model-function

//input-giving-function
float[] vectorizedInputFunction(int numSamplePoints,float start,float end,float frequency){
  float[] samples = new float[numSamplePoints];
  float delta = (end - start)/numSamplePoints;
  float x = start;
  for(int i = 0; i < numSamplePoints; i++){
    samples[i] = 0.5+0.5*sin(x*2.0*PI*frequency);
    x += delta;
  }
  return samples;
}
