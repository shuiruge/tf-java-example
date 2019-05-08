package com.wz.tfjava;

import java.util.HashMap;
import java.util.Map;

import com.wz.tfjava.BasePredictor;

import org.tensorflow.Tensor;

public class IrisPredictor extends BasePredictor {

  @Override
  public Map<String, String> getInputNodeMap() {
    Map<String, String> inputNodeMap = new HashMap<String, String>();
    inputNodeMap.put("SepalLength", "IteratorGetNext:2");
    inputNodeMap.put("SepalWidth", "IteratorGetNext:3");
    inputNodeMap.put("PetalLength", "IteratorGetNext:0");
    inputNodeMap.put("PetalWidth", "IteratorGetNext:1");
    return inputNodeMap;
  }

  @Override
  public String getFrozenGraphPath() {
    return "../data/iris_classifier_model/frozen_graph.pb";
  }

  @Override
  public String getOutputNode() {
    return "neural_network/probabilities";
  }
  
  /**
   * Returns the index corresponding to the maximum probability.
   */
  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }
  
  public static void main (String[] args) {
    // Construct input batch instance
    Map<String, Tensor<Double>> inputs = new HashMap<String, Tensor<Double>>();
    inputs.put("SepalLength", Tensor.create(new double[]{ 5.9, 6.9 }, Double.class));
    inputs.put("SepalWidth", Tensor.create(new double[]{ 3.0, 3.1 }, Double.class));
    inputs.put("PetalLength", Tensor.create(new double[]{ 4.2, 5.4 }, Double.class));
    inputs.put("PetalWidth", Tensor.create(new double[]{ 1.5, 2.1 }, Double.class));    

    IrisPredictor predictor = new IrisPredictor();

    try {
      Tensor<Float> outputs = predictor.executeGraph(inputs);

      try {
        // Convert to {@code float[][]}
        long[] rshape = outputs.shape();  // shape: [batchSize, nLabels]
        int batchSize = (int) rshape[0];
        int nLabels = (int) rshape[1];
        float[][] batchedLabelProbabilities = outputs.copyTo(new float[batchSize][nLabels]);  // shape: [batchSize, nLabels]

        for (float[] labelProbabilities : batchedLabelProbabilities) {
          int bestLabelIdx = maxIndex(labelProbabilities);
          System.out.format("%nPredicted class: %d%n", bestLabelIdx);
        }

      } finally {
        outputs.close();
      }

    } finally {
      for (Tensor<Double> t : inputs.values()) t.close();
    }
  }

}