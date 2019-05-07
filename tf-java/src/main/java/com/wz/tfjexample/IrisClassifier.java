package com.wz.tfjexample;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.HashMap;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * Sample use of the TensorFlow Java API to classify Iris data using a pre-trained (frozen) model.
 */
public class IrisClassifier {

  public static void main(String[] args) {

    byte[] graphDef = getGraphDef();

    Map<String, Tensor<Double>> inputs = getInputs();
    try {
      float[][] batchedLabelProbabilities = executeGraph(graphDef, inputs);
      for (float[] labelProbabilities : batchedLabelProbabilities) {
        int bestLabelIdx = maxIndex(labelProbabilities);
        System.out.format("%nPredicted class: %d%n", bestLabelIdx);
      }

    } finally {
      for (Tensor<Double> t : inputs.values())
        t.close();
    }
  }

  /**
   * @return The pre-trained frozen model (as a graph).
   */
  private static byte[] getGraphDef() {
    String frozenGraphPath = "../data/iris_classifier_model/frozen_graph.pb";
    try {
      return Files.readAllBytes(Paths.get(frozenGraphPath));
    } catch (IOException e) {
      System.err.println("Failed to read [" + frozenGraphPath + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }

  /**
   * Example of creating input data.
   * We put two samples in one batch as the inputs by hands.
   * @return Map from feature name to its feature value batch.
   */
  private static Map<String, Tensor<Double>> getInputs() {
    Map<String, Tensor<Double>> inputs = new HashMap<String, Tensor<Double>>();
    inputs.put("SepalLength", Tensor.create(new double[]{ 5.9, 6.9 }, Double.class));
    inputs.put("SepalWidth", Tensor.create(new double[]{ 3.0, 3.1 }, Double.class));
    inputs.put("PetalLength", Tensor.create(new double[]{ 4.2, 5.4 }, Double.class));
    inputs.put("PetalWidth", Tensor.create(new double[]{ 1.5, 2.1 }, Double.class));
    return inputs;
  }

  /**
   * Executes the graph by feeding the inputs and returns the outputs.
   */
  private static float[][] executeGraph(byte[] graphDef, Map<String, Tensor<Double>> inputs) {
    // Map from feature name to input node. This is model specific.
    Map<String, String> inputTensorNodes = new HashMap<String, String>();
    inputTensorNodes.put("SepalLength", "IteratorGetNext:2");
    inputTensorNodes.put("SepalWidth", "IteratorGetNext:3");
    inputTensorNodes.put("PetalLength", "IteratorGetNext:0");
    inputTensorNodes.put("PetalWidth", "IteratorGetNext:1");

    Graph g = new Graph();
    try {
      g.importGraphDef(graphDef);
      Session s = new Session(g);
      try {
        // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
        Session.Runner runner = s.runner();
        // Feeding
        for (String featureName : inputTensorNodes.keySet()) {
          String tensorNode = inputTensorNodes.get(featureName);
          runner.feed(tensorNode, inputs.get(featureName));
        }
        // Fetching
        // The output node (i.e. the "neural_network/probabilities" herein) is model specific.
        Tensor<Float> result = runner.fetch("neural_network/probabilities").run().get(0).expect(Float.class);

        // Convert to {@code float[][]}
        long[] rshape = result.shape();  // shape: [batchSize, nLabels]
        int batchSize = (int) rshape[0];
        int nLabels = (int) rshape[1];
        float[][] outputs = result.copyTo(new float[batchSize][nLabels]);  // shape: [batchSize, nLabels]
        return outputs;

      } finally {
        s.close();
      }

    } finally {
      g.close();
    }
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

}