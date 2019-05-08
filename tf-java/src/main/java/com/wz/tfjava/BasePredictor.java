package com.wz.tfjava;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public abstract class BasePredictor {

  byte[] graphDef;

  public BasePredictor() {
    this.graphDef = getGraphDef();
  }

  public abstract Map<String, String> getInputNodeMap();

  public abstract String getFrozenGraphPath();

  public abstract String getOutputNode();

  private byte[] getGraphDef() {
    String frozenGraphPath = getFrozenGraphPath();

    try {
      return Files.readAllBytes(Paths.get(frozenGraphPath));

    } catch (IOException e) {
      System.err.println("Failed to read [" + frozenGraphPath + "]: " + e.getMessage());
      System.exit(1);
    }

    return null;
  }

  /**
   * Executes the graph by feeding the inputs and returns the outputs.
   */
  public <T> Tensor<Float> executeGraph(Map<String, Tensor<T>> inputs) {
    Graph g = new Graph();
    try {
      g.importGraphDef(graphDef);
      Session s = new Session(g);
      try {
        // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
        Session.Runner runner = s.runner();
        // Feeding
        Map<String, String> inputNodeMap = getInputNodeMap();
        for (String featureName : inputNodeMap.keySet()) {
          String tensorNode = inputNodeMap.get(featureName);
          runner.feed(tensorNode, inputs.get(featureName));
        }
        // Fetching
        // The output node (i.e. the "neural_network/probabilities" herein) is model specific.
        String outputNode = getOutputNode();
        Tensor<Float> outputs = runner.fetch(outputNode).run().get(0).expect(Float.class);
        return outputs;

      } finally {
        s.close();
      }

    } finally {
      g.close();
    }
  }

}
