package com.wz.tfjava;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.Tensor;
import com.wz.tfjava.BasePredictor;

public class UniversalSentenceEncoder extends BasePredictor {

    @Override
    public Map<String, String> getInputNodeMap() {
        Map<String, String> inputNodeMap = new HashMap<String, String>();
        inputNodeMap.put("inputText", "text_input_placeholder:0");
        return inputNodeMap;
    }

    @Override
    public String getFrozenGraphPath() {
        return "../data/universal_sentence_encoder/frozen_use_graph.pb";
    }

    @Override
    public String getOutputNode() {
        return "module_apply_default/Encoder/l2_normalize";
    }

    public static void main(String[] args) throws UnsupportedEncodingException {
        // Construct input batch instance
        Map<String, Tensor<String>> inputs = new HashMap<String, Tensor<String>>();
        inputs.put("inputTexts", Tensor.create("What is your name?".getBytes("UTF-8"), String.class));
    
        UniversalSentenceEncoder predictor = new UniversalSentenceEncoder();
    
        try {
            Tensor<Float> outputs = predictor.executeGraph(inputs);
    
            try {
                // Convert to {@code float[][]}
                long[] rshape = outputs.shape();  // shape: [batchSize, nLabels]
                int batchSize = (int) rshape[0];
                int dim = (int) rshape[1];
                float[][] embedded = outputs.copyTo(new float[batchSize][dim]);
                String message = "";
                for (float x: embedded[0])
                    message += String.valueOf(x) + ", ";
                System.out.format("%nEmbedded: %s%n", message);
        
            } finally {
                outputs.close();
            }
        
        } finally {
            for (Tensor<String> t : inputs.values()) t.close();
        }
    }
}