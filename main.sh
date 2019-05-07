SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DATA_DIR="$SCRIPTPATH/data/iris_classifier_model"

# Clean up the directory storing model data.
rm -rf $DATA_DIR

# --- TensorFlow part ---
cd ./tf-python
# Train the model
python iris_classifier.py
# Freeze the model
python freeze_graph.py \
    --input_graph="$DATA_DIR/graph.pbtxt" \
    --input_checkpoint="$DATA_DIR/model.ckpt-1000" \
    --output_graph="$DATA_DIR/frozen_graph.pb" \
    --output_node_names="neural_network/probabilities"
cd ../

# --- Java part ---
cd ./tf-java
# Compile the JAVA file.
mvn clean package
# Run the main method.
java -cp target/tf-java-1.0-SNAPSHOT-jar-with-dependencies.jar com.wz.tfjexample.IrisClassifier
cd ../