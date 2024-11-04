# CS441 Fall2024 - HW2
## Vibha Navale
#### UIN: 676301415
#### NetID: vnava22@uic.edu

Repo for the Spark Homework-2 for CS441 Fall 2024

Project walkthrough:

## Environment:
**OS** : macOS (M3 Chip)

---

## Prerequisites:

- Spark Version 3.5.3
- Scala 2.12.18
- SBT (1.10.2) and SBT Assembly (2.2.0)
- Java 11
- Download the IMDB dataset of 50K movie reviews from Kaggle (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Run Tokenizer and Word2Vec files from HW 1 to get the vector embeddings for this assignment, or download embeddings from HuggingFace or JFastText websites
---

## Running the project
1) Download the repo from git
2) The root file is found in _src/main/scala/app/MainApp.scala_
3) Run `sbt clean update` and `sbt clean compile` from the terminal.
4) Run `sbt "run app.MainApp"` without any parameters;
   - OR run on Spark using the command `spark-submit --class app.MainApp target/scala-2.12/CS-441-HW-2-assembly-0.1.jar --hdfs`
   - Use `spark-submit --class app.MainApp --driver-memory 4g --executor-memory 4g target/scala-2.12/CS-441-HW-2-assembly-0.1.jar --hdfs` if there are any memory issues.
5) Run `sbt test` to test.
6) To create the jar file, run the command `sbt clean assembly`.
7) The resulting jar file can be found at _target/scala-2.13/CS-441-HW-assembly-0.1.jar_

- Make sure that your local input/output folder has the requisite permissions to allow the program to read and write to it
- Make sure Spark and Hadoop is running on your machine before you run the program. 
  - Start Hadoop using `start-dfs.sh` and `start-yarn.sh`
  - Start Spark using `$SPARK_HOME/sbin/start-master.sh` and `$SPARK_HOME/sbin/start-worker.sh spark://localhost:7077`

---

### Parameters
1. Sample input path - ```input_data/embeddings/part-r-00000```
2. Sample output path - ```output_data/model.zip``` and ```output_data/training_metrics.txt``` 

---

## Requirements:

In this homework, the goal is to implement a Large Language Model (LLM) training process using a Sliding Window approach in a distributed cloud environment with Apache Spark and DL4J, utilizing a Neural Network to train the model.
1) Use the tokens and embeddings generated from HW 1, or download pre-trained embeddings if needed.
2) Split the input embeddings into shards for parallel processing, ensuring that each shard is large enough to capture meaningful patterns while remaining manageable in size.
3) Implement the Sliding Window approach to create overlapping sequences of tokens from the input data, allowing the model to learn context effectively. This involves defining the window size and step size to control how much data is included in each training sample.
4) Finally, train the model using a neural network in Spark, optimizing for performance and scalability. During training, output measurements such as loss, accuracy, and other relevant metrics to evaluate the model's learning progress and performance. Additionally, visualize these metrics to provide insights into the training process.

Other Requirements:

1) Logging used for all programs
2) Configurable input and output paths for the program
3) Compilable through sbt
4) Deployed on AWS EMR

---

## Technical Design

We will take a look at the detailed description of how each of these pieces of code work below. Line by line comments explaining every step are also added to the source code in this git repo:

1) ### [MainApp](src/main/scala/app/MainApp.scala) [App]
    This object serves as the entry point for running the LLM training process, implementing a Sliding Window approach using Apache Spark and DL4J. It handles the initialization and execution of the embedding processing and neural network training tasks.
- The main method takes command-line arguments to determine the execution mode: --local, --hdfs, or --emr. Only one mode can be specified to avoid conflicts.
- Based on the selected mode, it retrieves the input path (for the .csv dataset), tokenized output path, and final output path from the configuration file using Typesafe Config.
- The application initializes a Spark session appropriate for the selected mode. In local mode, it runs on a local Spark cluster, while hdfs and emr modes leverage distributed environments.
- Job Flow:
    - Loads embeddings from the input path and logs the count.
    - Generates sliding windows from tokens and embeddings, logging the total created.
    - Constructs a neural network model and trains the model using the sliding windows RDD.
- Logs each stage and handles errors if any output is missing or something fails.

2) ### [SlidingWindow](src/main/scala/WordEmbeddingProcessor/SlidingWindow.scala) [WordEmbeddingProcessor]
   Handles the processing of word embeddings and tokens for the training of a language model.
- Loads embeddings from a specified file path, returning a map of tokens to their corresponding vectors.
- Reads tokens from a file, extracting the first token from each line and returning them as a list.
- Generates sliding windows of tokens and their embeddings, applying positional encodings to enhance the embeddings with positional information.
- Calculates positional embeddings using sine and cosine functions to improve the model's understanding of token order in the window.

3) ### [NeuralNetwork](src/main/scala/WordEmbeddingProcessor/NeuralNetwork.scala) [WordEmbeddingProcessor]
   Designed to build and train a neural network using Deep Learning for Java (DL4J) within a Spark environment.
- A custom training listener that logs training metrics, including scores and learning rates, to a file for monitoring model performance during training.
- Configures a multi-layer LSTM network with specified input and output sizes, utilizing Adam optimizer and Xavier weight initialization. It also sets up a TrainingMaster for distributed training.
- Trains the neural network on a dataset prepared from sliding windows of embeddings. It handles model persistence, including checking and creating output directories, deleting existing models, and saving the trained model to disk.

4) ### [TextGenerator](src/main/scala/app/TextGenerator.scala) [App]
   Facilitates the generation of text sequences using a trained LSTM model within a Spark environment.
- Predicts the next word based on a given context by embedding the context and passing it through the model. It returns the predicted word as a string.
- Constructs a sentence starting from a seed text by iteratively generating words until a specified maximum word count is reached or a termination condition is met (e.g., a period or "END").
- Converts an array of words into an INDArray by looking up their embeddings. It returns a zero vector if no valid embeddings are found or if the concatenated vector length is not 50.
- Processes an RDD of seed texts in parallel to generate sentences using the specified model and embeddings.

---

## Test Cases
These are run through the command `sbt test`

---

