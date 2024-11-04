package WordEmbeddingProcessor

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import java.io.{File, IOException}

object NeuralNetwork {
  private val logger = LoggerFactory.getLogger(getClass)

  def buildModel(sc: SparkContext, outputSize: Int): SparkDl4jMultiLayer = {
    val inputSize = 50 // Input size for LSTM
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(inputSize) // Set input size for the LSTM layer
        .nOut(100) // Hidden layer size
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(100) // LSTM output size
        .nOut(outputSize) // Number of output classes
        .build())
      .build()

    // Configure TrainingMaster for distributed training
    val tm: TrainingMaster[_, _] =
      new ParameterAveragingTrainingMaster.Builder(1)
        .workerPrefetchNumBatches(1)
        .batchSizePerWorker(64) // Match the batch size
        .averagingFrequency(3)
        .build()

    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNet.setListeners(new ScoreIterationListener(10))

    sparkNet
  }

  def trainModel(spark: SparkSession, sparkNet: SparkDl4jMultiLayer, windowsRDD: RDD[(Array[String], Array[Array[Double]])], outputSize: Int): Unit = {
    val sc: SparkContext = spark.sparkContext
    val modelPath = "model.zip"
    val modelFolder = new File(modelPath)

    if (modelFolder.exists()) {
      try {
        modelFolder.delete()
        logger.info(s"Deleted existing model folder at $modelPath.")
      } catch {
        case e: IOException =>
          logger.error(s"Failed to delete existing model folder: ${e.getMessage}")
      }
    }

    // Define your window size (time steps)
    val windowSize = 10
    val featuresAndLabelsRDD = windowsRDD.flatMap {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == 50 =>
        val numSamples = vectors.length - windowSize + 1 // Calculate number of samples based on window size
        val batchSize = 64 // This is the number of samples in a batch

        // Create a list to hold DataSets
        val dataSets = for (i <- 0 until numSamples) yield {
          // Extract a window of data
          val featureWindow = vectors.slice(i, i + windowSize)
          val features = Nd4j.create(featureWindow).reshape(1, 50, windowSize) // Reshape to (batchSize, features, windowSize)

          // Create labels for this sample (this is just a placeholder, adjust according to your needs)
          val labels = Nd4j.zeros(1, outputSize, windowSize) // Ensure this matches your output size

          new DataSet(features, labels) // Create DataSet for the current window
        }

        // Convert the sequence of DataSets to an RDD
        Some(dataSets) // Wrap in Option to handle empty cases
      case _ => None // Filter out any invalid entries
    }.flatMap(identity) // Flatten the sequences into a single RDD of DataSets

    // Perform distributed training
    sparkNet.fit(featuresAndLabelsRDD)

    // Save the model if needed
    sparkNet.getNetwork.save(modelFolder, true)
    println(s"Model saved at ${modelFolder.getAbsolutePath}")
  }
}
