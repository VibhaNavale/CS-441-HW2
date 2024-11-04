package WordEmbeddingProcessor

import com.typesafe.config.ConfigFactory
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
import org.deeplearning4j.nn.api.Model
import org.slf4j.LoggerFactory

import java.io.{File, FileWriter, IOException}

object NeuralNetwork {
  private val logger = LoggerFactory.getLogger(getClass)
  private val config = ConfigFactory.load()

  class CustomListener(initialLearningRate: Double) extends ScoreIterationListener(10) {
    private val logger = LoggerFactory.getLogger(getClass)
    private val outputFile = new File("output_data/training_metrics.txt")

    // Override the iterationDone method from ScoreIterationListener
    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      super.iterationDone(model, iteration, epoch)

      // Ensure model is of type SparkDl4jMultiLayer
      model match {
        case sparkModel: SparkDl4jMultiLayer => // Use pattern matching
          val score = sparkModel.getNetwork.score()
          val learningRate = initialLearningRate

          // Safely get parameters from the model
          val paramsShape = sparkModel.getNetwork.getLayer(0).getParam("W").shape.mkString(",")

          val output = s"Iteration: $iteration, Epoch: $epoch, Score: $score, Learning Rate: $learningRate, Params Shape: $paramsShape\n"
          println(output)

          // Write to file
          try {
            val writer = new FileWriter(outputFile, true) // Append mode
            writer.write(output)
            writer.close()
          } catch {
            case e: IOException =>
              logger.error(s"Failed to write to output file: ${e.getMessage}")
          }
        case _ =>
          logger.error(s"Model is not an instance of SparkDl4jMultiLayer, received: ${model.getClass.getName}")
      }
    }
  }

  def buildModel(sc: SparkContext, outputSize: Int): SparkDl4jMultiLayer = {
    val inputSize = config.getInt("app.embeddingDimensions") // Input size for LSTM
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
        .batchSizePerWorker(32) // Adjust batch size for better learning
        .averagingFrequency(1) // Increase frequency of parameter averaging
        .build()

    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNet.setListeners(new CustomListener(0.001)) // Use the CustomListener

    sparkNet
  }

  def trainModel(spark: SparkSession, sparkNet: SparkDl4jMultiLayer, windowsRDD: RDD[(Array[String], Array[Array[Double]])], outputSize: Int): Unit = {
    val sc: SparkContext = spark.sparkContext
    val modelPath = "output_data/model.zip"
    val modelFolder = new File(modelPath)

    // Check if the parent directory exists; if not, create it
    val outputDir = modelFolder.getParentFile
    if (!outputDir.exists()) {
      outputDir.mkdirs() // Create the directory
      logger.info(s"Created output directory at ${outputDir.getAbsolutePath}.")
    }

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
    val windowSize = config.getInt("app.windowSize")
    val featuresAndLabelsRDD = windowsRDD.flatMap {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == config.getInt("app.embeddingDimensions") =>
        val numSamples = vectors.length - windowSize + 1 // Calculate number of samples based on window size

        // Create a list to hold DataSets
        val dataSets = for (i <- 0 until numSamples) yield {
          // Extract a window of data
          val featureWindow = vectors.slice(i, i + windowSize)
          val features = Nd4j.create(featureWindow).reshape(config.getInt("app.batchSize"), config.getInt("app.embeddingDimensions"), windowSize) // Reshape to (1, features, windowSize)

          // Create labels for this sample (adjust expectedClassIndex according to your needs)
          val labels = Nd4j.zeros(config.getInt("app.batchSize"), outputSize, windowSize)

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
