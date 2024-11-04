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
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.{File, FileWriter, IOException}

object NeuralNetwork {
  private val logger = LoggerFactory.getLogger(getClass)
  private val config = ConfigFactory.load()

  class CustomListener(initialLearningRate: Double, outputPath: String) extends ScoreIterationListener(10) {
    private val metricsFileName = "training_metrics.txt"

    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      super.iterationDone(model, iteration, epoch)

      model match {
        case sparkModel: SparkDl4jMultiLayer =>
          val score = sparkModel.getNetwork.score()
          val learningRate = initialLearningRate
          val paramsShape = sparkModel.getNetwork.getLayer(0).getParam("W").shape.mkString(",")

          val output = s"Iteration: $iteration, Epoch: $epoch, Score: $score, Learning Rate: $learningRate, Params Shape: $paramsShape\n"
          println(output)

          try {
            val metricsPath = new Path(s"$outputPath/$metricsFileName")
            val fs = FileSystem.get(sparkModel.getSparkContext.hadoopConfiguration)
            val outputStream = fs.create(metricsPath, true)
            outputStream.writeBytes(output)
            outputStream.close()
          } catch {
            case e: IOException =>
              logger.error(s"Failed to write to output file: ${e.getMessage}")
          }

        case _ =>
          logger.error(s"Model is not an instance of SparkDl4jMultiLayer, received: ${model.getClass.getName}")
      }
    }
  }

  def buildModel(sc: SparkContext, outputSize: Int, outputPath: String): SparkDl4jMultiLayer = {
    val inputSize = config.getInt("app.embeddingDimensions")
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(inputSize)
        .nOut(100)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputSize)
        .build())
      .build()

    val tm: TrainingMaster[_, _] =
      new ParameterAveragingTrainingMaster.Builder(1)
        .workerPrefetchNumBatches(1)
        .batchSizePerWorker(32)
        .averagingFrequency(1)
        .build()

    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNet.setListeners(new CustomListener(0.001, outputPath))

    sparkNet
  }

  def trainModel(spark: SparkSession, sparkNet: SparkDl4jMultiLayer, windowsRDD: RDD[(Array[String], Array[Array[Double]])], outputSize: Int, outputPath: String): Unit = {
    val sc: SparkContext = spark.sparkContext
    val modelPath = s"$outputPath/model.zip"
    val modelFolder = new File(modelPath)

    if (!modelFolder.getParentFile.exists()) {
      modelFolder.getParentFile.mkdirs()
      logger.info(s"Created output directory at ${modelFolder.getParentFile.getAbsolutePath}.")
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

    val windowSize = config.getInt("app.windowSize")
    val featuresAndLabelsRDD = windowsRDD.flatMap {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == config.getInt("app.embeddingDimensions") =>
        val numSamples = vectors.length - windowSize + 1
        val dataSets = for (i <- 0 until numSamples) yield {
          val featureWindow = vectors.slice(i, i + windowSize)
          val features = Nd4j.create(featureWindow).reshape(config.getInt("app.batchSize"), config.getInt("app.embeddingDimensions"), windowSize)
          val labels = Nd4j.zeros(config.getInt("app.batchSize"), outputSize, windowSize)

          new DataSet(features, labels)
        }
        Some(dataSets)
      case _ => None
    }.flatMap(identity)

    sparkNet.fit(featuresAndLabelsRDD)
    sparkNet.getNetwork.save(modelFolder, true)
    println(s"Model saved at ${modelFolder.getAbsolutePath}")
  }
}
