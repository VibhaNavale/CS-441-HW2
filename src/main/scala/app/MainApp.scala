package app

import WordEmbeddingProcessor.{NeuralNetwork, SlidingWindow}
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.util.ModelSerializer
import org.slf4j.LoggerFactory

import java.io.File

object MainApp {
  private val config = ConfigFactory.load()
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val isLocal = args.contains("--local")
    val isHdfs = args.contains("--hdfs")
    val isEmr = args.contains("--emr")

    if (List(isLocal, isHdfs, isEmr).count(identity) > 1) {
      logger.error("Specify only one mode: --local, --hdfs, or --emr")
      System.exit(1)
    }

    val (inputPath, tokensPath, outputPath) = if (isEmr) {
      (
        config.getString("app.s3InputPath"),
        config.getString("app.s3TokensPath"),
        config.getString("app.s3OutputPath")
      )
    } else if (isHdfs) {
      (
        config.getString("app.hdfsInputPath"),
        config.getString("app.hdfsTokensPath"),
        config.getString("app.hdfsOutputPath")
      )
    } else {
      (
        config.getString("app.localInputPath"),
        config.getString("app.localTokensPath"),
        config.getString("app.localOutputPath")
      )
    }

    // Initialize Spark session based on the mode
    val spark = if (isEmr || isHdfs) {
      SparkSession.builder.appName("EmbeddingApp").getOrCreate()
    } else {
      SparkSession.builder.appName("EmbeddingApp").master("local[*]").getOrCreate()
    }

    try {
      logger.info("Starting loading and processing embeddings.")
      val embeddings = SlidingWindow.loadEmbeddings(inputPath, spark)
      logger.info(s"Loaded ${embeddings.size} embeddings from $inputPath")

      val tokens = SlidingWindow.loadTokens(tokensPath, spark)
      val windows = SlidingWindow.createSlidingWindows(tokens, embeddings, spark)

     //  val inputSize = embeddings.head._2.length
      val outputSize = embeddings.head._2.length

      // Pass the SparkContext to buildModel
      val model = NeuralNetwork.buildModel(spark.sparkContext, outputSize)

      logger.info("Model summary:\n" + model.getNetwork.summary())

      // Train the model with the Spark session and the RDD of windows
      NeuralNetwork.trainModel(spark, model, windows, outputSize)

      saveModel(model, outputPath)

      // Generate text using the trained model
      val generatedSentence = TextGenerator.generateSentence("romantic", model, 10, embeddings)
      println(s"Generated Sentence: $generatedSentence")
    } finally {
      spark.stop()
    }
  }

  private def saveModel(model: SparkDl4jMultiLayer, outputPath: String): Unit = {
    val modelFile = new File(outputPath, "trained_model.zip")
    ModelSerializer.writeModel(model.getNetwork, modelFile, true) // Save the underlying MultiLayerNetwork
    logger.info(s"Model saved at ${modelFile.getAbsolutePath}")
  }
}
