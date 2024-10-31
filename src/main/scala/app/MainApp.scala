package app

import WordEmbeddingProcessor.{NeuralNetwork, SlidingWindow}
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

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
      val windows = SlidingWindow.createSlidingWindows(tokens, embeddings)

      val inputSize = embeddings.head._2.length
      val outputSize = inputSize
      val model = NeuralNetwork.buildModel(inputSize, outputSize)

      NeuralNetwork.trainModel(model, windows)
      logger.info("Model training completed.")

      saveModel(model, outputPath, spark)
    } finally {
      spark.stop()
    }
  }

  private def saveModel(model: Any, outputPath: String, spark: SparkSession): Unit = {
    // Add logic to save the model
    logger.info(s"Model saved at $outputPath")
  }
}
