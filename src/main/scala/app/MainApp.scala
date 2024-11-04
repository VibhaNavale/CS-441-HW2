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
      logger.info(s"Loaded ${tokens.size} tokens from $tokensPath")

      val windows = SlidingWindow.createSlidingWindows(tokens, embeddings, spark)
      logger.info(s"Created ${windows.count()} sliding windows.")

      // Pass the SparkContext explicitly
      val outputSize = embeddings.head._2.length
      val model = NeuralNetwork.buildModel(spark.sparkContext, outputSize)

      logger.info("Model summary:\n" + model.getNetwork.summary())

      // Train the model with the Spark session and the RDD of windows
      NeuralNetwork.trainModel(spark, model, windows, outputSize)

      // Generate text using the trained model
      val generatedSentence = TextGenerator.generateSentence("Marilyn", model, 10, embeddings)
      println(s"Generated Sentence: $generatedSentence")
    } catch {
      case e: Exception =>
        logger.error("An error occurred during processing", e)
    } finally {
      spark.stop()
      logger.info("Spark session stopped.")
    }
  }
}
