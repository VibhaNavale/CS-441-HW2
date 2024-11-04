package WordEmbeddingProcessor

import com.typesafe.config.ConfigFactory
import org.apache.spark.mllib.rdd.RDDFunctions.fromRDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

object SlidingWindow {
  private val logger = LoggerFactory.getLogger(getClass)
  private val config = ConfigFactory.load()
  private val windowSize: Int = config.getInt("app.windowSize") // Adjust window size based on task

  def loadEmbeddings(filePath: String, spark: SparkSession): Map[String, Array[Double]] = {
    try {
      val embeddingsRDD = spark.sparkContext.textFile(filePath)
      val embeddings = embeddingsRDD
        .map { line =>
          val parts = line.split("\\t")
          val token = parts(0)
          val vector = parts(1).split(",").map(_.toDouble)
          (token, vector)
        }
        .collectAsMap()
        .toMap
      logger.info(s"Successfully loaded embeddings from $filePath")
      embeddings
    } catch {
      case e: Exception =>
        logger.error(s"Error loading embeddings from $filePath", e)
        throw e
    }
  }

  def loadTokens(filePath: String, spark: SparkSession): List[String] = {
    spark.sparkContext.textFile(filePath)
      .map(_.split("\\s+").head)
      .collect()
      .toList
  }

  def createSlidingWindows(tokens: List[String], embeddings: Map[String, Array[Double]], spark: SparkSession): RDD[(Array[String], Array[Array[Double]])] = {
    val tokensRDD = spark.sparkContext.parallelize(tokens)
    val slidingWindows = tokensRDD.sliding(windowSize)
      .map { windowTokens =>
        val embeddedVectors = windowTokens.flatMap(embeddings.get)
        if (embeddedVectors.length == windowSize) {
          val positionalEmbeddings = computePositionalEmbedding(windowSize)
          val positionAwareEmbedding = embeddedVectors.zip(positionalEmbeddings).map { case (vec, pos) =>
            vec.zip(pos).map { case (v, p) => v + p }
          }
          // Log the shape of the created window
          // logger.info(s"Created window with tokens: ${windowTokens.mkString(", ")}; Shape: (${windowTokens.length}, ${positionAwareEmbedding.length}, ${positionAwareEmbedding.head.length})")
          (windowTokens, positionAwareEmbedding)
        } else {
          (Array.empty[String], Array.empty[Array[Double]])
        }
      }
      .filter { case (_, vectors) => vectors.nonEmpty }

    slidingWindows
  }

  def computePositionalEmbedding(windowSize: Int): List[Array[Double]] = {
    val embeddingDim = config.getInt("app.embeddingDimensions")
    (0 until windowSize).map { pos =>
      (0 until embeddingDim).map { i =>
        if (i % 2 == 0) Math.sin(pos / Math.pow(10000, (2.0 * i) / embeddingDim))
        else Math.cos(pos / Math.pow(10000, (2.0 * (i - 1)) / embeddingDim))
      }.toArray
    }.toList
  }
}
