package WordEmbeddingProcessor

import org.apache.spark.mllib.rdd.RDDFunctions.fromRDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

object SlidingWindow {
  private val windowSize: Int = 3 // Adjust window size based on task

  def loadEmbeddings(filePath: String, spark: SparkSession): Map[String, Array[Double]] = {
    val embeddingsRDD = spark.sparkContext.textFile(filePath)
    embeddingsRDD
      .map { line =>
        val parts = line.split("\\t")
        val token = parts(0)
        val vector = parts(1).split(",").map(_.toDouble)
        (token, vector)
      }
      .collectAsMap()
      .toMap
  }

  def loadTokens(filePath: String, spark: SparkSession): List[String] = {
    spark.sparkContext.textFile(filePath)
      .map(_.split("\\s+").head)
      .collect()
      .toList
  }

  def createSlidingWindows(tokens: List[String], embeddings: Map[String, Array[Double]], spark: SparkSession): RDD[(Array[String], Array[Array[Double]])] = {
    val tokensRDD = spark.sparkContext.parallelize(tokens)
    tokensRDD.sliding(windowSize)
      .map { windowTokens =>
        val embeddedVectors = windowTokens.flatMap(embeddings.get)
        if (embeddedVectors.length == windowSize) {
          val positionalEmbeddings = computePositionalEmbedding(windowSize)
          val positionAwareEmbedding = embeddedVectors.zip(positionalEmbeddings).map { case (vec, pos) => vec.zip(pos).map { case (v, p) => v + p } }
          (windowTokens, positionAwareEmbedding)
        } else {
          (Array.empty[String], Array.empty[Array[Double]])
        }
      }
      .filter { case (_, vectors) => vectors.nonEmpty }
  }

  def computePositionalEmbedding(windowSize: Int): List[Array[Double]] = {
    val embeddingDim = 50
    (0 until windowSize).map { pos =>
      (0 until embeddingDim).map { i =>
        if (i % 2 == 0) Math.sin(pos / Math.pow(10000, (2.0 * i) / embeddingDim))
        else Math.cos(pos / Math.pow(10000, (2.0 * (i - 1)) / embeddingDim))
      }.toArray
    }.toList
  }
}
