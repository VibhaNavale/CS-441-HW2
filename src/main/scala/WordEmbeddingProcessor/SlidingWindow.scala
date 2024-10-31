package WordEmbeddingProcessor

import org.apache.spark.sql.SparkSession
import scala.collection.Map

object SlidingWindow {
  private val windowSize: Int = 50

  def loadEmbeddings(filePath: String, spark: SparkSession): Map[String, Array[Double]] = {
    val embeddingsRDD = spark.sparkContext.textFile(filePath)
    embeddingsRDD
      .map { line =>
        val parts = line.split("\t")
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

  def createSlidingWindows(tokens: List[String], embeddings: Map[String, Array[Double]]): List[(List[String], List[Array[Double]])] = {
    tokens.sliding(windowSize)
      .map { windowTokens =>
        val embeddedVectors = windowTokens.flatMap(embeddings.get)
        (windowTokens, embeddedVectors)
      }
      .filter { case (_, vectors) => vectors.nonEmpty }
      .toList
  }
}
