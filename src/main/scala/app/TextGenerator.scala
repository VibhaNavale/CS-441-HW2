package app

import org.apache.spark.rdd.RDD
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object TextGenerator {

  def generateNextWord(context: Array[String], model: SparkDl4jMultiLayer, embeddings: Map[String, Array[Double]]): String = {
    val contextEmbedding = tokenizeAndEmbed(context, embeddings)
    val output = model.getNetwork.rnnTimeStep(contextEmbedding)
    val predictedWordIndex = Nd4j.argMax(output, 1).getInt(0)
    convertIndexToWord(predictedWordIndex, embeddings)
  }

  def generateSentence(seedText: String, model: SparkDl4jMultiLayer, maxWords: Int, embeddings: Map[String, Array[Double]]): String = {
    val generatedText = new StringBuilder(seedText)
    var context = seedText.split(" ")
    for (_ <- 0 until maxWords) {
      val nextWord = generateNextWord(context, model, embeddings)
      generatedText.append(" ").append(nextWord)
      context = generatedText.toString.split(" ")
      if (nextWord == "." || nextWord == "END") return generatedText.toString
    }
    generatedText.toString
  }

  private def tokenizeAndEmbed(words: Array[String], embeddings: Map[String, Array[Double]]): INDArray = {
    val vectors = words.flatMap(embeddings.get)
    if (vectors.isEmpty) {
      Nd4j.zeros(1, 50)
    } else {
      val concatenated = vectors.flatten
      if (concatenated.length == 50) {
        Nd4j.create(concatenated).reshape(1, 50)
      } else {
        Nd4j.zeros(1, 50)
      }
    }
  }

  private def convertIndexToWord(index: Int, embeddings: Map[String, Array[Double]]): String = {
    val vocabulary = embeddings.keys.toArray
    vocabulary(index % vocabulary.length)
  }

  def parallelGenerateSentences(seedTexts: RDD[String], model: SparkDl4jMultiLayer, maxWords: Int, embeddings: Map[String, Array[Double]]): RDD[String] = {
    seedTexts.map(seedText => generateSentence(seedText, model, maxWords, embeddings))
  }
}
