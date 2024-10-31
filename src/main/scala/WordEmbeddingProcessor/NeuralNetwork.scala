package WordEmbeddingProcessor

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.factory.Nd4j
import scala.jdk.CollectionConverters._

class CustomDataFetcher(dataSetList: java.util.List[DataSet]) extends BaseDataFetcher {
  private var index = 0

  override def fetch(batch: Int): Unit = {
    for (i <- 0 until batch) {
      if (index < dataSetList.size()) {
        val dataSet = dataSetList.get(index)
        // Add your dataSet to your internal storage if needed
        index += 1
      }
    }
  }

  def numExamples(): Int = dataSetList.size()

  override def totalOutcomes(): Int = 50 // Replace with actual number of outcomes

  override def reset(): Unit = {
    index = 0
  }
}

object NeuralNetwork {
  def buildModel(inputSize: Int, outputSize: Int): MultiLayerNetwork = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.001))
      .list()
      .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(100)
        .activation(Activation.RELU)
        .build())
      .layer(1, new OutputLayer.Builder()
        .activation(Activation.SOFTMAX)
        .nIn(100).nOut(outputSize).build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100)) // Print score every 100 iterations
    model
  }

  def trainModel(model: MultiLayerNetwork, windows: List[(List[String], List[Array[Double]])]): Unit = {
    // Create 3D tensor features with shape [batchSize, windowSize, embeddingSize]
    val features3D = windows.collect {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == 50 =>
        // Create a 3D array for the embeddings
        val array = Nd4j.create(vectors.toArray).reshape(1, vectors.length, 50) // Shape: [1, slidingWindowSize, 50]
        array
    }

    val labels = windows.collect {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == 50 =>
        Nd4j.create(Array.ofDim[Double](1, 50)) // Adjust this based on your actual label requirement
    }

    // Create DataSet for each feature-label pair
    val dataSetList = features3D.zip(labels).map { case (f, l) => new DataSet(f, l) }.asJava

    // Use CustomDataFetcher for loading the dataset
    val fetcher = new CustomDataFetcher(dataSetList)

    // Create a DataSetIterator with the appropriate batch size
    val dataSetIterator: DataSetIterator = new BaseDatasetIterator(1, dataSetList.size(), fetcher)

    // Train the model with the dataset iterator
    model.fit(dataSetIterator)
  }
}
