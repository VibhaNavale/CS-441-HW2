package WordEmbeddingProcessor

import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.{ScoreIterationListener, PerformanceListener, EvaluativeListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scala.jdk.CollectionConverters._

class CustomDataFetcher(dataSetList: java.util.List[DataSet]) extends BaseDataFetcher {
  private var index = 0
  var features: org.nd4j.linalg.api.ndarray.INDArray = _
  var labels: org.nd4j.linalg.api.ndarray.INDArray = _

  override def fetch(batchSize: Int): Unit = {
    cursor = index
    totalExamples = dataSetList.size()

    // Set the features and labels based on the current dataset at index
    val dataSet = dataSetList.get(index)
    features = dataSet.getFeatures
    labels = dataSet.getLabels
    index += 1
  }

  override def totalOutcomes(): Int = if (totalExamples > 0) dataSetList.get(0).getLabels.size(1).toInt else 0
  override def reset(): Unit = { index = 0 }
}

object NeuralNetwork {
  private val logger = LoggerFactory.getLogger(getClass)

  def buildModel(inputSize: Int, outputSize: Int): MultiLayerNetwork = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
      .list()
      .layer(0, new LSTM.Builder().nIn(inputSize).nOut(100)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(100).nOut(outputSize)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    // Create your dataSetIterator
    val dataSetList = new java.util.ArrayList[DataSet]() // Create an empty list for datasets
    val fetcher = new CustomDataFetcher(dataSetList)
    val dataSetIterator = new BaseDatasetIterator(1, dataSetList.size(), fetcher)

    // Listeners for various metrics
    model.setListeners(
      new ScoreIterationListener(10),          // Logs loss every 10 iterations
      new PerformanceListener(1, true),        // Logs performance (time and memory) every iteration
      new EvaluativeListener(dataSetIterator, 1) // Use dataSetIterator for evaluation
    )
    model
  }

  def trainModel(model: MultiLayerNetwork, windowsRDD: RDD[(Array[String], Array[Array[Double]])]): Unit = {
    val featuresAndLabelsRDD = windowsRDD.map {
      case (_, vectors) if vectors.nonEmpty && vectors.head.length == 50 =>
        val features = Nd4j.create(vectors).reshape(1, vectors.length, 50)
        val labels = Nd4j.zeros(1, vectors.length, 50) // Adjust based on your labels
        new DataSet(features, labels)
    }

    val dataSetList = featuresAndLabelsRDD.collect().toList.asJava
    val fetcher = new CustomDataFetcher(dataSetList)
    val dataSetIterator = new BaseDatasetIterator(1, dataSetList.size(), fetcher)

    val startTime = System.currentTimeMillis()
    model.fit(dataSetIterator)
    val endTime = System.currentTimeMillis()

    // Time per Epoch
    logger.info(s"Epoch time: ${endTime - startTime}ms")

    println("Training complete.")
  }
}
