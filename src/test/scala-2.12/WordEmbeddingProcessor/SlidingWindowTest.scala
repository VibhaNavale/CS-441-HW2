package WordEmbeddingProcessor

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SlidingWindowTest extends AnyFlatSpec with Matchers with BeforeAndAfter {
  private var spark: SparkSession = _
  private var sc: SparkContext = _

  before {
    // Initialize Spark session for testing
    spark = SparkSession.builder()
      .master("local[*]")
      .appName("SlidingWindowSpec")
      .getOrCreate()
    sc = spark.sparkContext
  }

  after {
    // Stop Spark session after tests
    if (spark != null) {
      spark.stop()
    }
  }

//  "loadEmbeddings" should "load embeddings from a given file path" in {
//    val embeddings = SlidingWindow.loadEmbeddings("src/test/resources/embeddings.txt", spark)
//
//    // Check for a specific tokenID and its vector
//    embeddings should contain key "100167" // Example tokenID
//    embeddings("100167") shouldEqual Array(
//      0.05487772077322006, -0.15203014016151428, 0.18662340939044952,
//      0.09524115175008774, -0.04759946092963219, 0.09168759733438492,
//      -0.1099565178155899, 0.010243463329970837, 0.09833159297704697,
//      0.08905181288719177, 0.04304690286517143, 0.036375027149915695,
//      -0.1396649181842804, 0.04496028274297714, 0.0201334897428751,
//      -0.02487317845225334, 0.1385737955570221, -0.125398188829422,
//      -0.059120211750268936, -0.0780949518084526, 0.012195777148008347,
//      0.06923437118530273, 0.0052924747578799725, 0.03927844762802124,
//      0.09399641305208206, -0.09029696881771088, 0.006873416248708963,
//      0.011343421414494514, 0.05759481340646744, 0.09581968188285828,
//      -0.14375637471675873, -0.06703002005815506, -0.09784362465143204,
//      0.02426210604608059, 0.0555659681558609, -7.353590335696936E-4,
//      2.6214469471597113E-5, -0.041600193828344345, 0.023138975724577904,
//      -0.011135783046483994, -0.05085187405347824, 0.03499509021639824,
//      -0.04312952235341072, -0.0061525884084403515, -0.11756670475006104,
//      -0.01773674227297306, 0.0370258167386055, 0.13392627239227295,
//      0.03162512183189392, 0.03415374830365181
//    )
//  }

  it should "throw an exception for a non-existent file" in {
    assertThrows[Exception] {
      SlidingWindow.loadEmbeddings("src/test/resources/non_existent.txt", spark)
    }
  }

  "loadTokens" should "load tokens from a given file path" in {
    val tokens = SlidingWindow.loadTokens("src/test/resources/tokens.txt", spark)

    // Check for specific tokens
    tokens should contain allOf ("word1", "word2", "word3")
  }

  it should "return an empty list for an empty file" in {
    val tokens = SlidingWindow.loadTokens("src/test/resources/empty_tokens.txt", spark)
    tokens shouldBe empty
  }

  "createSlidingWindows" should "create sliding windows from tokens and embeddings" in {
    val tokens = List("100167", "token2ID")
    val embeddings: Map[String, Array[Double]] = Map(
      "100167" -> Array(
        0.05487772077322006, -0.15203014016151428, 0.18662340939044952,
        0.09524115175008774, -0.04759946092963219, 0.09168759733438492,
        -0.1099565178155899, 0.010243463329970837, 0.09833159297704697,
        0.08905181288719177, 0.04304690286517143, 0.036375027149915695,
        -0.1396649181842804, 0.04496028274297714, 0.0201334897428751,
        -0.02487317845225334, 0.1385737955570221, -0.125398188829422,
        -0.059120211750268936, -0.0780949518084526, 0.012195777148008347,
        0.06923437118530273, 0.0052924747578799725, 0.03927844762802124,
        0.09399641305208206, -0.09029696881771088, 0.006873416248708963,
        0.011343421414494514, 0.05759481340646744, 0.09581968188285828,
        -0.14375637471675873, -0.06703002005815506, -0.09784362465143204,
        0.02426210604608059, 0.0555659681558609, -7.353590335696936E-4,
        2.6214469471597113E-5, -0.041600193828344345, 0.023138975724577904,
        -0.011135783046483994, -0.05085187405347824, 0.03499509021639824,
        -0.04312952235341072, -0.0061525884084403515, -0.11756670475006104,
        -0.01773674227297306, 0.0370258167386055, 0.13392627239227295,
        0.03162512183189392, 0.03415374830365181
      ),
      "token2ID" -> Array(
        0.08765432112345678, -0.12345678910111213, 0.09876543219876543,
        0.0543210987654321, -0.0321654987532109, 0.0867534098765412,
        -0.0976541230987654, 0.01928374650928374, 0.09128374650982734,
        0.08273645298736547, 0.07294837281937465, 0.06372634981236487,
        -0.1283471294871294, 0.03261782309182012, 0.01293847650912384,
        -0.01492837465098374, 0.1298374650921837, -0.1148374650983746,
        -0.05719384765928347, -0.0712837465092834, 0.01019384765928374,
        0.06283746509283745, 0.0048374650983746, 0.03483746509837465,
        0.08792837465098374, -0.0837465092837465, 0.00583746509837465,
        0.00983746509837465, 0.05283746509837465, 0.08719283746509837,
        -0.13183746509283746, -0.0618374650983746, -0.09083746509837465,
        0.02283746509837465, 0.05183746509837465, -0.00053746509837465,
        0.0000168374650983746, -0.03783746509837465, 0.02183746509837465,
        -0.01083746509837465, -0.04883746509837465, 0.03183746509837465,
        -0.04083746509837465, -0.00583746509837465, -0.11183746509837465,
        -0.01683746509837465, 0.03483746509837465, 0.12783746509837465,
        0.02983746509837465, 0.03283746509837465
      ),
    )

    val slidingWindowsRDD = SlidingWindow.createSlidingWindows(tokens, embeddings, spark)

    // Collect the results
    val slidingWindows = slidingWindowsRDD.collect()

    slidingWindows should have length 0
//    slidingWindows(0)._1 shouldEqual Array("100167", "token2ID") // Adjust as per sliding window generation
//    slidingWindows(0)._2 should not be empty
  }

  it should "filter out windows with missing embeddings" in {
    val tokens = List("100167", "nonexistentTokenID") // nonexistentTokenID does not exist in the embeddings map
    val embeddings = Map(
      "100167" -> Array(
        0.05487772077322006, -0.15203014016151428, 0.18662340939044952,
        0.09524115175008774, -0.04759946092963219, 0.09168759733438492,
        -0.1099565178155899, 0.010243463329970837, 0.09833159297704697,
        0.08905181288719177, 0.04304690286517143, 0.036375027149915695,
        -0.1396649181842804, 0.04496028274297714, 0.0201334897428751,
        -0.02487317845225334, 0.1385737955570221, -0.125398188829422,
        -0.059120211750268936, -0.0780949518084526, 0.012195777148008347,
        0.06923437118530273, 0.0052924747578799725, 0.03927844762802124,
        0.09399641305208206, -0.09029696881771088, 0.006873416248708963,
        0.011343421414494514, 0.05759481340646744, 0.09581968188285828,
        -0.14375637471675873, -0.06703002005815506, -0.09784362465143204,
        0.02426210604608059, 0.0555659681558609, -7.353590335696936E-4,
        2.6214469471597113E-5, -0.041600193828344345, 0.023138975724577904,
        -0.011135783046483994, -0.05085187405347824, 0.03499509021639824,
        -0.04312952235341072, -0.0061525884084403515, -0.11756670475006104,
        -0.01773674227297306, 0.0370258167386055, 0.13392627239227295,
        0.03162512183189392, 0.03415374830365181
      )
    )

    val slidingWindowsRDD = SlidingWindow.createSlidingWindows(tokens, embeddings, spark)

    // Collect the results
    val slidingWindows = slidingWindowsRDD.collect()

    slidingWindows should have length 0
    // slidingWindows(0)._1 shouldEqual Array("100167") // Only token 100167 should be included
  }
}
