import sbt.*
import sbtassembly.AssemblyPlugin.autoImport._

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"

enablePlugins(AssemblyPlugin)

lazy val root = (project in file("."))
  .settings(
    name := "CS-441-HW-2",

    libraryDependencies ++= Seq(
      // Spark dependencies
      "org.apache.spark" %% "spark-core" % "3.5.1",
      "org.apache.spark" %% "spark-sql" % "3.5.1",
      "org.apache.spark" %% "spark-mllib" % "3.5.1",

      // Hadoop dependencies for HDFS interaction
      "org.apache.hadoop" % "hadoop-common" % "3.3.6",
      "org.apache.hadoop" % "hadoop-client" % "3.3.6",

      // Logging dependencies
      "org.slf4j" % "slf4j-api" % "2.0.16",
      "org.slf4j" % "slf4j-simple" % "2.0.16",
      "ch.qos.logback" % "logback-classic" % "1.5.6",

      // Configuration management
      "com.typesafe" % "config" % "1.4.3",

      // Scala testing
      "org.scalatest" %% "scalatest" % "3.2.19" % Test,
      "org.scalatestplus" %% "scalatestplus-mockito" % "1.0.0-M2" % Test,
      "org.mockito" %% "mockito-scala" % "1.17.37" % Test,

      // Neural network library
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.deeplearning4j" % "deeplearning4j-ui" % "1.0.0-M2.1",
      "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-api" % "1.0.0-M2.1",

      // Jackson dependencies for compatibility
      "com.fasterxml.jackson.core" % "jackson-databind" % "2.15.2",
      "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.15.2",
    ),

    resolvers ++= Seq(
      "Maven Central" at "https://repo1.maven.org/maven2/",
      "jitpack.io" at "https://jitpack.io"
    ),

    assembly / assemblyJarName := "CS-441-HW-2-assembly-0.1.jar",

    assembly / mainClass := Some("app.MainApp"),

    // sbt-assembly settings for creating a fat jar
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) =>
        xs match {
          case "MANIFEST.MF" :: Nil => MergeStrategy.discard
          case "services" :: _ => MergeStrategy.concat
          case _ => MergeStrategy.discard
        }
      case "reference.conf" => MergeStrategy.concat
      case x if x.endsWith(".proto") => MergeStrategy.rename
      case _ => MergeStrategy.first
    }
  )
