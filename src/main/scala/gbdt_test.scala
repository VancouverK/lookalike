package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.udf

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.用来测试资讯push
  */
object gbdt_test {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  var sqlContext: SQLContext = null
  //  val seed_file = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  val partition_num = 200
  val feature_col = "/tmp/xueyuan/feature_col.txt"
  val oriFeatureMap = new mutable.HashMap[String, Int]()

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("xueyuan_lookalike")
    sc = new SparkContext(sparkConf)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    hiveContext = new HiveContext(sc)
    sqlContext = new SQLContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val feature = Array("sex", "marriage_status")
    //load_feature()
    val user_feature = load_data(feature).take(10)
    for (row <- user_feature) {
      //      print("imei:" + row.fieldIndex("imei") + ",")
      //      for (f <- feature) {
      //        print(f + ":" + row.toSeq + ",")
      //      }
      println(row.toString())

    }

    for ((k, v) <- oriFeatureMap) {
      println(k + ":" + v)
    }


  }


  def load_feature() = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val path1 = new Path(feature_col)
    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    var feature = new ArrayBuffer[String]()
    var line1 = reader1.readLine()
    while (line1 != null) {
      if (!line1.equals("null")) {
        feature += line1.trim
      }
      line1 = reader1.readLine()
    }
    feature.toArray
  }

  def load_data(feature: Array[String]) = {

    val feature_string = feature.mkString(",")
    val feature_num = feature.length
    val sql_1 = "select imei," + feature_string + " from user_profile.idl_fdt_dw_tag"
    //    var df = hiveContext.sql(sql_1).cache()

    var df = {
      sqlContext.createDataFrame(Seq(
        (0, "male,other", 5),
        (1, "female", 6),
        (2, "female", 5),
        (3, "female", 7)
      )).toDF("imei", "sex", "marriage_status")
    }
    println("***********************load_data finished" + df.count() + "*****************************")

    df.withColumn("imei", df.col("imei"))
    for (col <- feature) {
      df = oneColProcess(col)(df)
    }
    println("***********************df_size" + df.count() + "*****************************")
    df
  }

  //对于非cat类型的字段进行分段处理, 没有涉及到对double类型的划分
  def oneColProcessWithSplit(col: String, colRange: Array[Long]) = (df: DataFrame) => {
    //分段范围计算
    val splitToInt = udf[Int, Long] { w =>
      var i = 0
      while (i < colRange.length && w > colRange(i)) {
        i += 1
      }
      i
    }
    oriFeatureMap(col) = colRange.length + 1
    df.withColumn(col, splitToInt(df(col)))
  }

  def oneColProcessWithOneHot(col: String, catSize: Int) = (df: DataFrame) => {
    val stringToVector = udf[Vector, Array[Int]] { w =>
      Vectors.sparse(catSize, w, Array(1.0))
    }
    df.withColumn(col, stringToVector(df(col)))
  }

  def oneColProcess(col: String) = (df: DataFrame) => {
    val sma = df.schema
    sma(col).dataType match {
      case StringType => {
        //        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        val catMap2 = df.select(col).flatMap(_.getString(0).split(",")).distinct.filter(!"".equals(_)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap2.size
        val stringToDouble = udf[Array[Int], String] { w =>
          val arr = w.split(",").filter(!"".equals(_))
          var res = new ArrayBuffer[Int]()
          for (ele <- arr) {
            val r = catMap2(ele)
            res += r
          }
          res.toArray
          //          catMap(_)
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case LongType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[Int, Long] {
          catMap(_)
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case DoubleType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[Int, Double] {
          catMap(_)
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case IntegerType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[Int, Int] {
          catMap(_)
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
    }
  }
}
