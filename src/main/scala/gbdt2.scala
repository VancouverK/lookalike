package com.test

import java.io.{BufferedReader, InputStreamReader}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.不包括搜索行为
  */
object gbdt2 {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  val seed_file1 = "./seed1.txt"

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("platformTest")
    sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    hiveContext = new HiveContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    //seed
    val seed = get_seed()
    //users
    val sql_1 = "select imei, sex, marriage_status, is_parent, mz_apps_car_owner, user_job, user_age, " +
      "user_life_city_lev, " +
      "app_contact_tag, app_education_tag, app_finance_tag, app_games_tag, app_health_tag, app_interact_tag, app_music_tag, " +
      "app_o2o_tag, app_read_tag, app_shopping_tag, app_travel_tag,  app_video_tag " +
      "keyword_game_top_7d, keyword_music_top_7d, keyword_read_top_7d, keyword_search_top_7d, keyword_appstore_top_7d, keyword_video_top_7d, keyword_browsers_top_7d,  keyword_life_top_7d ,  keyword_theme_top_7d "+
      "keyword_all_top_1d, keyword_all_top_7d " +
      "from user_profile.idl_fdt_dw_tag limit " + args(0)
    val users = load_data(sql_1)
    training_reg(users)

  }

  def get_seed(): Array[String] = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val path1 = new Path(seed_file1)
    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    println("***********************reader2*****************************")
    var seed_id = new ArrayBuffer[String]()
    var line1 = reader1.readLine()
    while (line1 != null) {
      if (!line1.equals("null")) {
        seed_id += line1
      }
      line1 = reader1.readLine()
    }
    seed_id.toArray
  }

  def training_reg(data: RDD[LabeledPoint]): Unit = {

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Regression use SquaredError by default.
    var boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(3) //Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.setMaxDepth(5)
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //      boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("***********************reg finished size=" + labelsAndPredictions.count() + "*****************************")
    for ((label, pre) <- labelsAndPredictions.take(10)) {
      println("*****************************(" + label + "," + pre + ")**********************************")
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)
    //    println("Learned regression GBT model:\n" + model.toDebugString)

    // Save and load model
    //    model.save(sc, "target/tmp/myGradientBoostingRegressionModel")
    //    val sameModel = GradientBoostedTreesModel.load(sc,
    //      "target/tmp/myGradientBoostingRegressionModel")
  }


  def load_data(sql_1: String): RDD[LabeledPoint] = {

    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")

    val data_rdd1 = df.map(r => (r.getLong(0), r.getString(1), r.getString(2).toDouble, r.getString(3).toDouble, r.getString(4).toDouble, r.getString(5).toDouble, r.getString(6).toDouble, r.getString(7).toDouble,
      r.getString(8).split(","), r.getString(9).split(","), r.getString(10).split(","), r.getString(11).split(","), r.getString(12).split(","), r.getString(13).split(","), r.getString(14).split(","),
      r.getString(15).split(","), r.getString(16).split(","), r.getString(17).split(","), r.getString(18).split(","), r.getString(19).split(",")
    ))
    //兴趣爱好是1-61
    val data_rdd2 = data_rdd1.map(r => (r._1, r._2, r._3, r._4, r._5, r._6, r._7, r._8, (r._9 ++ r._10 ++ r._11 ++ r._12 ++ r._13 ++ r._14 ++ r._15 ++ r._16 ++ r._17 ++ r._18 ++ r._19 ++ r._20).toSet
      .filter(r => (!r.equals("") && !r.equals("-999"))).toArray))
    //sex*2,marr*2,pare*2,car*2,job*10, age*5, city*6
    val lable_feature_rdd = data_rdd2.map(r => {
      var sex = new Array[Double](2)
      if ("male".equals(r._2)) {
        sex(0) = 1
      } else if ("female".equals(r._2)) {
        sex(1) = 1
      }
      var marr = new Array[Double](2)
      if (r._3 == 2) {
        marr(0) = 1
      } else if (r._3 == 1) {
        marr(1) = 1
      }
      var pare = new Array[Double](2)
      if (r._4 == 2) {
        pare(0) = 1
      } else if (r._4 == 1) {
        pare(1) = 1
      }
      var car = new Array[Double](2)
      if (r._5 == 1) {
        car(0) = 1
      } else if (r._5 == 0) {
        car(1) = 0
      }
      var job = new Array[Double](10)
      if (r._6 >= 0 && r._6 < 10) {
        job(r._6.toInt) = 1
      }
      var age = new Array[Double](5)
      if (r._7 >= 0 && r._7 < 5) {
        age(r._7.toInt) = 1
      }
      var city = new Array[Double](6)
      if (r._8 >= 1 && r._8 <= 6) {
        city(r._8.toInt - 1) = 1
      }

      val array1 = sex ++ marr ++ pare ++ car ++ job ++ age ++ city
      val array2 = new Array[Double](61)
      for (index <- r._9) {
        array2(index.toInt - 1) = 1
      }
      val dv = new DenseVector(array1 ++ array2)
      val label = Math.random()
      if (label > 0.5) {
        (1, dv)
      } else {
        (0, dv)
      }

    })
    val label_array = lable_feature_rdd.map(r => r._1).take(100)
    for (i <- label_array) {
      print(i + ", ")
    }
    println()
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }

}
