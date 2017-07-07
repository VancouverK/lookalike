package com.test

/**
  * Created by xueyuan on 2017/4/24.
  */

import java.util.regex.Pattern
import java.io._
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.hadoop.fs.{FSDataOutputStream, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}

import scala.collection.{Map, mutable}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object lr {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  val seed_file1 = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("xueyuan_lookalike")
    sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    hiveContext = new HiveContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    var uxip_boot_users_cycle = 3
    var samp = 0.1
    var sim = 0.5
    var topn = 10
    var table_out = ""
    var save = false
    if (args.length >= 4) {
      uxip_boot_users_cycle = args(0).toInt
      println("***********************uxip_boot_users_cycle=" + uxip_boot_users_cycle + "*****************************")
      samp = args(1).toDouble
      println("***********************samp=" + samp + "*****************************")
      sim = args(2).toDouble
      println("***********************sim=" + sim + "*****************************")
      topn = args(3).toInt
      println("***********************topn=" + topn + "*****************************")
      table_out = args(4)
      println("***********************table_out=" + table_out + "*****************************")
      save = args(5).toBoolean
      println("***********************save=" + save + "*****************************")
    }
    //seed
    val seed = get_seed()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed.length + "*****************************")
    for (s <- seed.take(10)) {
      print(s + ", ")
    }
    println()
    //all user = seed + other
    val user_feature = load_data_onehot(uxip_boot_users_cycle)
    //seed
    val seeduser_feature = user_feature.filter(r => seed.contains(r._1.toString))
    //    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature.count() + "*****************************")
    val point_seed = get_lable_feature_rdd(seeduser_feature, 1)
    //other
    val otheruser_feature = user_feature.filter(r => seed.contains(r._1.toString) == false)
    //    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************otheruser_feature_size=" + otheruser_feature.count() + "*****************************")
    val user_feature_forpre = otheruser_feature.map(r => (r._1, new DenseVector(r._2)))
    //sample
    val sampuser_feature = otheruser_feature.sample(false, samp)
    val point_samp = get_lable_feature_rdd(sampuser_feature, 0)
    //    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************sampuser_feature_size=" + sampuser_feature.count() + "*****************************")

    //training
    val model = training(point_seed, point_samp)
    //predict
    val result = pre(model, user_feature_forpre, sim, topn)
    //save
    save_data(table_out, result)
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
        seed_id += line1.trim
      }
      line1 = reader1.readLine()
    }
    seed_id.toArray
  }


  def get_lable_feature_rdd(user_feature: RDD[(Long, Array[Double])], label: Int): RDD[LabeledPoint] = {
    val lable_feature_rdd = user_feature.map(r => {
      val dv = new DenseVector(r._2)
      (label, dv)
    })
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }

  def training(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint]): LogisticRegressionModel = {
    val data = point_seed ++ point_samp
    val lr = new LogisticRegressionWithLBFGS().setIntercept(true).setNumClasses(2)
    lr.optimizer.setRegParam(0.6).setConvergenceTol(0.000001).setNumIterations(101).setUpdater(new SquaredL2Updater)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training start*****************************")
    val model = lr.run(data)
    //.setThreshold(0.5)
    val scoreAndLabels = data.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }
    for ((p, l) <- scoreAndLabels.take(100)) {
      println(p + "," + l)
    }
    val TP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 1)).count()
    val FP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 0)).count()
    val FN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 1)).count()
    val TN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 0)).count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************TP = " + TP + ", FP = " + FP + ", FN = " + FN + ", TN = " + TN + "*****************************")
    if (TP + FP != 0) {
      val p = TP / (TP + FP)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************p = " + p + "*****************************")
    }
    if (TP + FN != 0) {
      val r = TP / (TP + FN)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************r = " + r + "*****************************")
    }


    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    var auc = metrics.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training end*****************************")
    model
  }

  def pre(model: LogisticRegressionModel, user_feature_forpre: RDD[(Long, DenseVector)], sim: Double, topn: Int): RDD[(Long, Double)] = {
    val user_pre = user_feature_forpre.map { r =>
      val prediction = model.predict(r._2)
      (r._1, prediction)
    }
    val data = user_pre.filter(r => r._2 > sim).sortBy(r => r._2, false, 1).take(topn)
    sc.parallelize(data)
  }

  def load_data_onehot(uxip_boot_users_cycle: Int): RDD[(Long, Array[Double])] = {
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val sql_1 = "select imei,feature from algo.xueyuan_lookalike_onehot where uxip_boot_users_cycle like '%," + uxip_boot_users_cycle + ",%' and stat_date=" + date1
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val feature_rdd = df.map(r => (r.getLong(0), r.getString(1).split(",").map(r1 => r1.toDouble)))
    feature_rdd
  }

  def save_data(table_out: String, user_pre: RDD[(Long, Double)]): Unit = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val candidate_rdd = user_pre.map(r => Row(r._1, r._2))

    val structType = StructType(
      StructField("user", LongType, false) ::
        StructField("score", DoubleType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( user BIGINT,score DOUBLE ) partitioned by (stat_date bigint) stored as textfile"
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val insertInto_table_sql: String = "insert overwrite table " + table_out + " partition(stat_date = " + date1 + ") select * from "
    val table_temp = "table_temp"
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save data start*****************************")
    candidate_df.registerTempTable(table_temp)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************register TempTable finished*****************************")
    hiveContext.sql(create_table_sql)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************create table finished*****************************")
    hiveContext.sql(insertInto_table_sql + table_temp)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************insertInto table finished*****************************")
  }


}
