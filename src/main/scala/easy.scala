package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.
  */
object easy {
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
    var numIterations = 3
    var maxDepth = 5
    var learningRate = 1.0
    var sim = 0.5
    var topn = 10
    var table_in = ""
    var save = false
    var table_out = ""
    if (args.length >= 10) {
      uxip_boot_users_cycle = args(0).toInt
      println("***********************uxip_boot_users_cycle=" + uxip_boot_users_cycle + "*****************************")
      samp = args(1).toDouble
      println("***********************samp=" + samp + "*****************************")
      numIterations = args(2).toInt
      println("***********************numIterations=" + numIterations + "*****************************")
      maxDepth = args(3).toInt
      println("***********************maxDepth=" + maxDepth + "*****************************")
      learningRate = args(4).toDouble
      println("***********************learningRate=" + learningRate + "*****************************")
      sim = args(5).toDouble
      println("***********************sim=" + sim + "*****************************")
      topn = args(6).toInt
      println("***********************topn=" + topn + "*****************************")
      table_in = args(7)
      println("***********************table_in=" + table_in + "*****************************")
      save = args(8).toBoolean
      println("***********************save=" + save + "*****************************")
      table_out = args(9)
      println("***********************table_out=" + table_out + "*****************************")
    }
    //seed
    val seed = get_seed()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed.length + "*****************************")
    for (s <- seed.take(10)) {
      print(s + ", ")
    }
    println()
    //users

    //all user = seed + other
    val user_feature = load_data_onehot(table_in, uxip_boot_users_cycle)
    user_feature.cache()
    //
    val seeduser_feature = user_feature.filter(r => seed.contains(r._1.toString))
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature.count() + "*****************************")
    val weight = get_weight(seeduser_feature).zip(get_weight(user_feature)).map(r => (r._1.toDouble / r._2))
    var max = weight.max()
    val weight_norl = weight.map(r => r / max)
    val w_br = sc.broadcast(weight_norl.collect())


    //seeduser_feature
    val pre_seed = seeduser_feature.map(r => (r._1, {
      val w = w_br.value
      val array = r._2
      val array_length = array.length
      var result = 0.0
      for (i <- 0 until array_length) {
        val wi = w(i)
        result += (array(i) * wi)
      }
      result
    }, 1.0))
    val pre_other = user_feature.sample(false, samp).map(r => (r._1, {
      val w = w_br.value
      val array = r._2
      val array_length = array.length
      var result = 0.0
      for (i <- 0 until array_length) {
        val wi = w(i)
        result += (array(i) * wi)
      }
      result
    }, 0.0))
    val scoreAndLabels = (pre_seed ++ pre_other).map(r => (r._2, r._3))
    scoreAndLabels.cache()
    for ((p, l) <- scoreAndLabels.take(100)) {
      println(p + "," + l)
    }
    val TP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 1)).count()
    val FP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 0)).count()
    val FN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 1)).count()
    val TN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 0)).count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************TP = " + TP + ", FP = " + FP + ", FN = " + FN + ", TN = " + TN + "*****************************")
    if (TP + FP != 0) {
      val p = TP.toDouble / (TP + FP)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************p = " + p + "*****************************")
    }
    if (TP + FN != 0) {
      val r = TP.toDouble / (TP + FN)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************r = " + r + "*****************************")
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    var auc = metrics.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc + "*****************************")

    scoreAndLabels.unpersist()
  }

  def get_weight(user_feature: RDD[(Long, Array[Int])]): RDD[Int] = {
    val imei_index_onehotfeature_cyc = user_feature.map(r => {
      var result = new ArrayBuffer[(Int, Int)]()
      for (i <- 0 until r._2.length) {
        val f = r._2(i)
        result += ((i, f))
      }
      (r._1, result.toArray)
    })
    val weight = imei_index_onehotfeature_cyc.flatMap(r => r._2).reduceByKey(_ + _).map(r => r._2)
    weight
  }

  def get_seed(): Array[String] = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val path1 = new Path(seed_file1)
    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
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
      //      val sp1 = new SparseVector(5, Array(1, 2, 3), Array.fill(3)(1.0))
      //      val sp2 = new SparseVector(5, Array(1, 2, 3), Array.fill(3)(1.0))
      //      val sp = (sp1.toArray ++ sp2.toArray)
      //      val label = Math.random()
      //      if (label > 0.5) {
      //        (1, dv)
      //      } else {
      //        (0, dv)
      //      }
    })
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }

  def training_reg(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint], numIterations: Int, maxDepth: Int, learningRate: Double): GradientBoostedTreesModel = {
    val data = point_seed ++ point_samp
    var boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(numIterations) //Note: Use more iterations in practice.
    boostingStrategy.setLearningRate(learningRate)
    boostingStrategy.treeStrategy.setMaxDepth(maxDepth)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training start*****************************")
    val model = GradientBoostedTrees.train(data, boostingStrategy)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training end*****************************")
    val scoreAndLabels = data.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }
    scoreAndLabels.cache()
    for ((p, l) <- scoreAndLabels.take(100)) {
      println(p + "," + l)
    }
    val TP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 1)).count()
    val FP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 0)).count()
    val FN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 1)).count()
    val TN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 0)).count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************TP = " + TP + ", FP = " + FP + ", FN = " + FN + ", TN = " + TN + "*****************************")
    if (TP + FP != 0) {
      val p = TP.toDouble / (TP + FP)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************p = " + p + "*****************************")
    }
    if (TP + FN != 0) {
      val r = TP.toDouble / (TP + FN)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************r = " + r + "*****************************")
    }
    //    for ((label, pre) <- labelsAndPredictions_seed.take(10)) {
    //      println("*****************************(" + label + "," + pre + ")**********************************")
    //    }
    //    val testMSE_seed = scoreAndLabels.map { case (v, p) => math.pow((v - p), 2) }.mean()
    //    println("Test Mean Squared Error seed = " + testMSE_seed)
    //ev
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    var auc = metrics.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc + "*****************************")

    scoreAndLabels.unpersist()
    model

  }

  def pre(model: GradientBoostedTreesModel, user_feature_forpre: RDD[(Long, DenseVector)], sim: Double, topn: Int): RDD[(Long, Double)] = {
    val user_pre = user_feature_forpre.map { r =>
      val prediction = model.predict(r._2)
      (r._1, prediction)
    }
    val data = user_pre.filter(r => r._2 > sim).sortBy(r => r._2, false, 1).take(topn)
    sc.parallelize(data)
  }

  def load_data_onehot(table_in: String, uxip_boot_users_cycle: Int): RDD[(Long, Array[Int])] = {
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val sql_1 = "select imei,feature,uxip_boot_users_cycle from " + table_in + " where stat_date=" + date1
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val feature_rdd = df.map(r => (r.getLong(0), r.getString(1).split(",").map(r1 => r1.toInt), r.getString(2)))
    feature_rdd.filter(r => r._3.contains("," + uxip_boot_users_cycle + ",")).map(r => (r._1, r._2))
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
    val sdf1 = new SimpleDateFormat("yyyyMMdd")
    val date1 = sdf1.format(c1.getTime())
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
