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
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.
  */
object getnerate_feature_by_gbdt {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  val seed_file1 = "/tmp/xueyuan/seed1.txt"
  val model_path = "/tmp/xueyuan/gbdt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  var strategy = "Regression"
  var partition_num = 200

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
    var stat_date = ""
    var seed_table = ""
    if (args.length >= 14) {
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
      stat_date = args(10)
      println("***********************stat_date=" + stat_date + "*****************************")
      strategy = args(11)
      println("***********************strategy=" + strategy + "*****************************")
      partition_num = args(12).toInt
      println("***********************partition_num=" + partition_num + "*****************************")
      seed_table = args(13)
      println("***********************seed_table=" + seed_table + "*****************************")
    }
    //seed
    val seed = get_seed(seed_table)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed.length + "*****************************")
    for (s <- seed.take(10)) {
      print(s + ", ")
    }
    println()
    //users

    //all user = seed + other
    //    val user_feature = load_data_onehot(table_in, uxip_boot_users_cycle, stat_date)
    val user_feature = load_data()
    user_feature.cache()
    //seed
    user_feature.repartition(partition_num)
    val seeduser_feature_label = user_feature.filter(r => seed.contains(r._1.toString)).mapPartitions(iter => for (r <- iter) yield (r._1, r._2, 1))
    seeduser_feature_label.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature_label.count() + "*****************************")
    val point_seed = get_lable_feature_rdd(seeduser_feature_label)
    //other
    val otheruser_feature_label = user_feature.filter(r => (seed.contains(r._1.toString) == false && r._3.contains("," + uxip_boot_users_cycle + ","))).mapPartitions(iter => for (r <- iter) yield (r._1, r._2, 0))
    otheruser_feature_label.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************otheruser_feature_size=" + otheruser_feature_label.count() + "*****************************")
    user_feature.unpersist()
    //sample
    val sampuser_feature_label = otheruser_feature_label.sample(false, samp).cache()
    val point_samp = get_lable_feature_rdd(sampuser_feature_label)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************sampuser_feature_size=" + sampuser_feature_label.count() + "*****************************")

    //training
    val model = training_reg(point_seed, point_samp, numIterations, maxDepth, learningRate)
    //    val model = GradientBoostedTreesModel.load(sc, model_path)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************model finished *****************************")
    // generate feature
    val trees = model.trees
    val user_feature_label = seeduser_feature_label ++ otheruser_feature_label
    user_feature_label.repartition(partition_num)
    val user_treeid_score_label = user_feature_label.mapPartitions(iter => {
      for (r <- iter) yield {
        val vector = new DenseVector(r._2)
        val result = new ArrayBuffer[(Int, Double)]()
        for (i <- 0 until trees.length) {
          result += ((i, trees(i).predict(vector))) //(treeid,pre_score)
        }
        (r._1, result.toArray, r._3)
      }

    })
    user_treeid_score_label.cache()
    user_treeid_score_label.repartition(partition_num)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user_treeid_score_label=" + user_treeid_score_label.count() + " ****************************")
    val treeid_scoreset = user_treeid_score_label.flatMap(r => r._2).mapPartitions(iter => for (r <- iter) yield (r._1, new HashSet[Double]() + r._2)).reduceByKey(_ ++ _)
    treeid_scoreset.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_scoreset=" + treeid_scoreset.count() + " ****************************")
    //100
    val treeid_score_scoreindex = treeid_scoreset.mapPartitions(iter => for (r <- iter) yield {
      var score_array = r._2.toArray
      var map = score_array.zipWithIndex
      (r._1, map)
    })
    treeid_score_scoreindex.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_score_scoreindex=" + treeid_score_scoreindex.count() + " ****************************")
    treeid_scoreset.unpersist()
    val treeid_score_gbdtfeature = treeid_score_scoreindex.mapPartitions(iter => for (r <- iter) yield {
      var treeid_score_gbdtfeaturearray = new ArrayBuffer[((Int, Double), Array[Int])]()
      for ((score, scoreindex) <- r._2) {
        val gbdtfeature = new Array[Int](r._2.size)
        gbdtfeature(scoreindex) = 1
        val temp = (((r._1, score), gbdtfeature))
        treeid_score_gbdtfeaturearray += temp
      }
      treeid_score_gbdtfeaturearray.toArray
    }).flatMap(r => r).collect()


    val treeid_score_gbdtfeature_br = sc.broadcast(treeid_score_gbdtfeature.toMap)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_score_gbdtfeature=" + treeid_score_gbdtfeature.length + " ****************************")
    treeid_score_scoreindex.unpersist()
    val user_onehot_label = user_treeid_score_label.mapPartitions(iter => for (r <- iter) yield {
      val map = treeid_score_gbdtfeature_br.value
      var onehot = new ArrayBuffer[Int]()
      val gbdtfeature = r._2
      for (g <- gbdtfeature) {
        val temp = map(g)
        onehot ++= temp
      }
      (r._1, onehot.toArray, r._3)
    })
    user_onehot_label.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user_onehot_label=" + user_onehot_label.count() + " ****************************")

    if (save) {

      save_data(table_out, user_onehot_label)
    }

  }

  def get_seed(table_in: String): Array[String] = {
    //    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    //    val path1 = new Path(seed_file1)
    //    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    //    var seed_id = new ArrayBuffer[String]()
    //    var line1 = reader1.readLine()
    //    while (line1 != null) {
    //      if (!line1.equals("null")) {
    //        seed_id += line1.trim
    //      }
    //      line1 = reader1.readLine()
    //    }
    //    seed_id.toArray
    val sql_1 = "select imei from " + table_in
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    df.repartition(partition_num)
    val seed_rdd = df.mapPartitions(iter => for (r <- iter) yield {
      r.getString(0)
    })
    seed_rdd.collect()
  }


  def get_lable_feature_rdd(user_feature: RDD[(Long, Array[Double], Int)]): RDD[LabeledPoint] = {
    val lable_feature_rdd = user_feature.map(r => {
      val dv = new DenseVector(r._2)
      (r._3, dv)
    })
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }

  def training_reg(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint], numIterations: Int, maxDepth: Int, learningRate: Double): GradientBoostedTreesModel = {
    val data = point_seed ++ point_samp
    var boostingStrategy = BoostingStrategy.defaultParams(strategy) //"Regression"
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

    //    model.save(sc, model_path)


    val weight = model.treeWeights
    for (w <- weight) {
      print(w + ", ")
    }
    println()
    println("numTreesf=" + model.numTrees)
    //    println("Learned regression GBT model:\n" + model.toDebugString)
    model

  }


  def load_data_onehot(table_in: String, uxip_boot_users_cycle: Int, stat_date: String): RDD[(Long, Array[Double], String)] = {
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val sql_1 = "select imei,feature,uxip_boot_users_cycle from " + table_in + " where stat_date=" + stat_date
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val feature_rdd = df.map(r => (r.getLong(0), r.getString(1).split(",").map(r1 => r1.toDouble), r.getString(2)))
    feature_rdd
  }

  def load_data(): RDD[(Long, Array[Double], String)] = {
    val sql_1 = "select imei, sex, marriage_status, is_parent, mz_apps_car_owner, user_job, user_age, " +
      "user_life_city_lev, " +
      "app_contact_tag, app_education_tag, app_finance_tag, app_games_tag, app_health_tag, app_interact_tag, app_music_tag, " +
      "app_o2o_tag, app_read_tag, app_shopping_tag, app_travel_tag,  app_video_tag, uxip_boot_users_cycle " +
      "keyword_game_top_7d, keyword_music_top_7d, keyword_read_top_7d, keyword_search_top_7d, keyword_appstore_top_7d, keyword_video_top_7d, keyword_browsers_top_7d,  keyword_life_top_7d ,  keyword_theme_top_7d " +
      "keyword_all_top_1d, keyword_all_top_7d " +
      "from user_profile.idl_fdt_dw_tag"
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")

    val data_rdd1 = df.map(r => (r.getLong(0), r.getString(1), r.getString(2).toDouble, r.getString(3).toDouble, r.getString(4).toDouble, r.getString(5).toDouble, r.getString(6).toDouble, r.getString(7).toDouble,
      r.getString(8).split(","), r.getString(9).split(","), r.getString(10).split(","), r.getString(11).split(","), r.getString(12).split(","), r.getString(13).split(","), r.getString(14).split(","),
      r.getString(15).split(","), r.getString(16).split(","), r.getString(17).split(","), r.getString(18).split(","), r.getString(19).split(","), r.getString(20)
    ))
    //兴趣爱好是
    val data_rdd2 = data_rdd1.map(r => (r._1, r._2, r._3, r._4, r._5, r._6, r._7, r._8, (r._9 ++ r._10 ++ r._11 ++ r._12 ++ r._13 ++ r._14 ++ r._15 ++ r._16 ++ r._17 ++ r._18 ++ r._19 ++ r._20).toSet
      .filter(r => (!r.equals("") && !r.equals("-999"))).toArray, r._21))
    //sex*2,marr*2,pare*2,car*2,job*10, age*5, city*6
    val feature_rdd = data_rdd2.map(r => {
      //      var sex = new Array[Double](2)
      //      if ("male".equals(r._2)) {
      //        sex(0) = 1
      //      } else if ("female".equals(r._2)) {
      //        sex(1) = 1
      //      }
      //      var marr = new Array[Double](2)
      //      if (r._3 == 2) {
      //        marr(0) = 1
      //      } else if (r._3 == 1) {
      //        marr(1) = 1
      //      }
      //      var pare = new Array[Double](2)
      //      if (r._4 == 2) {
      //        pare(0) = 1
      //      } else if (r._4 == 1) {
      //        pare(1) = 1
      //      }
      //      var car = new Array[Double](2)
      //      if (r._5 == 1) {
      //        car(0) = 1
      //      } else if (r._5 == 0) {
      //        car(1) = 1
      //      }
      //      var job = new Array[Double](10)
      //      if (r._6 >= 0 && r._6 < 10) {
      //        job(r._6.toInt) = 1
      //      }
      //      var age = new Array[Double](5)
      //      if (r._7 >= 0 && r._7 < 5) {
      //        age(r._7.toInt) = 1
      //      }
      //      var city = new Array[Double](6)
      //      if (r._8 >= 1 && r._8 <= 6) {
      //        city(r._8.toInt - 1) = 1
      //      }
      //
      //      val array1 = sex ++ marr ++ pare ++ car ++ job ++ age ++ city
      var array1 = new Array[Double](7)
      if ("male".equals(r._2)) {
        array1(0) = 1
      } else if ("female".equals(r._2)) {
        array1(0) = 2
      }
      array1(1) = r._3
      array1(2) = r._4
      array1(3) = r._5
      array1(4) = r._6
      array1(5) = r._7
      array1(6) = r._8

      val array2 = new Array[Double](61)
      for (index <- r._9) {
        array2(index.toInt - 1) = 1
      }
      (r._1, array1 ++ array2, r._10)


    })
    feature_rdd

  }

  def save_data(table_out: String, user_onehot_label: RDD[(Long, Array[Int], Int)]): Unit = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val user_onehot_label_forsave = user_onehot_label.map(r => {
      var onehot = ""
      for (i <- r._2) {
        onehot += i + ","
      }
      if (onehot.length > 2) {
        onehot = onehot.substring(0, onehot.length - 1)
      }
      (r._1, onehot, r._3)
    })
    val candidate_rdd = user_onehot_label_forsave.map(r => Row(r._1, r._2, r._3))

    val structType = StructType(
      StructField("imei", LongType, false) ::
        StructField("feature", StringType, false) ::
        StructField("label", IntegerType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei bigint,feature string,label int ) partitioned by (stat_date bigint) stored as textfile"
    val c1 = Calendar.getInstance()
    //    c1.add(Calendar.DATE, -1)
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
