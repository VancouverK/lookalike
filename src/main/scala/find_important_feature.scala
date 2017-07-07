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
object find_important_feature {
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
    var samp = 0.05
    var save = false
    var table_out = ""
    if (args.length >= 4) {
      uxip_boot_users_cycle = args(0).toInt
      println("***********************uxip_boot_users_cycle=" + uxip_boot_users_cycle + "*****************************")
      samp = args(1).toDouble
      println("***********************samp=" + samp + "*****************************")
      table_out = args(2)
      println("***********************table_out=" + table_out + "*****************************")
      save = args(3).toBoolean
      println("***********************save=" + save + "*****************************")
    }
    //seed
    val seed = get_seed()
    val seed_size = seed.length
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed_size + "*****************************")
    for (s <- seed.take(10)) {
      print(s + ", ")
    }
    println()
    //users

    //all user = seed + other
    val user_feature = load_data_onehot(uxip_boot_users_cycle)
    user_feature.cache()
    val user_size = user_feature.count()
    //seed
    val seeduser_feature = user_feature.filter(r => seed.contains(r._1.toString)).map(r => (r._1, r._2))
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature.count() + "*****************************")
    val feature = seeduser_feature.map(r => {
      var result = new ArrayBuffer[(Int, Int)]()
      for (i <- 0 until r._2.length) {
        val f = r._2(i)
        result += ((i, f))
      }
      (r._1, result.toArray)
    })
    val feature_thr = samp * seed_size
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature_thr=" + feature_thr + "*****************************")
    val index_featurecount = feature.flatMap(r => r._2).reduceByKey(_ + _).filter(r => r._2 > feature_thr)


    val index = index_featurecount.map(r => r._1).collect()


    val feature_samp = user_feature.map(r => {
      var result = new ArrayBuffer[Int]()
      for (i <- 0 until r._2.length) {
        if (index.contains(i)) {
          result += r._2(i)
        }
      }
      (r._1, result.toArray, r._3)
    })
    save_feature(table_out, feature_samp)


  }

  def save_feature(table_out: String, data: RDD[(Long, Array[Int], String)]): Unit = {

    val user_feature = data.map { r =>
      var f_string = ""
      for (f <- r._2) {
        f_string += f + ","
      }
      val size = f_string.length
      if (size >= 2) {
        f_string = f_string.substring(0, size - 1)
      }
      (r._1, f_string, r._3)
    }
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val candidate_rdd = user_feature.map(r => Row(r._1, r._2, r._3))

    val structType = StructType(
      StructField("imei", LongType, false) ::
        StructField("feature", StringType, false) ::
        StructField("uxip_boot_users_cycle", StringType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei bigint,feature string,uxip_boot_users_cycle string ) partitioned by (stat_date bigint) stored as textfile"
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


  def load_data_onehot(uxip_boot_users_cycle: Int): RDD[(Long, Array[Int], String)] = {
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val sql_1 = "select imei,feature,uxip_boot_users_cycle from algo.lookalike_feature_onehot where stat_date=" + date1
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    val feature_rdd = df.map(r => (r.getLong(0), r.getString(1).split(",").map(r1 => r1.toInt), r.getString(2)))
    feature_rdd
  }


}
