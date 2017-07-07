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
object generate_seed_table {
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
    save_seed("algo.xueyuan_looklike_seed1")

  }

  def save_seed(table_out: String): Unit = {
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
    val seed = sc.parallelize(seed_id)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save seed start*****************************")
    val candidate_rdd = seed.map(r => Row(r))

    val structType = StructType(
      StructField("imei", StringType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei String ) partitioned by (stat_date bigint) stored as textfile"
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
