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
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.
  * 定时生成onehot编码，入库. online
  */
object generate_feature_onehot {
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
    var table_out = ""
    if (args.length == 1) {
      table_out = args(0)
    }
    save_feature(table_out)
  }

  def save_feature(table_out: String): Unit = {
    val user_feature2 = load_data()
    val user_feature = user_feature2.map { r =>
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

  def load_data(): RDD[(Long, Array[Int], String)] = {
    val sql_1 = "select imei, sex, marriage_status, is_parent, mz_apps_car_owner, user_job, user_age, " +
      "user_life_city_lev, " +
      "app_contact_tag, app_education_tag, app_finance_tag, app_games_tag, app_health_tag, app_interact_tag, app_music_tag, " +
      "app_o2o_tag, app_read_tag, app_shopping_tag, app_travel_tag,  app_video_tag, " +
      "uxip_boot_users_cycle " +
      "from user_profile.idl_fdt_dw_tag"
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")

    val data_rdd1 = df.map(r => (r.getLong(0), r.getString(1), r.getString(2).toInt, r.getString(3).toInt, r.getString(4).toInt, r.getString(5).toInt, r.getString(6).toInt, r.getString(7).toInt,
      r.getString(8).split(","), r.getString(9).split(","), r.getString(10).split(","), r.getString(11).split(","), r.getString(12).split(","), r.getString(13).split(","), r.getString(14).split(","),
      r.getString(15).split(","), r.getString(16).split(","), r.getString(17).split(","), r.getString(18).split(","), r.getString(19).split(","), r.getString(20)
    ))
    //兴趣爱好是
    val data_rdd2 = data_rdd1.map(r =>
      (r._1, r._2, r._3, r._4, r._5, r._6, r._7, r._8, (r._9 ++ r._10 ++ r._11 ++ r._12 ++ r._13 ++ r._14 ++ r._15 ++ r._16 ++ r._17 ++ r._18 ++ r._19 ++ r._20).toSet
        .filter(r => (!r.equals("") && !r.equals("-999"))).toArray, r._21)
    )
    //sex*2,marr*2,pare*2,car*2,job*10, age*5, city*6
    val feature_rdd = data_rdd2.map(r => {
      var sex = new Array[Int](2)
      if ("male".equals(r._2)) {
        sex(0) = 1
      } else if ("female".equals(r._2)) {
        sex(1) = 1
      }
      var marr = new Array[Int](2)
      if (r._3 == 2) {
        marr(0) = 1
      } else if (r._3 == 1) {
        marr(1) = 1
      }
      var pare = new Array[Int](2)
      if (r._4 == 2) {
        pare(0) = 1
      } else if (r._4 == 1) {
        pare(1) = 1
      }
      var car = new Array[Int](2)
      if (r._5 == 1) {
        car(0) = 1
      } else if (r._5 == 0) {
        car(1) = 1
      }
      var job = new Array[Int](10)
      if (r._6 >= 0 && r._6 <= 9) {
        job(r._6.toInt) = 1
      }
      var age = new Array[Int](5)
      if (r._7 >= 0 && r._7 <= 4) {
        age(r._7.toInt) = 1
      }
      var city = new Array[Int](6)
      if (r._8 >= 1 && r._8 <= 6) {
        city(r._8.toInt - 1) = 1
      }

      val array1 = sex ++ marr ++ pare ++ car ++ job ++ age ++ city
      val array2 = new Array[Int](61)
      for (index <- r._9) {
        array2(index.toInt - 1) = 1
      }
      (r._1, array1 ++ array2, r._10)

    }
    )
    feature_rdd

  }


}
