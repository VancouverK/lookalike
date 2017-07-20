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
  * 定时生成onehot编码，入库. 过滤掉不重要的属性和属性太少的用户
  */
object generate_feature_onehot_filter {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  val seed_file1 = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  var uxip_boot_users_cycle = 7
  var samp_feature = 0.05
  var feature_num = 10

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
    var table_out_filter = ""
    if (args.length == 5) {
      table_out = args(0)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************table_out=" + table_out + "*****************************")
      table_out_filter = args(1)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************table_out_filter=" + table_out_filter + "*****************************")
      uxip_boot_users_cycle = args(2).toInt
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************uxip_boot_users_cycle=" + uxip_boot_users_cycle + "*****************************")
      samp_feature = args(3).toDouble
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************samp_feature=" + samp_feature + "*****************************")
      feature_num = args(4).toInt
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature_num=" + feature_num + "*****************************")
    }
    val imei_onehotfeature_cyc = load_data_new()
    imei_onehotfeature_cyc.cache()
    save_feature(table_out, imei_onehotfeature_cyc)
    //one-hot
    //filter by uxip_boot_users_cycle
    val imei_onehotfeature_cyc_filter0 = imei_onehotfeature_cyc.filter(r => r._3.contains("," + uxip_boot_users_cycle + ","))
    //filter user
    val imei_onehotfeature_cyc_filter1 = imei_onehotfeature_cyc_filter0.filter(r => r._2.sum > feature_num)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************imei_onehotfeature_cyc_filter1=" + imei_onehotfeature_cyc_filter1.count() + "*****************************")
    //filter feature less than 5%
    val imei_onehotfeature_cyc_filter2 = filter_by_feature(imei_onehotfeature_cyc_filter1)


    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************imei_onehotfeature_cyc_filter2=" + imei_onehotfeature_cyc_filter2.count() + "*****************************")
    save_feature(table_out_filter, imei_onehotfeature_cyc_filter2)
  }

  def filter_by_feature(imei_onehotfeature_cyc: RDD[(Long, Array[Int], String)]): RDD[(Long, Array[Int], String)] = {
    val imei_index_onehotfeature_cyc = imei_onehotfeature_cyc.map(r => {
      var result = new ArrayBuffer[(Int, Int)]()
      for (i <- 0 until r._2.length) {
        val f = r._2(i)
        result += ((i, f))
      }
      (r._1, result.toArray, r._3)
    })
    val size = imei_onehotfeature_cyc.count
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user total size=" + size + "*****************************")
    val feature_thr = samp_feature * size
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature_thr=" + feature_thr + "*****************************")
    val index_onehotfeaturecount = imei_index_onehotfeature_cyc.flatMap(r => r._2).reduceByKey(_ + _)
    index_onehotfeaturecount.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature total size=" + index_onehotfeaturecount.count + "*****************************")
    val index_onehotfeaturecount_filter = index_onehotfeaturecount.filter(r => r._2 > feature_thr)

    val index = index_onehotfeaturecount_filter.map(r => r._1).collect()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature filter size=" + index.length + "*****************************")

    val imei_onehotfeature_cyc_filter = imei_onehotfeature_cyc.map(r => {
      var result = new ArrayBuffer[Int]()
      for (i <- 0 until r._2.length) {
        if (index.contains(i)) {
          result += r._2(i)
        }
      }
      (r._1, result.toArray, r._3)
    })
    imei_onehotfeature_cyc_filter
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
    //        c1.add(Calendar.DATE, -1)
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
    println("***********************load_data finished" + df.count() + "*****************************")

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
    println("***********************feature_rdd finished" + feature_rdd.count() + "*****************************")
    feature_rdd

  }

  def load_data_new(): RDD[(Long, Array[Int], String)] = {
    val sql_1 = "select imei, sex, marriage_status, is_parent, mz_apps_car_owner, mz_apps_car_new_owner,mz_apps_car_old_owner,user_job, user_age, " +
      "user_life_city_lev, " +
      "app_contact_tag, app_education_tag, app_finance_tag, app_games_tag, app_health_tag, app_interact_tag, app_music_tag, " +
      "app_o2o_tag, app_read_tag, app_shopping_tag, app_travel_tag,  app_video_tag, " +
      "fcate_qc," +
      "uxip_boot_users_cycle " +
      "from user_profile.idl_fdt_dw_tag"
    //近3天活跃用户
    val df = hiveContext.sql(sql_1)

    val data_rdd1 = df.map(r => (r.getLong(0), r.getString(1), r.getString(2).toInt, r.getString(3).toInt, r.getString(4).toInt, r.getString(5).toInt, r.getString(6).toInt, r.getString(7).toInt, r.getString(8).toInt,
      r.getString(9).toInt,
      r.getString(10).split(",") ++ r.getString(11).split(",") ++ r.getString(12).split(",") ++ r.getString(13).split(",") ++ r.getString(14).split(",") ++ r.getString(15).split(",") ++ r.getString(16).split(",") ++
        r.getString(17).split(",") ++ r.getString(18).split(",") ++ r.getString(19).split(",") ++ r.getString(20).split(",") ++ r.getString(21).split(","),
      r.getString(22).split(","),
      r.getString(23)
    ))
    //兴趣爱好是
    df.repartition(400)
    val data_rdd2 = data_rdd1.mapPartitions(iter => for (r <- iter) yield
      (r._1, r._2, r._3, r._4, r._5, r._6, r._7, r._8, r._9, r._10, r._11.toSet
        .filter(r => (!r.equals("") && !r.equals("-999"))).toArray, r._12.filter(r => (!r.equals("") && !r.equals("-111"))), r._13)
    ).cache()
    println("***********************load_data finished" + data_rdd2.count() + "*****************************")
    val r11 = data_rdd2.flatMap(r => r._11).distinct.collect.zipWithIndex.toMap
    val r12 = data_rdd2.flatMap(r => r._12).distinct.collect.zipWithIndex.toMap
    val r11_br = sc.broadcast(r11)
    val r12_br = sc.broadcast(r12)
    //sex*2,marr*2,pare*2,car*2,job*10, age*5, city*6
    val feature_rdd = data_rdd2.mapPartitions(iter => for (r <- iter) yield {
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
      val newcar = new Array[Int](2)
      if (r._6 == 1) {
        newcar(0) = 1
      } else {
        newcar(1) = 1
      }
      val oldcar = new Array[Int](2)
      if (r._7 == 1) {
        oldcar(0) = 1
      } else {
        oldcar(1) = 1
      }
      var job = new Array[Int](10)
      if (r._8 >= 0 && r._8 <= 9) {
        job(r._8.toInt) = 1
      }
      var age = new Array[Int](5)
      if (r._9 >= 0 && r._9 <= 4) {
        age(r._9.toInt) = 1
      }
      var city = new Array[Int](6)
      if (r._10 >= 1 && r._10 <= 6) {
        city(r._10.toInt - 1) = 1
      }

      val array1 = sex ++ marr ++ pare ++ car ++ newcar ++ oldcar ++ job ++ age ++ city
      val r11 = r11_br.value
      val array2 = new Array[Int](r11.values.max + 1)
      for (key <- r._11) {
        array2(r11(key)) = 1
      }
      val fcate_qc = r._12
      val r12 = r12_br.value
      var fcate_qc_array = new Array[Int](r12.values.max + 1)
      for (key <- fcate_qc) {
        val i = r12_br.value(key)
        fcate_qc_array(i) = 1
      }

      (r._1, array1 ++ array2 ++ fcate_qc_array, r._13)

    }
    )
    println("***********************feature_rdd finished" + feature_rdd.count() + "*****************************")
    feature_rdd

  }

}
