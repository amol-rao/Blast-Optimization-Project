# Databricks notebook source
# MAGIC %md # Investigating Mine Blasting Data Using Clustering Techniques
# MAGIC <strong>Project Plan v1.0</strong><br/>
# MAGIC Date: May 1, 2017<br/>
# MAGIC Version: V4<br/>
# MAGIC Author: Amol Rao<br/>
# MAGIC Student Number: 994610453<br/>
# MAGIC Course: MIE1512

# COMMAND ----------

# MAGIC %md ## Contents 
# MAGIC The contents of this notebook are as follows:
# MAGIC 
# MAGIC ### 1. Introduction
# MAGIC 
# MAGIC ### 2. Project Plan & Task Breakdown
# MAGIC 
# MAGIC ### 3. Data Understanding
# MAGIC 
# MAGIC ### 4. Data Preparation
# MAGIC 
# MAGIC ### 5. Modeling 
# MAGIC 
# MAGIC ### 6. Alternative Iterations
# MAGIC 
# MAGIC ### 7. Conclusions & Recommendations
# MAGIC 
# MAGIC ### 8. Bibliography

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1.0) Introduction
# MAGIC 
# MAGIC ### 1.1) Motivation of the Problem
# MAGIC 
# MAGIC The motivation for this investigation is to better understand the factors driving the relationship between blast outcomes and rock mass parameters for an open pit gold mine located in North Eastern Ontario. 
# MAGIC 
# MAGIC The blasting of rock at mines serves two primary purposes: a) to enable access to the mineral deposit containing the ore and b) fragment the ore to a suitable size for digging and processing. 
# MAGIC 
# MAGIC Both objectives require optimization of the blasting process to minimize cost and maximize speed at which fragmented rock can be moved on for processing through improved digability. The efficiency of the blasting operation depends on a few key parameters, including; rock characteristics, properties and quantities of explosives, blast geometry, blast size, priming method and initiation sequence. 
# MAGIC 
# MAGIC As part of this analysis, <strong> the geotechnical and blasting parameters </strong> will be explored to better understand groupings within the data. It is hoped the analysis can be used to eventually optimize blasting to reduce movement and make the fragmentation process more efficient. 
# MAGIC 
# MAGIC ### 1.1) Analysis Method (Selected from Bibliography)
# MAGIC 
# MAGIC Data clustering refers to the unsupervised classification of features into patterns that then form meaningful groups. Data within 1 cluster are more similar than data in another cluster. As described by Jain et. al [8] there are four main components to a clustering task: 
# MAGIC 
# MAGIC * Pattern representation - Comprised of selection of the features and extraction for the dataset. This step also involves identifying number of classes in the representation.
# MAGIC 
# MAGIC * Definition of pattern proximity - This is the distance function defined on the pairs of patterns. For example Euclidean distance is a common measure.
# MAGIC 
# MAGIC * Clustering or grouping - There are many types of clustering techniques.  
# MAGIC 
# MAGIC * Data abstraction - Once clustering has been done, a person with knowledge of the domain needs to identify what is represented and ensure the clusters are simple and compact for scalability. 
# MAGIC 
# MAGIC * Assessment of output - Perform some type of validity analysis on the cluster. 
# MAGIC 
# MAGIC Hudaverdi [1] uses a hierarchical clustering technique to group 88 blast data into two groups of similarity. Blast design parameters such as burden, spacing, bench height and stemming as well as geotechnical data are used to to obtain two seperate clusters of data. Following this, each cluster is fed into a regression model to predict ground vibration, as indicated by a measure of the Peak Particle Velocity (PPV). A hierarchical technique is used to cluster all blasts into one of two groups. After standardization of the data, the Pearson coefficient of similarity is used to group data. 
# MAGIC 
# MAGIC ### 1.3) Application to This Dataset
# MAGIC Due to the constraint of using the MlLib toolkit - the clustering algorithm that will be explored in this analysis is the K-Means method. K-means is one of the most commonly used clustering algorithms that clusters the data points into a predefined number of clusters. The number of clusters is optimized by reducing the error on the within set sum of squared error. 
# MAGIC 
# MAGIC The items to be clustered are <strong> individual blasts </strong> based on the features to be described later on this notebook. The blast data  was recorded with instrumentation and manually and provided by Professor Kamran Esmaeili for this investigation.  

# COMMAND ----------

# MAGIC %md <a id="plan"></a>
# MAGIC ## 2.0) Project Plan & Task Breakdown
# MAGIC 
# MAGIC <br/>
# MAGIC ### 2.1) Project Plan
# MAGIC 
# MAGIC Based on the CRISP-DM method, these are the proposed tasks for the project:
# MAGIC 
# MAGIC * <strong>Week 1 - Data Understanding</strong>
# MAGIC   * Data Review 
# MAGIC   * Data verification (with mining professors)
# MAGIC   * Exploratory analysis
# MAGIC   * Review + identify potential analytics techniques
# MAGIC  
# MAGIC * <strong>Week 2 - Data Preparation</strong>
# MAGIC   * Data Cleaning
# MAGIC   * Feature extraction and selection
# MAGIC   * Integrate and format data for model
# MAGIC   * Finalize modeling techniques to be used
# MAGIC   
# MAGIC * <strong>Week 3 - Modeling</strong>
# MAGIC   * Implement selected modeling technique
# MAGIC   * Build model
# MAGIC   * Assess model by computing error
# MAGIC   
# MAGIC * <strong>Week 4 - Evaluation of Alternatives</strong>
# MAGIC   * Evaluate alternative approaches to clustering
# MAGIC   * Review process and next steps
# MAGIC   
# MAGIC 
# MAGIC <br/>
# MAGIC ### 2.2) Task Breakdown
# MAGIC It is anticipated that 10 hours will be allocated each week to perform the tasks required for completion of the project.
# MAGIC <br/>
# MAGIC #### 2.2.1) Data Understanding (Week 1)
# MAGIC The data understanding task involves the following steps: 
# MAGIC 
# MAGIC 
# MAGIC * Data Review  (1 hour)
# MAGIC   * Review all tables to develop understanding of data 
# MAGIC 
# MAGIC * Data verification (3 hours)
# MAGIC   * Meet with Mining Professors to better understand data
# MAGIC   * Eliminate data not required or redundant 
# MAGIC   * Clarify required analyses 
# MAGIC 
# MAGIC * Exploratory analysis
# MAGIC   * Completed by J.Ponn for select tables
# MAGIC 
# MAGIC * Project planning (3 hours)
# MAGIC   * Setup notebook
# MAGIC   * Identify breakdown of tasks for each iteration
# MAGIC   * Allocate time estimates for tasks
# MAGIC 
# MAGIC * Revisions (1 hour)
# MAGIC   * Group discussions about project plans
# MAGIC   * Revisions based on feedback and discussion decisions
# MAGIC 
# MAGIC * Misc (2 hours)
# MAGIC   * Buffer for unexpected problems, or changes based on group discussion for this iteration
# MAGIC   
# MAGIC #### 2.2.2) Data Preparation (Week 2)
# MAGIC The data preparation task involves the following steps: 
# MAGIC 
# MAGIC 
# MAGIC * Data Cleaning (3 hours)
# MAGIC   * Identify tables that need to be cleaned : the important tables are smry_pen_rate, smry_geotech, smry_movement, smry_bmm
# MAGIC   * Normalize the text, remove duplicate or error entries
# MAGIC * Feature extraction and pattern representation (3 hours)
# MAGIC   * Identify features to be used for input to the model.
# MAGIC   * Pattern representation consists of defining the number of classes and features to be used in the clustering algorithm.
# MAGIC 
# MAGIC * Finalize modeling techniques to be used (2 hours)
# MAGIC   * Model to be considered is the <strong> K-Means</strong>.  
# MAGIC   * The spark.mllib implementation includes a parallelized variant of the k-means++ method called kmeans||.   
# MAGIC * Misc. (2 hour)
# MAGIC   * Group discussion, professor feedback, corrections to notebook 
# MAGIC 
# MAGIC #### 2.2.3) Modeling (Week 3)
# MAGIC The data modeling task involves the following steps: 
# MAGIC 
# MAGIC * Integrate and format data for model (2 hours)
# MAGIC   * Identify the steps needed to input the data into the model. 
# MAGIC   * Modify features to be used to ensure they can be passed to the model.
# MAGIC * Build model (5 hours)
# MAGIC   * Run the model, identify clusters
# MAGIC * Assess model (2 hours)
# MAGIC   * Perform validity analysis on the clusters produced
# MAGIC   * Ensure that clusters are tightly bound
# MAGIC   * Compute Within Set of Squared Error (WSSSE)
# MAGIC      
# MAGIC 
# MAGIC #### 2.2.3) Evaluation of Alternatives (Week 4)
# MAGIC The evaluation task involves the following steps: 
# MAGIC 
# MAGIC * Evaluate results (6 hours)
# MAGIC   * Run several iterations on the k-means model, with varying K. Use 'Elbow Method' to identify optimal K.
# MAGIC   * Perform other iterations of clustering algorithm.
# MAGIC * Review process and next steps (4 hours)
# MAGIC   * Make changes to the model as necessary. 
# MAGIC   * Draft conclusions and recommendations for next steps
# MAGIC * Misc.
# MAGIC   * Group discussion, professor feedback, corrections to notebook.

# COMMAND ----------

# MAGIC %md <a id="understanding"></a>
# MAGIC ## 3.0) Data Understanding

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.1) Data Loading

# COMMAND ----------

# MAGIC %md
# MAGIC Upload data into databricks, set implicits and define functions to be used to ensure data will be in the type required for processing. 

# COMMAND ----------

# Import necessary libraries
import pandas as pd # For csv I/O and data formatting
import re # To search through strings
import numpy as np

# COMMAND ----------

# MAGIC %scala
# MAGIC import java.sql.Timestamp
# MAGIC import org.apache.commons.io.IOUtils
# MAGIC import java.net.URL
# MAGIC import java.nio.charset.Charset
# MAGIC // val sqlContext= new org.apache.spark.sql.SQLContext(sc)
# MAGIC //      import sqlContext.implicits._
# MAGIC 
# MAGIC 
# MAGIC // patching the String class with new functions that have a defualt value if conversion to another type fails.
# MAGIC implicit class StringConversion(val s: String) {
# MAGIC def toTypeOrElse[T](convert: String=>T, defaultVal: T) = try {
# MAGIC     convert(s)
# MAGIC   } catch {
# MAGIC     case _: Throwable => defaultVal
# MAGIC   }
# MAGIC   
# MAGIC   def toIntOrElse(defaultVal: Int = 0) = toTypeOrElse[Int](_.toInt, defaultVal)
# MAGIC   def toDoubleOrElse(defaultVal: Double = 0D) = toTypeOrElse[Double](_.toDouble, defaultVal)
# MAGIC   def toDateOrElse(defaultVal: java.sql.Timestamp = java.sql.Timestamp.valueOf("1970-01-01 00:00:00")) = toTypeOrElse[java.sql.Timestamp](java.sql.Timestamp.valueOf(_), defaultVal)
# MAGIC }
# MAGIC 
# MAGIC //Fix the date format in this dataset
# MAGIC def fixDateFormat(orig: String): String = {
# MAGIC     val splited_date = orig.split(" ")
# MAGIC     val fixed_date_parts = splited_date(0).split("-").map(part => if (part.size == 1) "0" + part else part)
# MAGIC     val fixed_date = List(fixed_date_parts(0), fixed_date_parts(1), fixed_date_parts(2)).mkString("-")
# MAGIC     val fixed_time = splited_date(1).split(":").map(part => if (part.size == 1) "0" + part else part).mkString(":")
# MAGIC     fixed_date + " " + fixed_time + ":00"
# MAGIC }

# COMMAND ----------

# MAGIC %scala
# MAGIC val bmmRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFQnhJY0pfclBLUmM"), Charset.forName("utf8")).split("\n"))
# MAGIC 
# MAGIC val geotechRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFTjN5WmQ3MmsyTEE"), Charset.forName("utf8")).split("\n"))
# MAGIC 
# MAGIC val load_timeRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFdS0yU29YdkhiWjQ"), Charset.forName("utf8")).split("\n"))
# MAGIC 
# MAGIC val movementRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFMUhEMjJoNk1JZTg"), Charset.forName("utf8")).split("\n"))
# MAGIC 
# MAGIC val truckRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFSkRhc2ZrZDM2aUk"), Charset.forName("utf8")).split("\n"))
# MAGIC 
# MAGIC val pen_rate_summRDD = sc.parallelize( IOUtils.toString( new URL("https://drive.google.com/uc?export=download&id=0BzQSUz4yGQIFS3J3WFJPV21ucW8"), Charset.forName("utf8")).split("\n"))

# COMMAND ----------

# MAGIC %scala
# MAGIC case class geotech(             
# MAGIC     Level: Int,
# MAGIC     Blast_Number: String,                                     
# MAGIC     gd_1: String,
# MAGIC     gd_2: String,
# MAGIC     rqd: Double,
# MAGIC     fracs: Double,
# MAGIC     is_50: Double,
# MAGIC     Q: Double
# MAGIC   )
# MAGIC 
# MAGIC def getgeotechCleaned(row:Array[String]):geotech = {
# MAGIC   return geotech(
# MAGIC     row(0).toIntOrElse(),
# MAGIC     row(1),
# MAGIC     row(2),
# MAGIC     row(3),
# MAGIC     row(4).toDoubleOrElse(),
# MAGIC     row(5).toDoubleOrElse(),
# MAGIC     row(6).toDoubleOrElse(),
# MAGIC     row(7).toDoubleOrElse()
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC val geotech_data = geotechRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = geotech_data.filter(s => s(0) != "Level").map(s => getgeotechCleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_geotech")

# COMMAND ----------

# MAGIC %sql -- View the Geotech Summary Table
# MAGIC SELECT *
# MAGIC FROM smry_geotech
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %scala
# MAGIC case class load_time(               
# MAGIC     BLAST_LOC: String,
# MAGIC     LVL: Int,
# MAGIC     BLAST_NO: Int,
# MAGIC     HS_AVG: Double,
# MAGIC     HS_QT_90: Double,
# MAGIC     HS_QT_10: Double,
# MAGIC     HS_MED: Double,
# MAGIC     HS_NUM_TRUCKS: Double,
# MAGIC     RS_AVG: Double,
# MAGIC     RS_QT_90: Double,
# MAGIC     RS_QT_10: Double,
# MAGIC     RS_MED: Double,
# MAGIC     RS_NUM_TRUCKS: Double
# MAGIC   )
# MAGIC   
# MAGIC def getload_timeCleaned(row:Array[String]):load_time = {
# MAGIC return load_time(
# MAGIC     row(0),
# MAGIC     row(1).toIntOrElse(),
# MAGIC     row(2).toIntOrElse(),
# MAGIC     row(3).toDoubleOrElse(),
# MAGIC     row(4).toDoubleOrElse(),
# MAGIC     row(5).toDoubleOrElse(),
# MAGIC     row(6).toDoubleOrElse(),
# MAGIC     row(7).toDoubleOrElse(),
# MAGIC     row(8).toDoubleOrElse(),
# MAGIC     row(9).toDoubleOrElse(),
# MAGIC     row(10).toDoubleOrElse(),
# MAGIC     row(11).toDoubleOrElse(),
# MAGIC     row(12).toDoubleOrElse()
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC 
# MAGIC val loading_time = load_timeRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = loading_time.filter(s => s(0) != "BLAST_LOC").map(s => getload_timeCleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_loading_time")

# COMMAND ----------

# MAGIC %sql -- View the Loading Time Summary Table
# MAGIC SELECT *
# MAGIC FROM smry_loading_time
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %scala
# MAGIC case class movement(                 
# MAGIC    BLAST_LOC: String,
# MAGIC    LOC: String,
# MAGIC    BENCH_HT: Int,
# MAGIC    HOLE_DIAM:Int,
# MAGIC    BURDEN: Int,
# MAGIC    SPACING: Int,
# MAGIC    HOLE_CONFIG: String,
# MAGIC    CONFINEMENT: String,
# MAGIC    PATTERN: String,
# MAGIC    STEMMING: Double,
# MAGIC    POWDER_FAC: Double,
# MAGIC    IR_TIME: Int,
# MAGIC    IH_TIME: Int,
# MAGIC    INITIATION_TYPE: String,
# MAGIC    EXPLOSIVE_TYPE: String,
# MAGIC    ROCK_TYPE: String,
# MAGIC    BMM: Int,
# MAGIC    COLLAR_NORTH: Double,
# MAGIC    COLLAR_EAST: Double,
# MAGIC    COLLAR_RL: Double,
# MAGIC    PRE_BMM_RL: Double,
# MAGIC    INST_DEPTH: Double,
# MAGIC    AFTER_EAST: Double,
# MAGIC    AFTER_NORTH: Double,
# MAGIC    SURFACE_RL: Double,
# MAGIC    POST_BMM_RL: Double,
# MAGIC    THREED_DIST: Double,
# MAGIC    H_DIST: Double,
# MAGIC    V_DIST: Double,
# MAGIC    DIRECTION: Double,
# MAGIC    INCLINATION: Double,
# MAGIC    HEAVE: Double
# MAGIC   )
# MAGIC   
# MAGIC def getmovementCleaned(row:Array[String]):movement = {
# MAGIC return movement(
# MAGIC     row(0),
# MAGIC     row(1),
# MAGIC     row(2).toIntOrElse(),
# MAGIC     row(3).toIntOrElse(),
# MAGIC     row(4).toIntOrElse(),
# MAGIC     row(5).toIntOrElse(),
# MAGIC     row(6),
# MAGIC     row(7),
# MAGIC     row(8),
# MAGIC     row(9).toDoubleOrElse(),
# MAGIC     row(10).toDoubleOrElse(),
# MAGIC     row(11).toIntOrElse(),
# MAGIC     row(12).toIntOrElse(),
# MAGIC     row(13),
# MAGIC     row(14),
# MAGIC     row(15),
# MAGIC     row(16).toIntOrElse(),
# MAGIC     row(17).toDoubleOrElse(),
# MAGIC     row(18).toDoubleOrElse(),
# MAGIC     row(19).toDoubleOrElse(),
# MAGIC     row(20).toDoubleOrElse(),
# MAGIC     row(21).toDoubleOrElse(),
# MAGIC     row(22).toDoubleOrElse(),
# MAGIC     row(23).toDoubleOrElse(),
# MAGIC     row(24).toDoubleOrElse(),
# MAGIC     row(25).toDoubleOrElse(),
# MAGIC     row(26).toDoubleOrElse(),
# MAGIC     row(27).toDoubleOrElse(),
# MAGIC     row(28).toDoubleOrElse(),
# MAGIC     row(29).toDoubleOrElse(),
# MAGIC     row(30).toDoubleOrElse(),
# MAGIC     row(31).toDoubleOrElse()
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC val smry_movement = movementRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = smry_movement.filter(s => s(0) != "BLAST_LOC").map(s => getmovementCleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_movement")

# COMMAND ----------

# MAGIC %sql -- View the Summary Movement Table
# MAGIC SELECT *
# MAGIC FROM smry_movement
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %scala
# MAGIC case class bmm(                 
# MAGIC    BLAST_LOC: String,
# MAGIC    GEO_DOM1: String,
# MAGIC    GEO_DOM2: String,
# MAGIC    RQD: Double,
# MAGIC    FRACS_FREQ: Double,
# MAGIC    IS_50: Double,
# MAGIC    Q: Double,
# MAGIC    LEVEL: String,
# MAGIC    PEN_RATE: Double,
# MAGIC    BLAST_DATE: String,
# MAGIC    ROCK_TYPE: String,
# MAGIC    MATERIAL: String,
# MAGIC    HG_MG: Int,
# MAGIC    LG: Int,
# MAGIC    NAG: Int,
# MAGIC    TOTAL_TONS: Int,
# MAGIC    BLAST_TYPE: String ,  
# MAGIC    DH_DELAY: Int,
# MAGIC    HS_DELAY: Int,
# MAGIC    RS_DELAY: Int,
# MAGIC    HOLE_DIA:Int, 
# MAGIC    BENCH_HEIGHT: Int,
# MAGIC    BURDEN: Int,
# MAGIC    SPACING: Int,
# MAGIC    STEMMING: Double,
# MAGIC    SUB_DRILL: Double ,
# MAGIC    EXPLOSIVE: String,
# MAGIC    BLAST_PATTERN: String,
# MAGIC    TOTAL_EXPLOSIVE: Double,
# MAGIC    HOLE_DEPTH: Double,
# MAGIC    FIRING_DIRECTION: String,
# MAGIC    POWDER_FACTOR: Double,
# MAGIC    INITIATION_TYPE: String,
# MAGIC    CONFINEMENT: String,
# MAGIC    BMM_NUM: Int,
# MAGIC    THREED_DIST: Double,
# MAGIC    H_DIST: Double,
# MAGIC    V_DIST: Double,
# MAGIC    DIRECTION: Double,
# MAGIC    INCLINATION: Double,
# MAGIC    HEAVE: Double 
# MAGIC   ) 
# MAGIC    
# MAGIC   
# MAGIC def getbmmcleaned(row:Array[String]):bmm = {
# MAGIC return new bmm(
# MAGIC     row(0),
# MAGIC     row(1),
# MAGIC     row(2),
# MAGIC     row(3).toDoubleOrElse(),
# MAGIC     row(4).toDoubleOrElse(),
# MAGIC     row(5).toDoubleOrElse(),
# MAGIC     row(6).toDoubleOrElse(),
# MAGIC     row(7),
# MAGIC     row(8).toDoubleOrElse(),
# MAGIC     row(9),
# MAGIC     row(10),
# MAGIC     row(11),
# MAGIC     row(12).toIntOrElse(),
# MAGIC     row(13).toIntOrElse(),
# MAGIC     row(14).toIntOrElse(),
# MAGIC     row(15).toIntOrElse(),
# MAGIC     row(16) ,
# MAGIC     row(17).toIntOrElse(),
# MAGIC     row(18).toIntOrElse(),
# MAGIC     row(19).toIntOrElse(),
# MAGIC     row(20).toIntOrElse(),
# MAGIC     row(21).toIntOrElse(),
# MAGIC     row(22).toIntOrElse(),
# MAGIC     row(23).toIntOrElse(),
# MAGIC     row(24).toDoubleOrElse(),
# MAGIC     row(25).toDoubleOrElse(),
# MAGIC     row(26),
# MAGIC     row(27),
# MAGIC     row(28).toDoubleOrElse(),
# MAGIC     row(29).toDoubleOrElse(),
# MAGIC     row(30),
# MAGIC     row(31).toDoubleOrElse(),
# MAGIC     row(32),
# MAGIC     row(33),
# MAGIC     row(34).toIntOrElse(),
# MAGIC     row(35).toDoubleOrElse(),
# MAGIC     row(36).toDoubleOrElse(),
# MAGIC     row(37).toDoubleOrElse(),
# MAGIC     row(38).toDoubleOrElse(),
# MAGIC     row(39).toDoubleOrElse(),
# MAGIC     row(40).toDoubleOrElse() 
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC val smry_bmm = bmmRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = smry_bmm.filter(s => s(0) != "BLAST_LOC").map(s => getbmmcleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_bmm")

# COMMAND ----------

# MAGIC %sql -- View the Summary BMM Table
# MAGIC SELECT *
# MAGIC FROM smry_bmm
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %scala
# MAGIC case class truck(               
# MAGIC     BLAST_LOC: String,
# MAGIC     LVL: Int,
# MAGIC     BLAST_NO: Int,
# MAGIC     HS_AVG: Double,
# MAGIC     HS_QT_90: Double,
# MAGIC     HS_QT_10: Double,
# MAGIC     HS_MED: Double,
# MAGIC     HS_NUM_TRUCKS: Int,
# MAGIC     RS_AVG: Double,
# MAGIC     RS_QT_90: Double,
# MAGIC     RS_QT_10: Double,
# MAGIC     RS_MED: Double,
# MAGIC     RS_NUM_TRUCKS: Int
# MAGIC   )
# MAGIC   
# MAGIC def gettruckCleaned(row:Array[String]):truck = {
# MAGIC return truck(
# MAGIC     row(0),
# MAGIC     row(1).toIntOrElse(),
# MAGIC     row(2).toIntOrElse(),
# MAGIC     row(3).toDoubleOrElse(),
# MAGIC     row(4).toDoubleOrElse(),
# MAGIC     row(5).toDoubleOrElse(),
# MAGIC     row(6).toDoubleOrElse(),
# MAGIC     row(7).toIntOrElse(),
# MAGIC     row(8).toDoubleOrElse(),
# MAGIC     row(9).toDoubleOrElse(),
# MAGIC     row(10).toDoubleOrElse(),
# MAGIC     row(11).toDoubleOrElse(),
# MAGIC     row(12).toIntOrElse()
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC val smry_truck_quantity = truckRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = smry_truck_quantity.filter(s => s(0) != "BLAST_LOC").map(s => gettruckCleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_truck_quantity")

# COMMAND ----------

# MAGIC %sql -- View the Summary Truck Quantity Table
# MAGIC SELECT *
# MAGIC FROM smry_truck_quantity
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %scala
# MAGIC case class penetration_rate_summ(                    
# MAGIC     BLAST_LOC: String,
# MAGIC     LVL: String,
# MAGIC     BLAST_NO: String,
# MAGIC     X: String,                            
# MAGIC     Y: String,                                  
# MAGIC     Z: String,                                   
# MAGIC     PEN_RATE: Double,
# MAGIC     ROCK_TYPE: String
# MAGIC   )
# MAGIC   
# MAGIC def getpenetration_rate_summCleaned(row:Array[String]):penetration_rate_summ = {
# MAGIC return penetration_rate_summ(
# MAGIC     row(0),
# MAGIC     row(1),
# MAGIC     row(2),
# MAGIC     row(3),
# MAGIC     row(4),
# MAGIC     row(5),
# MAGIC     row(6).toDoubleOrElse(),
# MAGIC     row(7)
# MAGIC   )
# MAGIC }
# MAGIC 
# MAGIC val pen_rates_summ = pen_rate_summRDD.map(line => line.split(",").map(elem => elem.trim))
# MAGIC val data = pen_rates_summ.filter(s => s(0) != "BLAST_LOC").map(s => getpenetration_rate_summCleaned(s)).toDF()
# MAGIC data.createOrReplaceTempView("smry_pen_rate")

# COMMAND ----------

# MAGIC %sql -- View the Summary Penetration Rate Table
# MAGIC SELECT *
# MAGIC FROM smry_pen_rate
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.2) Data Exploration 

# COMMAND ----------

# MAGIC %md
# MAGIC After loading of the data, conduct data exploration to see what is in the tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review of Data within the SMRY_GEOTECH Table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How many unique blast levels are there?
# MAGIC SELECT COUNT(LEVEL)
# MAGIC FROM smry_geotech

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Counts of distinct levels in smry_geotech
# MAGIC SELECT DISTINCT (LEVEL), COUNT(LEVEL) as counts
# MAGIC FROM smry_geotech
# MAGIC GROUP BY LEVEL 
# MAGIC ORDER BY counts

# COMMAND ----------

# MAGIC %md
# MAGIC The levels range from 148 to 208, and include a new row for each new blasting record. So for example, Level 244 has over 120 blast rows associated with it. See the summary table below. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distinct number of blast numbers in the smry_geotech table
# MAGIC SELECT DISTINCT blast_number 
# MAGIC FROM smry_geotech
# MAGIC GROUP BY blast_number
# MAGIC ORDER BY blast_number DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of RQD values
# MAGIC SELECT rqd
# MAGIC FROM smry_geotech

# COMMAND ----------

# MAGIC %md
# MAGIC The largest bin of RQD values is between 90-95 (there are >300) and between 95-100 (almost 450). The values around 0 appear to be errors or blanks and should be removed. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How many distinct rqd values does the smry_geotech have? 
# MAGIC SELECT distinct(rqd)
# MAGIC FROM smry_geotech
# MAGIC GROUP BY rqd

# COMMAND ----------

# MAGIC %md
# MAGIC Since there are only 93 distinct rqd values, it appears that these values have been summarized and applied to many more blast locations.  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of fracs
# MAGIC SELECT fracs
# MAGIC FROM smry_geotech

# COMMAND ----------

# MAGIC %md
# MAGIC The number of fracs appear to be distributed around 5, with a range from 0 - 14. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of is_50 values
# MAGIC SELECT is_50
# MAGIC FROM smry_geotech

# COMMAND ----------

# MAGIC %md
# MAGIC Distribution of is50 values is around 9, with a range from 4 - 16. The values around 0 appear to be erroneous and should be removed.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of q values
# MAGIC SELECT q
# MAGIC FROM smry_geotech

# COMMAND ----------

# MAGIC %md
# MAGIC Distribution of q values is around 14, with the range mostly between 5 - 25. The values around 0 appear to be erroneous and should be removed. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review of Data within the smry_pen_rate table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How many individual blast locaitons are there?
# MAGIC SELECT distinct (BLAST_LOC)
# MAGIC FROM smry_pen_rate

# COMMAND ----------

# MAGIC %md
# MAGIC The levels in smry_pen_rate range from 148 - 268, with four records for levels that have alpha-numeric characters. These should be removed. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Counts of distinct levels 
# MAGIC SELECT DISTINCT (lvl), COUNT(lvl) as counts
# MAGIC FROM smry_pen_rate
# MAGIC GROUP BY lvl 
# MAGIC ORDER BY counts

# COMMAND ----------

# MAGIC %md 
# MAGIC Number of levels range from 160-234 which is similar to that shown in the smry_geotech. There are some discrepancies, such as the alphabetic characters. These should be removed.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Counts of distinct levels 
# MAGIC SELECT DISTINCT (BLAST_NO), COUNT(BLAST_NO) as counts
# MAGIC FROM smry_pen_rate
# MAGIC GROUP BY BLAST_NO 
# MAGIC ORDER BY counts

# COMMAND ----------

# MAGIC %md
# MAGIC This table should correspond to the blast_number summary from smry_geotech above. However it's clear that there are large differences. For example, in this table there are many values with alphabetic characters. Ideally they should have a three digit numeric character. These will need to be normalized or removed during cleaning.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of pen_rate values
# MAGIC SELECT pen_rate
# MAGIC FROM smry_pen_rate

# COMMAND ----------

# MAGIC %md
# MAGIC There are many penetrations rates above 2. It is unclear if these are erroneous and should be exploded. Check with the mining professor.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Rock type classifications
# MAGIC SELECT DISTINCT rock_type, count(rock_type)
# MAGIC FROM smry_pen_rate
# MAGIC GROUP BY rock_type

# COMMAND ----------

# MAGIC %md
# MAGIC The majority of rock type classification is mf, followed by kp and ovb.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review of Data within the smry_bmm table

# COMMAND ----------

# MAGIC %md
# MAGIC Following discussions with Prof. Esmaeili and B.Ohadi, another data table was provided (smry_bmm). This table contains geotechnical data, blast parameters and blast outcomes for distinct blast locations. The data was compiled by B.Ohadi.

# COMMAND ----------

# MAGIC %sql -- Let's see what's in the smry_bmm table
# MAGIC SELECT *
# MAGIC FROM smry_bmm
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- See how many blast locations per blast level
# MAGIC SELECT LEVEL, COUNT(LEVEL)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY LEVEL
# MAGIC SORT BY COUNT(LEVEL) DESC

# COMMAND ----------

# MAGIC %md
# MAGIC Some levels have a large number of blasts - for example, 208 and 256, while some levels have far fewer blast data associated. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- See distribution of RQD
# MAGIC SELECT RQD
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Based on visual assessment of the RQD, it appears that the data has been taken from the smry_geotech table. As that table has already been explored above, the subsequent visualizations will focus on the non-geotech parameters. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Box plot of Total Tons distribution
# MAGIC SELECT TOTAL_TONS
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Median tonnage is around 320,00o T, with 75% of data occuring within 200,000 to 600,000 T.

# COMMAND ----------

# MAGIC %sql --See distribution of IS_50 values
# MAGIC SELECT is_50
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What Blast Types are there?
# MAGIC SELECT DISTINCT Blast_Type, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY BLAST_TYPE

# COMMAND ----------

# MAGIC %md
# MAGIC Majority of blast types are for production

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most prominant HS_delay used in the blasting process?
# MAGIC SELECT DISTINCT HS_DELAY, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY HS_DELAY

# COMMAND ----------

# MAGIC %md
# MAGIC A blasting delay of 42 ms was most used. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most prominant RS_delay used in the blasting process?
# MAGIC SELECT DISTINCT RS_DELAY, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY RS_DELAY

# COMMAND ----------

# MAGIC %md
# MAGIC The most common blasting dleay was 100 ms. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Lets visualize the hole diameter distributions
# MAGIC SELECT DISTINCT HOLE_DIA, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY HOLE_DIA

# COMMAND ----------

# MAGIC %md
# MAGIC The greatest proportion of hole diameter was 216 mm. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common burden? 
# MAGIC SELECT DISTINCT BURDEN, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY BURDEN

# COMMAND ----------

# MAGIC %md
# MAGIC Burden of 6 m was used in almost all cases. Values of 0 are likely erroneous and should be removed.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common bencht height? 
# MAGIC SELECT DISTINCT BENCH_HEIGHT, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY BENCH_HEIGHT

# COMMAND ----------

# MAGIC %md
# MAGIC Almost all bench heights were 12 m.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common spacing? 
# MAGIC SELECT DISTINCT SPACING, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY SPACING

# COMMAND ----------

# MAGIC %md
# MAGIC Most common stemming height was 7 m.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common stemming? 
# MAGIC SELECT DISTINCT STEMMING, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY STEMMING

# COMMAND ----------

# MAGIC %md
# MAGIC Most common stemming was 4.5 m, followed by 5 m and 4 m. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common explosive? 
# MAGIC SELECT DISTINCT EXPLOSIVE, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY EXPLOSIVE

# COMMAND ----------

# MAGIC %md
# MAGIC Almost all explosive used was th Fortis Extra 1.05

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What types of blast patterns were used? 
# MAGIC SELECT DISTINCT BLAST_PATTERN, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY BLAST_PATTERN

# COMMAND ----------

# MAGIC %md
# MAGIC Almost all th blast patterns were staggered. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is the most common hole depth? 
# MAGIC SELECT DISTINCT HOLE_DEPTH, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY HOLE_DEPTH

# COMMAND ----------

# MAGIC %md
# MAGIC Hole depth ranged from 1 to 8 m, with most holes drilled to 7 m depth.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is breakdown of blast initiation types? 
# MAGIC SELECT DISTINCT INITIATION_TYPE, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY INITIATION_TYPE

# COMMAND ----------

# MAGIC %md
# MAGIC Most initiation types were "V" followed by "Echelon"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is breakdown of confinement? 
# MAGIC SELECT DISTINCT CONFINEMENT, COUNT(*)
# MAGIC FROM smry_bmm
# MAGIC GROUP BY CONFINEMENT

# COMMAND ----------

# MAGIC %md
# MAGIC Two types of confinement are presented. Free face and Choked. The values need to be normalized.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is distribution of three d distances? 
# MAGIC SELECT THREED_DIST
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Most 3d-distance values range between 3.6 to 5.1 m. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is distribution of horizontal distances? 
# MAGIC SELECT H_DIST
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, most horizontal distances range from 3.1 m to 4.1 m

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is distribution of vertical distances? 
# MAGIC SELECT V_DIST
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Vertical distance has a greater fluctuation, and a median around 1.5 m.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What is distribution of heave? 
# MAGIC SELECT heave
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md
# MAGIC Most heave was measured between 2 and 5 m. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(BLAST_LOC)
# MAGIC FROM smry_bmm

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Data Selection
# MAGIC Based on a review of the tables, the smry_bmm table was selected for further analysis. This table was selected as it contains already contains the necessary elements to be assessed with the clustering technique. As it was shown, the Table contains:
# MAGIC * Geotechnical data (Such as RQD, FRAQS_FREQ, Q, and PEN_RATE)
# MAGIC * Blast data (Such as BLAST_LOC, LEVEL)
# MAGIC * Rock data (Such as ROCK_TYPE, MATERIAL)
# MAGIC * Blast parameters (Such as BENCH_HEIGHT, SPACING, STEMMING, SUB_DRILL, POWDER_FAC, INITIATION)
# MAGIC * Blast outcomes (Such as D_Dist, V_Dist, HEAVE )

# COMMAND ----------

# MAGIC %md <a id="preparation"></a>
# MAGIC ## 4.0) Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1) Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ####List of Cleaning to be Performed 
# MAGIC * Remove GEO_DOM, GEO_DOM1 - These columns appear to not provide any further useful information 
# MAGIC * Remove blast locations where no RQD, FRAQS_FREQ, IS_50 or Q exist, or are 0 - Geotech parameters are an important component of the clustering process
# MAGIC * Remove LEVEL column - not required for analysis 
# MAGIC * Remove BLAST_DATE column - not required for analysis 
# MAGIC * Normalize BLAST PATTERN & CONFINEMENT labels

# COMMAND ----------

#Convert spark df to pandas, do not select GEO_DOM, GEO_DOM1, BLAST_DATE, LEVEL, BMM_NUM 
pDf = sqlContext.sql("SELECT BLAST_LOC, RQD, FRACS_FREQ, IS_50, Q, PEN_RATE, ROCK_TYPE, BLAST_TYPE, DH_DELAY, HS_DELAY, RS_DELAY, HOLE_DIA, BENCH_HEIGHT, BURDEN, SPACING, STEMMING, SUB_DRILL, EXPLOSIVE, BLAST_PATTERN, TOTAL_EXPLOSIVE, HOLE_DEPTH, POWDER_FACTOR, INITIATION_TYPE, CONFINEMENT, THREED_DIST, H_DIST,V_DIST, DIRECTION, INCLINATION, HEAVE FROM smry_bmm ").toPandas()

# COMMAND ----------

#Confirm that appropriate columns selected
pDf.head()

# COMMAND ----------

#Lets whats in the dataframe
pDf.info()

# COMMAND ----------

#Remove the rows where RQD, FRACS_FREQ, IS_50, Q and PEN_RATE are zeros
pDf = pDf[(pDf.RQD > 0) & (pDf.FRACS_FREQ > 0) & (pDf.IS_50 > 0) & (pDf.Q > 0) & (pDf.PEN_RATE > 0)]
#We only have 258 rows in the dataframe
pDf.shape

# COMMAND ----------

# Normalize string labels by removing excessive whitespace in the middle of a word, shifting to lowercase, and removing any whitespace after the text
def normalizeLabels(label):
  return re.sub('[ ]+',' ',label).lower().rstrip().lstrip()

# COMMAND ----------

#Make the labels lowercase and strip blank spaces
pDf['CONFINEMENT'] = pDf['CONFINEMENT'].apply(normalizeLabels)
pDf['BLAST_TYPE'] = pDf['BLAST_TYPE'].apply(normalizeLabels)
pDf['ROCK_TYPE'] = pDf['ROCK_TYPE'].apply(normalizeLabels)
pDf['EXPLOSIVE'] = pDf['EXPLOSIVE'].apply(normalizeLabels)
pDf['BLAST_PATTERN'] = pDf['BLAST_PATTERN'].apply(normalizeLabels)
pDf['INITIATION_TYPE'] = pDf['INITIATION_TYPE'].apply(normalizeLabels)

# COMMAND ----------

#Confirm that the labels are normalized
pDf.head()

# COMMAND ----------

# MAGIC %md <a id="modeling"></a>
# MAGIC ## 5.0) Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1) Feature Extraction

# COMMAND ----------

# MAGIC %md
# MAGIC According to Hudaverdi [4] the ratios to be used for clustering are :
# MAGIC * Spacing (S) / Burden (B)
# MAGIC * Bench Height (H) / Burden (B)
# MAGIC * Burden (B) / Hole Diameter (D)
# MAGIC * Stemming (T) / Burden (B)
# MAGIC * Subdrill (U) / Burden (B)
# MAGIC * Powder Factor

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM smry_bmm
# MAGIC LIMIT 5

# COMMAND ----------

#Calculate ratios
pDf['SB_ratio'] = pDf['SPACING'] / pDf['BURDEN']
pDf['HB_ratio'] = pDf['BENCH_HEIGHT'] / pDf['BURDEN']
pDf['BD_ratio'] = pDf['BURDEN'] / pDf['HOLE_DIA']
pDf['TB_ratio'] = pDf['STEMMING'] / pDf['BURDEN']
pDf['UB_ratio'] = pDf['SUB_DRILL'] / pDf['BURDEN']

# COMMAND ----------

#Inspect dataframe to ensure the ratios have been calculated
pd.set_option('display.max_columns', None)
pDf.head()

# COMMAND ----------

# Convert a label feature into an index number
def facorizeFeature(feat, df, labelDict):
  # Convert string labels into an index number
  labels, levels = pd.factorize(df[feat])
  
  # Add index numbers to main dataframe
  df[feat] = labels
  
  # Store labels in dictionary for later reference
  labelDict[feat] = levels
  
  return df, labelDict

# COMMAND ----------

# Initalize label dictionary
mLabelDict = {}

# COMMAND ----------

# Apply label normalization to data
pDf, mLabelDict = facorizeFeature('ROCK_TYPE', pDf, mLabelDict)
pDf, mLabelDict = facorizeFeature('BLAST_TYPE', pDf, mLabelDict)
pDf, mLabelDict = facorizeFeature('EXPLOSIVE', pDf, mLabelDict)
pDf, mLabelDict = facorizeFeature('BLAST_PATTERN', pDf, mLabelDict)
pDf, mLabelDict = facorizeFeature('INITIATION_TYPE', pDf, mLabelDict)
pDf, mLabelDict = facorizeFeature('CONFINEMENT', pDf, mLabelDict)

# COMMAND ----------

#Check the dataframe to ensure strings have been factorized
pDf.head()

# COMMAND ----------

#See the indexed labels
mLabelDict

# COMMAND ----------

#Drop columns that are not required for the clustering process
features = pDf.drop(['BLAST_LOC','DIRECTION','INCLINATION','HEAVE'],1)

# COMMAND ----------

#Drop columns that include blast movement as these are outcomes 
features.drop(['THREED_DIST','H_DIST', 'V_DIST'],1,inplace='true')

# COMMAND ----------

#Drop columns that created ratios for
features.drop(['HOLE_DIA','BENCH_HEIGHT','BURDEN','SPACING','STEMMING','SUB_DRILL' ],1,inplace='true')

# COMMAND ----------

#View the dataframe to ensure the correct columns are in place
features.tail()

# COMMAND ----------

#Count the number of times that infinity occurs in the dataset
np.isinf(features).sum()

# COMMAND ----------

#Check if the dataframe has any null values 
features.isnull().sum()

# COMMAND ----------

#Convert infinities to NaN's so they can all be handled together
features.replace(np.inf, np.nan,inplace='True')

# COMMAND ----------

#Confirm infinities are removed
np.isinf(features).sum()

# COMMAND ----------

#Remove the NaN rows from the dataframe
features.dropna(inplace='True')

# COMMAND ----------

#Check the shape of the features dataframe now that the NaN's have been dropped
features.shape

# COMMAND ----------

#Create a list of the features 
featuresList = list(features)

# COMMAND ----------

#Let's see the features list to ensure it has all the features we want
featuresList

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2) Train the Model - Base Case  

# COMMAND ----------

# MAGIC %md
# MAGIC The first iteration of the model involves trialling with all the initial set of data and with a randomly selected k. The results from this assessment are not representative of the true clustering.
# MAGIC 
# MAGIC The parameters to be used are those as default in the spark.mllib implementation guide. Max iterations have been increased to 100, and the initialization mode is kept as random.  

# COMMAND ----------

#Import necessary libraries 
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from pyspark.mllib.stat import Statistics
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn import cluster
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler

# COMMAND ----------

#Parse the features pandas dataframe into an RDD of features which will be input into the model
featuresRDD = sqlContext.createDataFrame(features).rdd.map(lambda row: np.array([float(item) for item in row]))

# COMMAND ----------

#Let's see some summary statistics of the Features RDD
summary = Statistics.colStats(featuresRDD)
print(summary.mean())  # a dense vector containing the mean value for each column
print(summary.variance())  # column-wise variance
print(summary.numNonzeros())  # number of nonzeros in each column

# COMMAND ----------

# MAGIC %md
# MAGIC As it can be seen - several columns have a variance of 0, indicating that these columns will likely not add substantially to the clustering. The columns are columns 8 - 12. In later iterations, these columns will be removed. 

# COMMAND ----------

#Train the clustering algorithm using the Features RDD. Try with 4 clusters in the first iteration.
clusters = KMeans.train(featuresRDD, 4, maxIterations=100, initializationMode="random")

# COMMAND ----------

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# COMMAND ----------

#Computer the error 
WSSSE = featuresRDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# COMMAND ----------

#Let's look at the coordinates of the centre points of the clusters, to help understand which features are the most discriminatory. 
d = clusters.clusterCenters
clusterDf = pd.DataFrame(d)
#Pass the feautures list as column labels 
clusterDf.columns = featuresList

# COMMAND ----------

#View the cluster centre coordinates for all four clusters
clusterDf

# COMMAND ----------

#View a stastical summary of the cluster centre coordinates
clusterDf.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Without normalization of the data, it's not very useful to describe mean and distribution of the data. However, based on an visual inspection of the features, the following features appear to be the most important in determining clusters, based on the differences between their cluster coordinates:
# MAGIC * FRACS_FREQ
# MAGIC * IS_50
# MAGIC * Q
# MAGIC * PEN_RATE
# MAGIC 
# MAGIC Most of the blast parameters have very similar cluster centre coordinates. This indicates that the blast parameters are not very useful in helping to determine cluster assignments. 

# COMMAND ----------

# MAGIC %md <a id="evaluation"></a>
# MAGIC ## 6.0) Alternative Iterations of Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1) Increasing K Sequentially 

# COMMAND ----------

# MAGIC %md
# MAGIC In order to evaluate the best number of clusters, the Elbow Method approach is carried out. Starting from k=1, the K-means algorithm is run on the same dataset with increasing k and observing the value of the cost function WSSE. The number of clusters selected should be 1 greater than the 'elbow' on the graph, indicating the biggest drop of WSSE.  

# COMMAND ----------

#Initialize the empty array
cluster_elbow = []
errorGraph = []

# COMMAND ----------

#Define the function which will be used to compute the cost for each iteration
def error_iter(clusters_iter, point):
    center = clusters_iter.centers[clusters_iter.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# COMMAND ----------

#define function to compute WSSE as K increases sequentially 
def num_k_iter(num_k, fRDD):
    for k in range(1,num_k):
      #Train the model, with k increasing from 1 to entered value
      clusters_iter = KMeans.train(fRDD, k, maxIterations=100, initializationMode="random")
      #Calculate the WSSE errors for each k
      WSSE_iter = fRDD.map(lambda point: error_iter(clusters_iter, point)).reduce(lambda x, y: x + y) 
      #Append it to an initialized list   
      errorGraph.append([k,WSSE_iter])   

# COMMAND ----------

#Let's try out the funcion with k from 1 to 10
num_k_iter(10,featuresRDD)

# COMMAND ----------

#Let's see what's in the Cluster Elbow array
cluster_elbow = errorGraph

# COMMAND ----------

#Create a dataframe so we can viusalize the Cluster Elbow graph
cluster_df = sqlContext.createDataFrame(cluster_elbow)
cluster_df.createOrReplaceTempView('elbows')

# COMMAND ----------

# MAGIC %sql -- Plot the number of K vs. the WSSE
# MAGIC SELECT _1 as NumK, _2 as WSSSE
# MAGIC FROM elbows

# COMMAND ----------

# MAGIC %md
# MAGIC The biggest drop occurs after the 2'nd cluster so this analysis suggests that 3 clusters would be the optimal solution for this dataset. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2) With Normalized Columns in Features Dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC Previously it was noted that features have large differences in actual values. For example RQD values are mostly in the range from 80 - 100, while FRACS_FREQ are largely between 2 - 6.These differences may cause discrepancies in the clustering process and should therefore be normalized. 
# MAGIC 
# MAGIC The method used by Hudaverdi [1] to normalize data is the Z-score normalization technique. In this method, each value in the feature table is adjusted by subtracting the column mean and dividing by the column standard deviation.

# COMMAND ----------

#Create a function that will normalize each value in the dataframe. Normalization involves subtracting each value by the column mean and dividing by the column standard deviation. 
def normalize(df):
    features_normalized = df.copy()
    for feature_name in df.columns:
        col_mean = df[feature_name].mean()
        col_std = df[feature_name].std()
        features_normalized[feature_name] = (df[feature_name] - col_mean) / col_std
    return features_normalized

# COMMAND ----------

#Let's see the original features table prior to normalization
features.head()

# COMMAND ----------

#Pass the features to be normalized to the function
normalized_features = normalize(features)
#See the new normalized values 
normalized_features.head()

# COMMAND ----------

# MAGIC %md
# MAGIC After normalization, a few columns appear to produce a lot of NaN values. This occus because all the values in these colums are the same, therefore subtracting the mean results in 0. Let's see how many are NaNs.

# COMMAND ----------

#View the null values in the normalized_features dataframe
normalized_features.isnull().sum()

# COMMAND ----------

#Remove NaN columns from normalized_feautres
normalized_features.drop(['DH_DELAY','HS_DELAY','RS_DELAY','EXPLOSIVE','BLAST_PATTERN'],1,inplace='true')

# COMMAND ----------

#Create a list of the column headers 
normalized_feautresList = list(normalized_features)
#Let's view the dataframe
normalized_features.head()

# COMMAND ----------

#Parse the features pandas dataframe into an RDD of features which will be input into the model
normalized_featuresRDD = sqlContext.createDataFrame(normalized_features).rdd.map(lambda row: np.array([float(item) for item in row]))

# COMMAND ----------

#Run the clustering function 
num_k_iter(10,normalized_featuresRDD)

# COMMAND ----------

#Pass the error graph to a new variable
normalizedElbow = errorGraph

# COMMAND ----------

#Create a spark dataframe so we can viusalize the Cluster Elbow graph
NormCluster_df = sqlContext.createDataFrame(normalizedElbow)
NormCluster_df.createOrReplaceTempView('norm_elbows')

# COMMAND ----------

# MAGIC %sql -- Plot normalized WSSE Error vs. Num K
# MAGIC SELECT _1 as NumK, _2 as WSSSE
# MAGIC FROM norm_elbows

# COMMAND ----------

# MAGIC %md
# MAGIC Based on this graph, the node after largest drop is 3, therefore it's recommended that 3 clusters be used for this dataset. Let's conduct some more exploration with 3 clusters and see how the data look. 

# COMMAND ----------

#Re run the model training with 3 clusters specified
normClusters = KMeans.train(normalized_featuresRDD, 3, maxIterations=100, initializationMode="random")

# COMMAND ----------

#Create an array of cluster centers
normCentres = normClusters.clusterCenters
#Pass the array to a pandas dataframe
normCentresDf = pd.DataFrame(normCentres)

# COMMAND ----------

#Add column headers to allow for visual assessment of features
normCentresDf.columns = normalized_feautresList
#Let's view the centre coordinates of each normalized table
normCentresDf

# COMMAND ----------

# MAGIC %md
# MAGIC The centre coordinates indicate the location of each cluster. Although we don't know the tightness of the cluster, we can use the centre coordinates to visualize the splits among the features.

# COMMAND ----------

#Create a spark dataframe so we can viusalize the Cluster Elbow graph
NormClusterCentre_df = sqlContext.createDataFrame(normCentresDf)
NormClusterCentre_df.createOrReplaceTempView('norm_cluster_cent')

# COMMAND ----------

# MAGIC %sql -- Let's see all the columns in our centre cluster dataframe
# MAGIC SELECT *
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in RQD values among the normalized clusters shows three relatively distinct groupings of clusters 
# MAGIC SELECT RQD
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in FRACS_FREQ values among the normalized clusters shows three distinct clusters.
# MAGIC SELECT FRACS_FREQ
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in IS_50 values among the normalized clusters shows that clusters 0 and 1 are more similar to each other than to 2
# MAGIC SELECT IS_50
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in Q values among the normalized clusters shows that 0,2 are more similar to each other than to 1. 
# MAGIC SELECT Q
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in PEN_RATE values among the normalized clusters shows that clusters 0 and 2 are more similar to each other than to 0
# MAGIC SELECT PEN_RATE
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in RQD values among the normalized clusters shows that clusters 1 and 2 are more similar to each other than to 0
# MAGIC SELECT ROCK_TYPE
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in confinement values among the normalized clusters shows that clusters 0 and 2 are more similar to each other than to 2
# MAGIC SELECT CONFINEMENT
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in ROCK_TYPE values among the normalized clusters shows that clusters 1 and 2 are more similar to each other than to 0
# MAGIC SELECT ROCK_TYPE
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in SB_Ratio values among the normalized clusters shows that only 1 cluster was formed.
# MAGIC SELECT SB_RATIO
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in HB_RATIO values among the normalized clusters shows that only 1 cluster was formed.
# MAGIC SELECT HB_RATIO
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %sql -- Differences in TB_RATIO values among the normalized clusters shows that only 1 cluster was formed. 
# MAGIC SELECT TB_RATIO
# MAGIC FROM norm_cluster_cent

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3) With Just the Geotechnical Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the results of the previous clustering, it appears that many of the blast parameters are not as useful for assigning discriminant clusters. As such, we should try assign clusters with just the geotechnical parameters to see if there are interesting groupings within the data.

# COMMAND ----------

#Create a geotech data features list
normalizedGeotech = normalized_features[['RQD','FRACS_FREQ','Q','PEN_RATE','ROCK_TYPE']]

# COMMAND ----------

#Let's view the new table
normalizedGeotech.head()

# COMMAND ----------

#Ensure it has the correct shape
normalizedGeotech.shape

# COMMAND ----------

#Create the RDD
normalized_geofeaturesRDD = sqlContext.createDataFrame(normalizedGeotech).rdd.map(lambda row: np.array([float(item) for item in row]))

# COMMAND ----------

#Pass it to the clustering function
num_k_iter(10,normalized_geofeaturesRDD)

# COMMAND ----------

#Create an elbow graph with the WSSSE error points
normalized_geoElbow = errorGraph

# COMMAND ----------

#Create a spark dataframe so error points can be visualized
NormgeoCluster_df = sqlContext.createDataFrame(normalized_geoElbow)
NormgeoCluster_df.createOrReplaceTempView('norm_geo_elbows')

# COMMAND ----------

# MAGIC %sql -- Let's view the WSSE vs. Num K
# MAGIC SELECT _1 as NumK, _2 as WSSSE
# MAGIC FROM norm_geo_elbows

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to previous iterations, it appears the best K is 3 indicating that the data can be grouped into three distinct clusters based on the geotechnical features. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0) Conclusions & Recommendations
# MAGIC 
# MAGIC Based on analysis conducted, the following observations are made :<br/>
# MAGIC * The K-means algorithm is useful for grouping data based on underlying similarity 
# MAGIC * Based on running the K-means algorithm with sequentially increasing K, the optimal number of clusters was found to be 3. This was also found with just the geotechnical dataset - which indicates validation of the initial test. This suggests that there are 3 underlying groupings of the data.
# MAGIC * Upon reviewing the centroids of the coordinates, it was typically noted that two of the clusters were closer to each other than the third, however this alternated depending on which parameter was assessed. 
# MAGIC * The blast parameters did not provide useful discriminant data to perform the clustering. This was shown by utilizing just the geotechnical data and obtaining 3 clusters - the same result as conducted with blast parameters. This suggests that most blasts have very similar data points, and is supported by the graphical plots depicted in Section 3.
# MAGIC 
# MAGIC 
# MAGIC The following recommendations are made regarding data management and future work:
# MAGIC * The dataset used for the final clustering was relatively small - only 217 points. This was as a result of data processing and cleaning which resulted in the removal of many datapoints. In order to build better models, a larger more accurate dataset should be collected. 
# MAGIC * The dataset used for the analysi (smry_bmm) contained many data that had been averaged from one location across many different points. This is understandable considering the limitations of data collections at a mine, however additional efforts should be made to collect discrete geospatial data which are representative - for example the RQD, Q, IS_50 attached to individual blast hole data points. 
# MAGIC * This analysis provided distinct clusters into which the data can be grouped based on similarity. Further work can develop regression models to predict movement or fragmentation based on the clusters. These will likely result in better predictive models. 
# MAGIC * Documentation for ML Lib is still being developed (or difficult to find) as such, detailed analysis on clusters was not possible. For example it would have been useful to see cluster assignments and relate it back to individual data points. 

# COMMAND ----------

# MAGIC %md <a id="bibliography"></a>
# MAGIC ## 7.0) Bibliography
# MAGIC <strong>Selected Paper:</strong><br/>
# MAGIC <a id="ref1"></a>
# MAGIC [1] T. Hudaverdi. Application of Multivariate Analysis for Prediction of Blast Induced Ground Vibrations. In Soil Dynamics and Earthquake Engineering. Pages 300 – 308, 2012.<br/><br/>
# MAGIC <a id="ref2"></a>
# MAGIC 
# MAGIC <strong>Also cited in literature review:</strong><br/>
# MAGIC [2] M. Rezaei, M. Monjezi, A. Varjani. Development of a fuzzy model to predict flyrock in surface mining. In Safety Science, pages 298-305, SS49, 2011<br/><br/>
# MAGIC <a id="ref3"></a>
# MAGIC [3] R. Trivedi, T. Singh, N. Gupta. Prediction of Blast-Induced Flyrock in Opencast Mines Using ANN and ANFIS. <br/><br/>
# MAGIC <a id="ref4"></a>
# MAGIC [4] CVB Cunningham. The Kuz-Ram fragmentation model – 20 years on. In Brighton Conference Proceedings, pages 201 – 210, 2005 <br/><br/>
# MAGIC <a id="ref5"></a>
# MAGIC [5] E. Hamdi, J. du Mouza. A methodology for rock mass characterization and classification to improve blast results. International Journal of Rock Mechanics and Mining Sciences. 42 Pages 177 – 194, 2005. <br/><br/>
# MAGIC <a id="ref6"></a>
# MAGIC [6] W. Zhou and N. Maerz. Implementation of multivariate clustering methods for characterizing discontinuities data from scanlines and oriented boreholes. Computers and Geosciences. Vol 28 Pages 827 – 839, 2002.<br/><br/>
# MAGIC <a id="ref7"></a>
# MAGIC [7] A Jain, M Murty, and P Flynn. Data Clustering : A Review. In ACM Digital Library. Vol 31, Iss 3. Pages 264 – 323, 1999.<br/><br/>
# MAGIC <a id="ref8"></a>
# MAGIC [8] A Jain. Data Clustering : 50 Years Beyond K-Means. Published in 19th International conference in Pattern Recognition (ICPR) . Vol 31, Iss. 8, Pages 651 – 666, 2010.<br/><br/>
