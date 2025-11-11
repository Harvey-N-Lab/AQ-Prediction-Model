import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

RAW_OPENAQ = sys.argv[1]  # s3://.../raw/openaq/
RAW_WX     = sys.argv[2]  # s3://.../raw/weather/
OUT_PATH   = sys.argv[3]  # s3://.../processed/

spark = SparkSession.builder.appName("aqi-etl").getOrCreate()

openaq = spark.read.json(RAW_OPENAQ)
wx     = spark.read.json(RAW_WX)

m = openaq.selectExpr("results as r").selectExpr("explode(r) as row")     .selectExpr("row.location as station",
                "row.coordinates.latitude as lat",
                "row.coordinates.longitude as lon",
                "explode(row.measurements) as meas")     .selectExpr("station","lat","lon",
                "meas.parameter as param",
                "meas.value as value",
                "to_timestamp(meas.lastUpdated) as ts")

pm = m.groupBy("station","lat","lon","ts").pivot("param").agg(F.first("value"))

wx_flat = wx.selectExpr(
    "null as station",
    "coord.lat as lat",
    "coord.lon as lon",
    "to_timestamp(from_unixtime(dt)) as ts",
    "main.temp as temp",
    "main.humidity as humidity",
    "wind.speed as wind_speed",
    "wind.deg as wind_deg",
    "main.pressure as pressure"
)

pm_h = pm.withColumn("ts_hour", F.date_trunc("hour", F.col("ts")))
wx_h = wx_flat.withColumn("ts_hour", F.date_trunc("hour", F.col("ts")))

df = (pm_h.join(wx_h, on="ts_hour", how="inner")
        .withColumn("hour", F.hour("ts_hour"))
        .withColumn("dow", F.date_format("ts_hour","u").cast("int"))
        .withColumn("is_weekend", (F.col("dow")>=6).cast("int"))
        .withColumn("month", F.month("ts_hour")))

w = Window.partitionBy("station").orderBy("ts_hour")
df = (df
      .withColumn("pm25_lag_1h", F.lag("pm25",1).over(w))
      .withColumn("pm25_roll_mean_6h", F.avg("pm25").over(w.rowsBetween(-6,-1))))

(df
 .write.mode("overwrite")
 .partitionBy("year","month","day")
 .parquet(OUT_PATH))
