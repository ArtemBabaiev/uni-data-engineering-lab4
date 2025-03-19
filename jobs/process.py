from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, avg, sum, count, month, col, dense_rank, max, expr

basepath = "/opt/bitnami/spark/jobs"


# trip_id,
# start_time,
# end_time,
# bikeid,
# tripduration,
# from_station_id,
# from_station_name,
# to_station_id,
# to_station_name,
# usertype,
# gender,
# birthyear

def write_file(res: DataFrame, csv_name):
    res.write.csv(f"{basepath}/out/{csv_name}", header=True, mode="overwrite")


# Яка «середня» тривалість поїздки на день?
def avg_trip_duration(data_frame: DataFrame):
    data_frame = data_frame.withColumn("date", to_date("start_time", "yyyy-MM-dd"))

    result = (data_frame.groupby("date")
              .agg(avg("tripduration").alias("avg_trip_duration")))

    write_file(result, "avg_trip_duration.csv")


# Скільки поїздок було здійснено кожного дня?
def trips_each_day(data_frame: DataFrame):
    data_frame = data_frame.withColumn("date", to_date("start_time", "yyyy-MM-dd"))
    result = (data_frame
              .groupby("date")
              .agg(count("*").alias("trips_each_day")))

    write_file(result, "trips_each_day.csv")


# Яка була найпопулярніша початкова станція для кожного місяця?
def most_popular_starting_station_for_each_month(data_frame: DataFrame):
    data_frame = data_frame.withColumn("month", month("start_time"))

    trips_per_month_per_station = (data_frame
                                   .groupby("month", "from_station_name")
                                   .agg(count("from_station_name").alias("trip_count")))

    window_spec = Window.partitionBy("month").orderBy(col("trip_count").desc())

    result = (trips_per_month_per_station.withColumn("rank", dense_rank().over(window_spec))
              .filter(col("rank") == 1)
              .drop("rank"))

    write_file(result, "most_popular_starting_station_for_each_month.csv")


# Які станції входять у трійку лідерів станцій для поїздок кожного дня протягом останніх двох тижнів?
def top_3_stations_for_daily_trips_over_past_two_weeks(data_frame: DataFrame):
    data_frame = data_frame.withColumn("date", to_date("start_time", "yyyy-MM-dd"))
    latest_date = data_frame.select(max("date")).collect()[0][0]

    df_filtered = data_frame.filter(col("date") >= latest_date - expr("INTERVAL 14 DAYS"))

    trip_count_df = (df_filtered
                     .groupBy("date", "to_station_name")
                     .agg(count("*").alias("trip_count")))

    window_spec = Window.partitionBy("date").orderBy(col("trip_count").desc())

    result = (trip_count_df
              .withColumn("rank", dense_rank().over(window_spec))
              .filter(col("rank") <= 3)
              .drop("rank")
              .sort(["date", "trip_count"]))

    write_file(result, "top_3_stations_for_daily_trips_over_past_two_weeks.csv")


# Чоловіки чи жінки їздять довше в середньому?
def men_or_women_drive_longer_on_average(data_frame: DataFrame):
    result = (data_frame
              .groupby("gender")
              .agg(avg("tripduration").alias("avg_trip_duration"))
              .filter(col("gender").isin("Male", "Female"))
              .orderBy(col("avg_trip_duration").desc())
              .limit(1))
    write_file(result, "men_or_women_drive_longer_on_average.csv")


spark = SparkSession.builder.appName("Driving Trips").getOrCreate()

df = spark.read.csv(f"{basepath}/Divvy_Trips_2019_Q4.csv", header=True, inferSchema=True)

avg_trip_duration(df)
trips_each_day(df)
most_popular_starting_station_for_each_month(df)
top_3_stations_for_daily_trips_over_past_two_weeks(df)
men_or_women_drive_longer_on_average(df)

spark.stop()
