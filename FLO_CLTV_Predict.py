############################ İŞ PROBLEMİ ###############################

# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan
# yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi
# gerekmektedir.

######################## Veri Seti Hikayesi ############################

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# 12 Değişken - 19.945 Gözlem - 2.7MB

# DEĞİŞKENLER
# master_id Eşsiz                   = Müşteri Numarası
# order_channel                     = Alışveriş yapılan platforma ait hangi kanalın kullanıldığı
# (Android, ios, Desktop, Mobile)
# last_order_channel                = En son alışverişin yapıldığı kanal
# first_order_date                  = Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date                   = Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online            = Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline           = Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online       = Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline      = Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline = Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online  = Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12       = Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import numpy as np
import seaborn as sns
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv("DATASETS/flo_data_20k.csv")
df = df_.copy()
df.head()
#                             master_id    order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online       interested_in_categories_12
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline       2020-10-30      2021-02-26             2021-02-21              2021-02-26                         4.00                          1.00                             139.99                            799.38                           [KADIN]
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile       2017-02-08      2021-02-16             2021-02-16              2020-01-10                        19.00                          2.00                             159.97                           1853.58  [ERKEK, COCUK, KADIN, AKTIFSPOR]
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App       2019-11-27      2020-11-27             2020-11-27              2019-12-01                         3.00                          2.00                             189.97                            395.35                    [ERKEK, KADIN]
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App       2021-01-06      2021-01-17             2021-01-17              2021-01-06                         1.00                          1.00                              39.99                             81.98               [AKTIFCOCUK, COCUK]
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop       2019-08-03      2021-03-07             2021-03-07              2019-08-03                         1.00                          1.00                              49.99                            159.99                       [AKTIFSPOR]

df.info()
# RangeIndex: 19945 entries, 0 to 19944
# Data columns (total 12 columns):
#  #   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
#  0   master_id                          19945 non-null  object
#  1   order_channel                      19945 non-null  object
#  2   last_order_channel                 19945 non-null  object
#  3   first_order_date                   19945 non-null  object
#  4   last_order_date                    19945 non-null  object
#  5   last_order_date_online             19945 non-null  object
#  6   last_order_date_offline            19945 non-null  object
#  7   order_num_total_ever_online        19945 non-null  float64
#  8   order_num_total_ever_offline       19945 non-null  float64
#  9   customer_value_total_ever_offline  19945 non-null  float64
#  10  customer_value_total_ever_online   19945 non-null  float64
#  11  interested_in_categories_12        19945 non-null  object
# dtypes: float64(4), object(8)

df.isnull().sum()
# master_id                            0
# order_channel                        0
# last_order_channel                   0
# first_order_date                     0
# last_order_date                      0
# last_order_date_online               0
# last_order_date_offline              0
# order_num_total_ever_online          0
# order_num_total_ever_offline         0
# customer_value_total_ever_offline    0
# customer_value_total_ever_online     0
# interested_in_categories_12          0
# dtype: int64

df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
#                                     count   mean    std   min    1%    5%   10%    25%    50%    75%     90%     95%     99%      max
# order_num_total_ever_online       19945.00   3.11   4.23  1.00  1.00  1.00  1.00   1.00   2.00   4.00    7.00   10.00   20.00   200.00
# order_num_total_ever_offline      19945.00   1.91   2.06  1.00  1.00  1.00  1.00   1.00   1.00   2.00    4.00    4.00    7.00   109.00
# customer_value_total_ever_offline 19945.00 253.92 301.53 10.00 19.99 39.99 59.99  99.99 179.98 319.97  519.95  694.22 1219.95 18119.14
# customer_value_total_ever_online  19945.00 497.32 832.60 12.99 39.99 63.99 84.99 149.98 286.46 578.44 1082.04 1556.73 3143.81 45220.13

# Yukarıdaki betimleme tablosunda da görüldüğü üzere sayısal değişkenlerin hepsinde aykırı değer mevcuttur.
# Bu sebeple bu aykırı değerleri baskılamamız gerekiyor.

# Boxplot yöntemi ile aykırı değerleri görme
for col in df.columns:
    if df[col].dtypes != "O":
        print(sns.boxplot(x=df[col]))
        print(plt.show(block=True))

# Eşik Değer Belirleme
def outlier_thresholds(dataframe, variable):
    # q1 ve q3 değerlerinin ön tanımlı değeri 0.25 / 0.75 tir.
    # Ancak biz aykırı değerleri sadece ucundan traşlamak istiyoruz.
    # Bu sayede gereksiz veri kaybının önüne geçmiş oluyoruz.
    q1 = dataframe[variable].quantile(0.01) # 0.25
    q3 = dataframe[variable].quantile(0.99) # 0.75
    IQR_Range = q3 - q1
    up_limit = q3 + 1.5 * IQR_Range
    low_limit = q1 - 1.5 * IQR_Range
    return low_limit, up_limit

# Aykırı Değerleri Baskılama (Eşik değerlere eşitliyoruz)
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# Aykırı değerleri saptamak için kullanıyoruz.
def check_outlier(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] > up_limit) | (dataframe[variable] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.columns:
    if df[col].dtypes != "O":
        print(col, check_outlier(df, col))
# order_num_total_ever_online       True
# order_num_total_ever_offline      True
# customer_value_total_ever_offline True
# customer_value_total_ever_online  True


# Aykırı değerleri baskılayalım.
for col in df.columns:
    if df[col].dtypes != "O":
        replace_with_thresholds(df, col)

df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
#                                    count   mean    std   min    1%    5%   10%    25%    50%    75%     90%     95%     99%     max
# order_num_total_ever_online       19945.00   3.09   3.81  1.00  1.00  1.00  1.00   1.00   2.00   4.00    7.00   10.00   20.00   48.00
# order_num_total_ever_offline      19945.00   1.89   1.43  1.00  1.00  1.00  1.00   1.00   1.00   2.00    4.00    4.00    7.00   16.00
# customer_value_total_ever_offline 19945.00 251.92 251.02 10.00 19.99 39.99 59.99  99.99 179.98 319.97  519.95  694.22 1219.95 3020.00
# customer_value_total_ever_online  19945.00 489.71 632.61 12.99 39.99 63.99 84.99 149.98 286.46 578.44 1082.04 1556.73 3143.81 7800.00

# Her bir müşterinin toplam alışveriş sayısı ve harcamasını bulmak için online ve offline
#olarak yapılan harcama ve alışveriş sayılarını toplayak yeni değişkenlere atayalım.

# Toplam Alışveriş Sayısı
df["total_order_num"] = df["order_num_total_ever_offline"] + \
                        df["order_num_total_ever_online"]
# Toplam Harcama
df["total_price"] = df["customer_value_total_ever_offline"] + \
                             df["customer_value_total_ever_online"]

# Tarih değişkenlerinin tipini kontrol edelim.
for col in df.columns:
    if "date" in col:
        print(col, df[col].dtypes)
# first_order_date        object
# last_order_date         object
# last_order_date_online  object
# last_order_date_offline object

# Tarih değişkenlerinin tipini date'e çevirelim.
# Çözüm 1
for col in df.columns:
    if "date" in col:
        df[col] = df[col].apply(pd.to_datetime)

# Çözüm 2
contains_date =  df.columns[df.columns.str.contains("date")]
df[contains_date] = df[contains_date].apply(pd.to_datetime)

# Artık verimiz hazır. CLTV veri yapısını oluşturabiliriz.

# Analiz tarihimizi atayalım.
df["last_order_date"].max()
# Timestamp('2021-05-30 00:00:00')
analysis_date = dt.datetime(2021, 6, 1)

# recency   : Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde, dinamik)
# T         : Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency : tekrar eden toplam satın alma sayısı (frequency>1)
# monetary  : satın alma başına ortalama kazanç

# customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerini oluşturalım.

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                        "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
                        "T_weekly": ((analysis_date - df["first_order_date"]).dt.days)/7,
                        "frequency": df["total_order_num"],
                        "monetary_cltv_avg": df["total_price"] / df["total_order_num"]})

cltv_df.head()
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88
# 2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06
# 3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99

# BG/NBD Modelinin Kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])
# <lifetimes.BetaGeoFitter: fitted with 19945 subjects, a: 0.00, alpha: 76.17, b: 0.00, r: 3.66>

# Deneme 1 = 3 ay içerisinde müşterilerden beklenen satın almaları tahmin edelim.

cltv_df["exp_sales_3_month"] = bgf.predict(4*3, # 1 içerisindeki hafta sayısı * ay sayısı
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"])

# Deneme 2 = 6 ay içerisinde müşterilerden beklenen satın almaları tahmin edelim.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"])

cltv_df.head(10)
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87               0.97               1.95
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88               0.98               1.97
# 2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06               0.67               1.34
# 3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98               0.70               1.40
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99               0.40               0.79
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f               120.86    132.29       3.00              66.95               0.38               0.77
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f                32.57     64.86       4.00              93.98               0.65               1.30
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f                12.71     54.57       2.00              81.81               0.52               1.04
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f                58.43     70.71       5.00             210.94               0.71               1.42
# 9  1143f032-440d-11ea-8b43-000d3a38a36f                61.71     96.00       2.00              82.98               0.39               0.79

# Gamma-Gamma Modelini kurarak; müşterilerin ortalama bırakacakları değeri tahminleyelim

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary_cltv_avg"])
cltv_df.head(10)
#                             customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87               0.97               1.95             193.63
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88               0.98               1.97              96.67
# 2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06               0.67               1.34             120.97
# 3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98               0.70               1.40              67.32
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99               0.40               0.79             114.33
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f               120.86    132.29       3.00              66.95               0.38               0.77              71.35
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f                32.57     64.86       4.00              93.98               0.65               1.30              98.13
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f                12.71     54.57       2.00              81.81               0.52               1.04              89.57
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f                58.43     70.71       5.00             210.94               0.71               1.42             217.30
# 9  1143f032-440d-11ea-8b43-000d3a38a36f                61.71     96.00       2.00              82.98               0.39               0.79              90.81

# 6 Aylık CLTV’nin Hesaplanması

cltv_df["CLTV"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,    # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_df[["customer_id", "CLTV"]].head(10)
#                             customer_id    CLTV
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   395.73
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   199.43
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   170.22
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   98.95
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f   95.01
# 5  e585280e-aae1-11e9-a2fc-000d3a38a36f   57.43
# 6  c445e4ee-6242-11ea-9d1a-000d3a38a36f   134.28
# 7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f   97.70
# 8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f   322.73
# 9  1143f032-440d-11ea-8b43-000d3a38a36f   75.22

# CLTV değerlerine göre Segmentlerimi oluşturalım.

cltv_df["SEGMENT"] = pd.qcut(cltv_df["CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("SEGMENT").\
    agg(["count", "sum", "mean"]).T
# SEGMENT                           D         C         B          A
# recency_cltv_weekly count   4987.00   4986.00   4986.00    4986.00
#                     sum   693193.86 461850.86 408794.00  336191.71
#                     mean     139.00     92.63     81.99      67.43
# T_weekly            count   4987.00   4986.00   4986.00    4986.00
#                     sum   808807.71 562512.14 500228.00  411592.86
#                     mean     162.18    112.82    100.33      82.55
# frequency           count   4987.00   4986.00   4986.00    4986.00
#                     sum    18795.00  21962.00  25392.00   33140.00
#                     mean       3.77      4.40      5.09       6.65
# monetary_cltv_avg   count   4987.00   4986.00   4986.00    4986.00
#                     sum   464547.05 627181.65 800933.96 1140952.07
#                     mean      93.15    125.79    160.64     228.83
# exp_sales_3_month   count   4987.00   4986.00   4986.00    4986.00
#                     sum     2039.16   2619.88   2997.11    3854.31
#                     mean       0.41      0.53      0.60       0.77
# exp_sales_6_month   count   4987.00   4986.00   4986.00    4986.00
#                     sum     4078.33   5239.77   5994.22    7708.63
#                     mean       0.82      1.05      1.20       1.55
# exp_average_value   count   4987.00   4986.00   4986.00    4986.00
#                     sum   492172.44 659401.45 837650.88 1186787.64
#                     mean      98.69    132.25    168.00     238.02
# cltv                count   4987.00   4986.00   4986.00    4986.00
#                     sum   400657.96 689621.18 994870.78 1806505.09
#                     mean      80.34    138.31    199.53     362.32
# CLTV                count   4987.00   4986.00   4986.00    4986.00
#                     sum   400657.96 689621.18 994870.78 1806505.09
#                     mean      80.34    138.31    199.53     362.32

cltv_df.groupby("SEGMENT").\
    agg({"CLTV":["describe"]})
#            CLTV
#         describe
#            count   mean    std    min    25%    50%    75%     max
# SEGMENT
# D        4987.00  80.34  21.73  12.11  65.06  83.72  98.52  112.25
# C        4986.00 138.31  15.32 112.25 125.09 138.03 151.44  165.47
# B        4986.00 199.53  21.21 165.47 181.13 198.10 217.10  240.06
# A        4986.00 362.32 158.42 240.09 270.70 312.93 395.16 3327.78

# 2 Adet SEGMENT için yorumlama yapalım.

# A segmentini ele aldığımızda diğer segmentlere göre standart sapmasının yüksek olduğunu görmekteyiz.
# Buna göre bakıldığında A segmentinin ayrı olarak alınarak A özelinde tekrar segmentlere bölünerek
# buna göre ürünler özelinde tekrar bir fiyatlandırma ya da kategori düzenlenmesi yapılabilir.

# D segmentini incelediğimizde ise min ve max değerleri arasında büyük farklar olduğunu görüyoruz.
# Buradanda min ve max ları katerogiler özelinde inceleyerek min satışların hangi kategorilerde
# olduğu tespit ederek onlara özel kampanyalar ve fiyatlandırma yapılabilir