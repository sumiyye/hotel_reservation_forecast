import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 160)


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv("PROJE_CALISMA/Hotel Reservations.csv")
df.head()

#########  EDA ##########
def check_df(dataframe, head = 5):
    print("###########shape###########")
    print(dataframe.shape)
    print("###########dtypes###########")
    print(dataframe.dtypes)
    print("###########head###########")
    print(dataframe.head(head))
    print("###########tail###########")
    print(dataframe.tail(head))
    print("########### NA ###########")
    print(dataframe.isnull().sum())
    print("########### Quantile ###########")
    print(dataframe.describe([0,0.05,0.50, 0.95, 0.99, 1]).T)


check_df(df)
df.describe().T

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def outlier_tresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def check_outlier(dataframe, col_name):
    up_limit, low_limit = outlier_tresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    up, low = outlier_tresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, check_outlier(df, col))

def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe[na_cols].shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_cols

missing_values_table(df)

def cat_value_count(dataframe , col):
    for col in cat_cols:
        value =dataframe[col].nunique()
        print(f'nunique degeri : {col, value}')

cat_value_count(df,cat_cols)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#######################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df,col)


######### Data Preprocessing & Feature Engineering

df.loc[(df["arrival_year"]==2018) & (df["arrival_month"]==2) & (df["arrival_date"]==29),['arrival_date']] = '28'

df['NEW_arrival_time'] = pd.to_datetime(df['arrival_year'].astype(str) + '-' + df['arrival_month'].astype(str) + '-' + df['arrival_date'].astype(str),format='%Y-%m-%d', errors='coerce')

df.head()
df = df.set_index('NEW_arrival_time')

## toplam kisi sayisi
df['NEW_no_of_total_person'] = df["no_of_children"] + df["no_of_adults"]
## toplam kalinan gun sayisi
df['NEW_total_number_of_stayed_days'] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
## kac hafta kalmis kac gun kalmis
df['NEW_number_weeks'], df['NEW_number_days'] = divmod(df['NEW_total_number_of_stayed_days'], 7)
## season
df["NEW_season"] = ["Winter" if col <= 2 else "Spring" if 3 <= col <= 5 else "Summer" if 6<= col <= 8 else "Fall"
                            if 9 <= col <=11 else "Winter" for col in df["arrival_month"]]

## yemek planı var mı yok mu ?
df["NEW_meal_plan"] = [0 if col=='Not Selected' else 1 for col in df["type_of_meal_plan"]]
##  oda segmenti
df["NEW_Room_Type"] = pd.qcut(df['avg_price_per_room'].rank(method="first"), 3, labels=['Economic_Room', 'Standard_Room', 'Luxury_Rooom'])
## Churn degerini sayisallastir.
df.booking_status = df.booking_status.replace({"Not_Canceled":0, "Canceled":1})

## gelme durumlarina gore derecelendiriyor
grby_month_book = df.groupby(["arrival_month","booking_status"]).count().reset_index()
list_of_months = []
for i in range(1,13):
    x = grby_month_book[grby_month_book.arrival_month == i]
    rate = x[x.booking_status == 1].Booking_ID.values / x[x.booking_status == 0].Booking_ID.values * 100
    list_of_months.append(rate)
pd.Series(list_of_months)
df["NEW_arrival_month_rate"] = df.arrival_month.replace({3:1, 4:1, 5:1, 6:1, 7:1, 9:1, 1:0, 2:0, 8:0, 10:0, 11:0, 12:0})

#Misafir başına düşen ortalama fiyatı hesaplama:
df['NEW_average_price_per_guest'] = df['avg_price_per_room'] / df['NEW_no_of_total_person']

# Booking Time Category (booking_time_category) sütunu oluşturma
df['booking_time_category'] = pd.cut(df['lead_time'], bins=[-float('inf'), 30, 60, float('inf')], labels=['Erken', 'Orta', 'Gec'])

##  temel bilesen analizi degerleri

pca = PCA(n_components=1)
NEW_pca_no_people = pca.fit_transform(df[["no_of_adults","no_of_children"]])
df["NEW_pca_no_people"] = NEW_pca_no_people

NEW_pca_no_week = pca.fit_transform(df[["no_of_weekend_nights","no_of_week_nights"]])
df["NEW_no_of_week_days"] = NEW_pca_no_week


# özel istek var mı yok mu

df["NEW_flag_special_requests"] = [1 if col > 0 else 0 for col in df["no_of_special_requests"]]

df['NEW_Total_Price'] = (df["no_of_weekend_nights"] + df["no_of_week_nights"]) * df["avg_price_per_room"]
df["NEW_Total_Price_Per"] = df["NEW_Total_Price"] / df["NEW_no_of_total_person"]
df.shape
df.head()

df.drop("Booking_ID", axis=1, inplace=True)



cols = [col for col in df.columns if col not in ["booking_status",
                                                 "type_of_meal_plan",
                                                 "room_type_reserved",
                                                 "market_segment_type",
                                                 "NEW_season",
                                                 "NEW_Room_Type",
                                                 "booking_time_category"]]



for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

encoder = ["type_of_meal_plan","room_type_reserved","market_segment_type","NEW_season","NEW_Room_Type", "booking_time_category"]

def one_hot_encoding(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoding(df, encoder)
df.head()
y = df["booking_status"]
X = df.drop("booking_status",  axis=1)


####
# LightGBM
######

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#Hiperparametre öncesi
#cv_results['test_accuracy'].mean()
#Out[62]: 0.8882426575886049
#cv_results['test_f1'].mean()
#Out[63]: 0.8222022770269668
#cv_results['test_roc_auc'].mean()
#Out[64]: 0.9500194200537152


#Hipermatretre sonrası
#Out[69]: 0.9020262097724132
#cv_results['test_f1'].mean()
#Out[70]: 0.8456278486143068
#cv_results['test_roc_auc'].mean()
#Out[71]: 0.9583852017116479

df['NEW_arrival_time'] = df.index

# Dizin sütununu kaldırma
df.reset_index(drop=True, inplace=True)

# Sonucu kontrol etme
print(df)

# Churn olan müşterileri belirlemek için gereken koşulu tanımlayın
churn_musteriler = (df['booking_status'] == 1)

# Gruplara ait mailleri oluşturmak için fonksiyonlar tanımlayın

def cocuklu_kisilere_mail(row):
    cocuklu_kisilere_churn_maili = "Konu: Unutulmaz Bir Tatil İçin Sizi ve Çocuklarınızı Bekliyoruz!" \
    "Merhaba," \
    "Siz ve değerli aileniz için unutulmaz bir tatil deneyimi sunmak istiyoruz! " \
    "Çocuklarınızla birlikte keyifli ve eğlenceli bir konaklama için sizi otelimizde ağırlamaktan mutluluk duyarız." \
    "Otelimiz, çocuklu ailelerin rahat ve güvenli bir tatil geçirmesi için özel olarak tasarlanmıştır. " \
    "Çocuk dostu aktiviteler, oyun alanları ve profesyonel çocuk bakım hizmetleri sunmaktayız. " \
    "Ayrıca, özel aile odalarımızda geniş ve konforlu bir konaklama imkanı bulabilirsiniz." \
    "Detaylı bilgiler, rezervasyon yapma ve özel talepleriniz için bizimle iletişime geçebilirsiniz. " \
    "Sizin ve çocuklarınızın unutulmaz bir tatil geçirmesi için buradayız." \
    "Keyifli ve eğlenceli bir tatil dileriz!" \
    "Otel Ekibi"

    return cocuklu_kisilere_churn_maili

def tek_basina_konaklayanlara_mail(row):
   tek_basina_konaklayanlara_churn_maili = "Konu: Yalnız Seyahat Eden Misafirlerimize Özel Fırsatlar!" \
    "Merhaba," \
    "Tek başına seyahat eden misafirlerimize özel fırsatlar ve konforlu konaklama imkanları sunuyoruz! " \
    "Yalnız seyahat etmenin keyfini çıkarırken rahat ve huzurlu bir ortamda konaklamak isterseniz, " \
    "sizleri otelimizde ağırlamaktan mutluluk duyarız. Otelimizdeki modern ve şık odalarımızda kendinizi evinizde hissedeceksiniz. " \
    "Rahat yataklar, çalışma alanları ve dinlenme alanlarıyla konforlu bir konaklama imkanı sunmaktayız. " \
    "Ayrıca, otelimizin sunduğu sosyal alanlarda diğer misafirlerle tanışabilir ve keyifli zamanlar geçirebilirsiniz." \
    "Rezervasyonunuzu yapmak veya detaylı bilgi almak için bizimle iletişime geçebilirsiniz. " \
    "Tek başına seyahat eden misafirlerimize özel fırsatlarımızı kaçırmayın!" \
    "Keyifli ve unutulmaz bir konaklama dileriz!" \
    "Otel Ekibi"
    return tek_basina_konaklayanlara_churn_maili

def arkadas_grubu_mail(row):
    arkadas_grubu_churn_maili = "Konu: Arkadaşlarınızla Keyifli Bir Tatil Deneyimi İçin Sizi Bekliyoruz!" \
        "Merhaba," \
        "Siz ve değerli arkadaş grubunuzla keyifli bir tatil deneyimi yaşamak için sizi otelimizde ağırlamaktan mutluluk duyarız! " \
        "Arkadaşlarınızla birlikte unutulmaz anılar biriktirebileceğiniz bir konaklama için buradayız." \
        "Otelimiz, geniş ve rahat odalarıyla arkadaş gruplarının ihtiyaçlarını karşılamak üzere tasarlanmıştır. " \
        "Grup indirimleri, özel aktiviteler ve sosyal alanlarımızda eğlenceli vakit geçirebilirsiniz. " \
        "Ayrıca, otelimizin sunduğu restoran, bar ve gece hayatı seçenekleriyle keyifli bir tatil deneyimi yaşayabilirsiniz." \
        "Detaylı bilgiler, rezervasyon yapma ve grup indirimlerimiz hakkında bilgi almak için bizimle iletişime geçebilirsiniz. " \
        "Arkadaşlarınızla unutulmaz bir tatil deneyimi için hazırız!" \
        "Keyifli ve eğlenceli bir tatil dileriz!" \
        "Otel Ekibi"
    return arkadas_grubu_churn_maili

def yeni_evli_ciftlere_mail(row):
    yeni_evli_ciftlere_churn_maili = "Konu: Romantik Bir Balayı Deneyimi İçin Sizi Bekliyoruz!" \
        "Merhaba," \
        "Siz ve sevgili eşiniz için romantik bir balayı deneyimi sunmak istiyoruz! " \
        "Evlilik hayatınızın başlangıcında unutulmaz anılar biriktirebileceğiniz bir konaklama için otelimizi tercih edebilirsiniz." \
        "Otelimiz, balayı çiftleri için özel olarak tasarlanmış romantik ortamlar sunmaktadır. Özel oda dekorasyonları, " \
        "romantik akşam yemekleri ve spa hizmetleriyle size unutulmaz bir deneyim yaşatmayı hedefliyoruz. " \
        "Ayrıca, otelimizin sunduğu aktiviteler ve çevredeki romantik mekanlarla dolu bir tatil geçirebilirsiniz." \
        "Balayı rezervasyonunuzu yapmak veya detaylı bilgi almak için bizimle iletişime geçebilirsiniz. " \
        "Sizi romantik bir balayı deneyimiyle karşılamaktan mutluluk duyarız!" \
        "Romantik ve unutulmaz bir tatil dileriz!" \
        "Otel Ekibi"
    return yeni_evli_ciftlere_churn_maili

def is_gezisi_mail(row):
    is_gezisi_churn_maili = "Konu: İş Gezisi İçin Konforlu ve Verimli Bir Konaklama Deneyimi İçin Sizi Bekliyoruz!" \
        "Merhaba," \
        "Siz ve değerli iş arkadaşlarınız için iş gezilerinizde konforlu ve verimli bir konaklama deneyimi sunmak istiyoruz! " \
        "Rezervasyonunuzun onaylandığını görmek bizi heyecanlandırdı. Birlikte iş gezilerinizi daha keyifli ve başarılı hale " \
        "getirmeye hazır olun.Otelimiz, iş gezilerine uygun modern ve işlevsel odalarıyla donatılmıştır. " \
        "Konforlu çalışma alanları, hızlı internet bağlantısı, toplantı salonları ve daha fazlası gibi birçok iş odaklı hizmet " \
        "sunmaktayız. Ayrıca, konumumuz iş merkezlerine yakınlığıyla da avantaj sağlamaktadır.İş geziniz boyunca size yardımcı " \
        "olabileceğimiz herhangi bir konu olursa, lütfen çekinmeden bize ulaşın. Sizin ve iş arkadaşlarınızın konforlu ve " \
        "verimli bir iş gezisi deneyimi yaşamasını diliyoruz." \
        "Başarılı iş gezileri dileriz!" \
        "Otel Ekibi"
    return is_gezisi_churn_maili

def daimi_musteri_mail(row):
    churn_olmayan_musteri_maili = "Merhaba," \
        "Uzun süredir sadık bir müşterimiz olduğunuz için teşekkür ederiz. Müşteri memnuniyeti bizim için en önemli önceliktir " \
        "ve sizin gibi değerli müşterilerimize hizmet vermekten gurur duyuyoruz.Bu özel fırsatı kaçırmamanız için size küçük " \
        "bir promosyon hediyesi sunuyoruz. İlerleyen rezervasyonlarınızda %10 indirim fırsatı sizleri bekliyor. " \
        "Bu indirimden faydalanmak için rezervasyon yaparken promosyon kodunu kullanmanız yeterli olacaktır: PROMO10" \
        "Size tekrar teşekkür eder, güzel bir konaklama geçirmenizi dileriz." \
        "Saygılarımızla," \
        "Otel İşletmesi Ekibi"
    return churn_olmayan_musteri_maili

def genel_musteri_mail(row):
    churn_olan_genel_musteri_maili = "Sayın [Müşterinin Adı/Soyadı]," \
        "Bu maili almanızın bizim için üzücü bir sebep olduğunu öncelikle belirtmek isteriz. " \
        "Rezervasyonunuzun iptal edildiğini görmek bizi üzdü ve size bu konuda bilgi vermek istiyoruz." \
        "Rezervasyonunuzun iptal edilmesiyle ilgili herhangi bir neden veya endişe varsa, lütfen bize bildirin. " \
        "Müşterilerimizin memnuniyeti bizim önceliğimizdir ve herhangi bir sorunu çözmek için elimizden geleni yapmaktan " \
        "mutluluk duyarız.Ayrıca, gelecekteki rezervasyonlarınız için size özel indirimler ve fırsatlar sunmaktan mutluluk duyarız. " \
        "Eğer tekrar otelimizde konaklamayı düşünürseniz, sizi ağırlamaktan mutluluk duyacağız." \
        "Sorularınız veya herhangi bir konuda yardım gerektiğinde lütfen bizimle iletişime geçmekten çekinmeyin. " \
        "Size yardımcı olmak için buradayız." \
        "İlginiz için teşekkür ederiz ve gelecekteki bir rezervasyonda sizi tekrar ağırlamayı umarız." \
        "Saygılarımızla," \
        "Otel İsmi / Müşteri Hizmetleri"
    return churn_olan_genel_musteri_maili

def generate_mail(customer_info):
    if churn_musteriler[index]:
        if customer_info['no_of_children'] > 0:
            mail = cocuklu_kisilere_mail(customer_info)
        elif customer_info['no_of_adults'] == 1:
            mail = tek_basina_konaklayanlara_mail(customer_info)
        elif (customer_info['no_of_adults'] >= 3 or customer_info['no_of_adults'] <= 6):
            mail = arkadas_grubu_mail(customer_info)
        elif customer_info['no_of_adults'] == 2:
            mail = yeni_evli_ciftlere_mail(customer_info)
        elif customer_info['no_of_adults'] > 6:
            mail = is_gezisi_mail(customer_info)
        else:
            mail = genel_musteri_mail(customer_info)
    else:
        mail = daimi_musteri_mail(customer_info)

    return mail

import random

# Veri çerçevenizi temsil eden bir DataFrame'e sahip olduğunuzu varsayalım
# df adını kullanarak veri çerçevenizi değiştirebilirsiniz

# booking_status değeri 1 olan müşterileri filtreleme
filtered_df = df[df['booking_status'] == 1]

# Rastgele 10 müşteri seçme
random_10_customers = filtered_df.sample(n=10, random_state=38)

# Sonucu kontrol etme
print(random_10_customers)

for index, row in random_10_customers.iterrows():
    print(f"Müşteri {index+1} için uygun e-posta içeriği:")
    print(generate_mail(row))  # generate_mail fonksiyonunu kullanarak e-posta içeriğini oluşturun
    print("-----------------------")


