import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt
#%matplotlib inline



def is_rain_Basel(row):
    if row['BASEL_precipitation'] > 0:
        return True
    else:
        return False

def is_rain_Dresen(row):
    if row['DRESDEN_precipitation'] > 0:
        return True
    else:
        return False

def main():
    weather = pd.read_csv('./archive/weather_prediction_dataset.csv') #read data
    
    #lay data TP Basel
    df_basel = weather.iloc[:, 0:11].copy()
    df_basel.head(5)
    df_basel['rain'] = df_basel.apply(is_rain_Basel, axis = 1)

    #lay data TP Dresen
    df_dresen = weather.loc[:, ['DATE', 'MONTH', 'DRESDEN_cloud_cover', 'DRESDEN_wind_speed', 'DRESDEN_wind_gust', 'DRESDEN_humidity', 'DRESDEN_global_radiation', 'DRESDEN_precipitation', 'DRESDEN_sunshine', 'DRESDEN_temp_mean', 'DRESDEN_temp_min', 'DRESDEN_temp_max']].copy()
    df_dresen['rain'] = df_dresen.apply(is_rain_Dresen, axis = 1)

    df_mean_temp = df_basel.groupby(['MONTH','rain'], as_index=False)['BASEL_temp_mean'].mean()
    df_max_temp = df_basel.groupby(['MONTH','rain'], as_index=False)['BASEL_temp_max'].mean()
    df_min_temp = df_basel.groupby(['MONTH','rain'], as_index=False)['BASEL_temp_min'].mean()

    
    #Chart nhiet do va ap suat muc nuoc bien
    df_mean_cloud_cover_month = df_basel.groupby(['MONTH'], as_index=False)[['BASEL_cloud_cover', 'BASEL_pressure']].mean()
    sns.lineplot(df_basel, x='MONTH', y='BASEL_temp_mean', color='g', legend='auto')
    ax2 = ax2 = plt.twinx()
    sns.lineplot(df_basel, x= 'MONTH' , y='BASEL_pressure', color='r', legend='auto', ax=ax2)
    #axs[1,1].set_title('Mean temparture through month')
    
    #chart nhiet va do che phu cua may
    df_mean_cloud_cover_month = df_basel.groupby(['MONTH'], as_index=False)['BASEL_cloud_cover'].mean()
    sns.lineplot(df_basel, x='MONTH', y='BASEL_temp_mean', color='g', legend='auto')
    ax2 = ax2 = plt.twinx()
    sns.lineplot(df_mean_cloud_cover_month.BASEL_cloud_cover, color='r', legend='auto', ax=ax2)
    #axs[1,1].set_title('Mean temparture through month')
    
    #chart nhiet do va do am
    sns.lineplot(df_basel, x='MONTH', y='BASEL_humidity', color='g', legend='auto')
    ax2 = ax2 = plt.twinx()
    sns.lineplot(df_basel.BASEL_temp_mean, color='r', legend='auto', ax=ax2)
    #axs[1,1].set_title('Mean temparture through month')
    
    #chart luong mua va ap suat muc nuoc bien 
    df_mean_sunshine_humidity = df_basel.groupby(['MONTH'], as_index=False)[['BASEL_global_radiation', 'BASEL_humidity']].mean()
    sns.lineplot(df_basel, x='MONTH', y='BASEL_pressure', color='g', legend='auto')
    ax2 = plt.twinx()
    sns.lineplot(df_basel, x = 'MONTH', y='BASEL_precipitation', color='r', legend='auto', ax= ax2)
    #axs[1,1].set_title('Mean temparture through month')

    #chart do am va luong mua
    df_mean_sunshine_humidity = df_basel.groupby(['MONTH'], as_index=False)[['BASEL_global_radiation', 'BASEL_humidity']].mean()
    sns.lineplot(df_basel, x='MONTH', y='BASEL_humidity', color='g', legend='auto')
    ax2 = plt.twinx()
    sns.lineplot(df_basel, x = 'MONTH', y='BASEL_precipitation', color='r', legend='auto', ax= ax2)
    #axs[1,1].set_title('Mean temparture through month')
    
    
    #chart ap suat muc nuoc bien va nhiet do
    df_mean_sunshine_humidity = df_basel.groupby(['MONTH', 'rain'], as_index=False)[['BASEL_global_radiation', 'BASEL_temp_mean']].mean()
    sns.lineplot(df_basel, x='MONTH', y='BASEL_temp_mean', color='g', legend='auto')
    ax2 = plt.twinx()
    sns.lineplot(df_basel, x = 'MONTH', y='BASEL_pressure', color='r', legend='auto', ax= ax2)
    #axs[1,1].set_title('Mean temparture through month')
    
    
    #chart nhiet do va toc do gio
    sns.lineplot(df_dresen, x='MONTH', y='DRESDEN_temp_mean', color='g', legend='auto')
    ax2 = plt.twinx()
    sns.lineplot(df_dresen, x = 'MONTH', y='DRESDEN_wind_speed', color='r', legend='auto', ax= ax2)
    
    # Model: Decision Tree
    # mo hinh 1 tinh do chinh xac va chart
    X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    clf_m1 = tree.DecisionTreeClassifier()
    clf_m1 = clf_m1.fit(X_train_m1, y_train_m1)
    print(clf_m1.score(X_test_m1, y_test_m1))
    
    y_pred_m1 = clf_m1.predict(X_test_m1)
    print(f1_score(y_test_m1, y_pred_m1))
    cm_m1 = confusion_matrix(y_test_m1, y_pred_m1, labels=clf_m1.classes_)
    disp_m1 = ConfusionMatrixDisplay(confusion_matrix=cm_m1,display_labels=clf_m1.classes_)
    disp_m1.plot()
    plt.show()
    
   #mo hinh 2
    
    X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean', 'DRESDEN_wind_speed']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    clf_m2 = tree.DecisionTreeClassifier()
    clf_m2 = clf_m2.fit(X_train_m2, y_train_m2)
    print(clf_m2.score(X_test_m2, y_test_m2))
    
    y_pred_m2 = clf_m2.predict(X_test_m2)
    print(f1_score(y_test_m2, y_pred_m2))
    cm_m2 = confusion_matrix(y_test_m2, y_pred_m2, labels=clf_m2.classes_)
    disp_m2 = ConfusionMatrixDisplay(confusion_matrix=cm_m2,display_labels=clf_m2.classes_)
    disp_m2.plot()
    plt.show()
    

    #mo hinh 3
    
    m3_X_train, m3_X_test, m3_y_train, m3_y_test = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean', 'DRESDEN_wind_speed', 'DRESDEN_wind_gust', 'DRESDEN_humidity', 'DRESDEN_global_radiation', 'DRESDEN_precipitation', 'DRESDEN_sunshine', 'DRESDEN_temp_mean', 'DRESDEN_temp_min', 'DRESDEN_temp_max']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    clf_m3 = tree.DecisionTreeClassifier()
    clf_m3 = clf_m3.fit(m3_X_train, m3_y_train)
    print(clf_m3.score(m3_X_test, m3_y_test))

    m3_y_pred = clf_m3.predict(m3_X_test)
    print(f1_score(m3_y_test, m3_y_pred))
    cm_m3 = confusion_matrix(m3_y_test, m3_y_pred, labels=clf_m3.classes_)
    disp_m3 = ConfusionMatrixDisplay(confusion_matrix=cm_m3,display_labels=clf_m3.classes_)
    disp_m3.plot()
    plt.show()
    
    # Model: Logistic Regression
    #mo hinh 4
    m4_X_train, m4_X_test, m4_y_train, m4_y_test = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean', 'DRESDEN_wind_speed']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    m4_logistic_model = LogisticRegression(random_state=0).fit(m4_X_train, m4_y_train)
    print(m4_logistic_model.score(m4_X_test, m4_y_test))
    m4_y_pred = m4_logistic_model.predict(m4_X_test)
    print(f1_score(m4_y_test, m4_y_pred))
    cm_m4 = confusion_matrix(m4_y_test, m4_y_pred, labels=m4_logistic_model.classes_)
    disp_m4 = ConfusionMatrixDisplay(confusion_matrix=cm_m4,display_labels=m4_logistic_model.classes_)
    disp_m4.plot()
    plt.show()
    
    
    
    #mo hinh 5
   
    m5_x1_train, m5_x1_test, m5_y1_train, m5_y1_test = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean', 'DRESDEN_wind_speed']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    m5_logistic_model = LogisticRegression(random_state=0).fit(m5_x1_train, m5_y1_train)
    print(m5_logistic_model.score(m5_x1_test, m5_y1_test))

    m5_y_pred = m5_logistic_model.predict(m5_x1_test)
    print(f1_score(m5_y1_test, m5_y_pred))
    cm_m5 = confusion_matrix(m5_y1_test, m5_y_pred, labels=m5_logistic_model.classes_)
    disp_m5 = ConfusionMatrixDisplay(confusion_matrix=cm_m5,display_labels=m5_logistic_model.classes_)
    disp_m5.plot()
    plt.show()



    #mo hinh 6
    
    m6_x1_train, m6_x1_test, m6_y1_train, m6_y1_test = train_test_split(df_dresen[['DRESDEN_cloud_cover', 'DRESDEN_temp_mean', 'DRESDEN_wind_speed']], df_dresen['rain'], test_size = 0.3, train_size = 0.7) 
    m6_logistic_model = LogisticRegression(random_state=0).fit(m6_x1_train, m6_y1_train)

    print(m6_logistic_model.score(m6_x1_test, m6_y1_test))

    m6_y_pred = m6_logistic_model.predict(m6_x1_test)
    print(f1_score(m6_y1_test, m6_y_pred))
    cm_m6 = confusion_matrix(m6_y1_test, m6_y_pred, labels=m6_logistic_model.classes_)
    disp_m6 = ConfusionMatrixDisplay(confusion_matrix=cm_m6,display_labels=m6_logistic_model.classes_)
    disp_m6.plot()
    plt.show()
    
if __name__ == '__main__':
    main()


