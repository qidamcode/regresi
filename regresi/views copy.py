import urllib.parse
from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
import io, urllib, base64
import statsmodels.api as sm

# Create your views here.
def index(request):
    context = {
        'data': "data"
    }
    return render(request, "regresi/index.html", context)

@requires_csrf_token
def analisis(request):
    matplotlib.use('agg')

    # membaca data dari input file (data prepration)
    df = pd.read_csv(request.FILES["file"])

    # membersihkan data yang bernilai null (data cleaning)
    df.dropna()

    # memisahkan data dependent (y) dan data independent (X) (preprocessing)
    X = df[['kelahiran', 'kematian', 'perpindahan_masuk', 'perpindahan_keluar']]
    y = df['jml_penduduk']

    # memisahkan data training dan data test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    display_X_train = pd.DataFrame({
        'kelahiran': X_train['kelahiran'],
        'kematian': X_train['kematian'],
        'perpindahan_masuk': X_train['perpindahan_masuk'],
        'perpindahan_keluar': X_train['perpindahan_keluar'],
    })

    display_y_train = pd.DataFrame({
        'jml_penduduk': y_train,
    })

    display_X_test = pd.DataFrame({
        'kelahiran': X_test['kelahiran'],
        'kematian': X_test['kematian'],
        'perpindahan_masuk': X_test['perpindahan_masuk'],
        'perpindahan_keluar': X_test['perpindahan_keluar'],
    })

    display_y_test = pd.DataFrame({
        'jml_penduduk': y_test,
    })

    # melakukan regresi linier
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    # membuat prediksi dari data test
    y_test_pred = linear_reg.predict(X_test)
    y_train_pred = linear_reg.predict(X_train)

    # menyimpan data koefision dan intercept
    coefisien = linear_reg.coef_
    intercept = linear_reg.intercept_

    feature_cols = ['kelahiran','kematian','perpindahan_masuk', 'perpindahan_keluar']
    list_baru = list(zip(feature_cols, coefisien))

    # menyimpan data r squeare, mean squeared error, dan root mean squeared error
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    meanAbErr = metrics.mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    X_train = sm.add_constant(X_train)
    summary = sm.OLS(y_train, X_train).fit().summary()
    html_summary = summary.as_html()

    # data frame selisih nilai test dan nilai prediksi
    pred_df = pd.DataFrame({
        'Nilai Test' :y_test, 
        'Nilai Prediksi':y_test_pred, 
        'Selisih':y_test-y_test_pred
    })

    # membuat grafik dan mengubah ke file gambar
    sns.regplot(x=y_train, y=y_train_pred)
    # plt.scatter(y_train, y_train_pred)
    plt.xlabel("Y train")
    plt.ylabel("Y train prediksi")
    plt.plot()

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    # membuat variabel context
    context = {
        'model': linear_reg,
        'mse': mse,
        'r2': r2,
        'rmse': rmse,
        'meanAbErr': meanAbErr,
        'df': df.reset_index(drop=True),
        'display_X_train': display_X_train,
        'display_y_train': display_y_train,
        'display_X_test': display_X_test,
        'display_y_test': display_y_test,
        'x_columns': X.columns.tolist,
        'pred_df': pred_df.reset_index(drop=True),
        'fig_test': uri,
        'summary': html_summary
    }
    return render(request, "regresi/analisis.html", context)