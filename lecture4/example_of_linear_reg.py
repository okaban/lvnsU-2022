import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

#engine_sizeで単回帰
def linear_reg_for_engine_size(X_train, X_test, y_train, y_test, save_path):

    # 直線の係数を導出
    LR = LinearRegression()
    LR.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    beta, beta0 = LR.coef_[0][0], LR.intercept_[0]

    # 直線を可視化
    xrange = np.linspace(np.min(X_train), np.max(X_train), 100)
    y_tmp = beta0 + beta * xrange
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x=X_train, y=y_train, color='blue')
    ax.plot(xrange, y_tmp, color='green')
    ax.set_xlabel('engine-size')
    ax.set_ylabel('Price')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'linear_reg_for_train_data.png'))

    # priceの推定
    y_pred = beta0 + beta * X_test

    # 推定値と実際の値の比較
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x=y_test, y=y_pred, color='red')
    xrange = np.linspace(np.min(y_pred.tolist() + y_test.tolist()), np.max(y_pred.tolist() + y_test.tolist()), 100)
    ax.plot(xrange, xrange, color='green')
    ax.set_xlabel('正解のPrice', fontsize=20)
    ax.set_ylabel('推定のPrice', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'compare_true_pred.png'))

#engine_sizeとwidthで重回帰
def linear_reg_for_engine_size_width(X_train, X_test, y_train, y_test, save_path):

    # 直線の係数を導出
    LR = LinearRegression()
    LR.fit(X_train, y_train.reshape(-1, 1))
    beta, beta0 = LR.coef_[0], LR.intercept_[0]

    # 直線を可視化
    tmp_x = X_train[:,0].tolist() + X_test[:,0].tolist()
    p = np.linspace(np.min(tmp_x), np.max(tmp_x), 10)
    tmp_x = X_train[:,1].tolist() + X_test[:,1].tolist()
    q = np.linspace(np.min(tmp_x), np.max(tmp_x), 10)
    xrange = np.meshgrid(p,q)
    y_tmp = beta0
    for i in range(len(beta)):
        y_tmp += beta[i] * xrange[i]
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('engine-size', size = 14)
    ax.set_ylabel('width', size = 14)
    ax.set_zlabel("price", size = 14)

    ax.scatter3D(X_train[:,0], X_train[:,1], y_train, color='b', label='学習用データ')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ex_multiple_linear_reg_scatter.png'))

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('engine-size', size = 14)
    ax.set_ylabel('width', size = 14)
    ax.set_zlabel("price", size = 14)

    ax.plot_wireframe(xrange[0], xrange[1], y_tmp, alpha=0.5, color='g')
    ax.scatter3D(X_test[:,0], X_test[:,1], y_test, color='r', label='検証用データ')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ex_multiple_linear_reg_scatter_with_surface.png'))

    # 推定値と実際の値の比較
    y_pred = beta0
    for i in range(len(beta)):
        y_pred += beta[i] * X_test[:,i]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x=y_test, y=y_pred, color='red')
    xrange = np.linspace(np.min(y_pred.tolist() + y_test.tolist()), np.max(y_pred.tolist() + y_test.tolist()), 100)
    ax.plot(xrange, xrange, color='green')
    ax.set_xlabel('正解のPrice', fontsize=20)
    ax.set_ylabel('推定のPrice', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'compare_true_pred_multiple_linear_reg.png'))

def main():
    #データの準備
    cwd = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cwd, 'data', 'strong_cor.csv')
    save_path = os.path.join(cwd, 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    df = pd.read_csv(file_path)

    #===============================
    # 単回帰モデル
    #===============================
    X, y = df['engine-size'].values, df['price'].values

    # 学習データ:検証用データを8:2に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 学習用データを散布図にして確認
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x=X_train, y=y_train, color='blue')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'visualization_train_data.png'))

    linear_reg = 'no' # 学習用データの散布図が確認できたら'yes'にする
    if linear_reg == 'yes':
        linear_reg_for_engine_size(X_train, X_test, y_train, y_test, save_path)

    #===============================
    # 重回帰モデル
    #===============================
    # データを可視化
    sns.pairplot(df[['engine-size', 'width', 'price']])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ex_pair_plot.png'))

    X, y=df[['engine-size', 'width']], df['price']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)
    linear_reg_for_engine_size_width(X_train, X_test, y_train, y_test, save_path)


if __name__ == '__main__':
    main()