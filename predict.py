import os
import pickle

# 保存したモデルを読み込む
def load_model(model_dir):
    model_file = os.path.join(model_dir, 'model.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

# 新しい画像データに対して予測を行う関数を定義する
def predict(model_dir, X):
    # モデルを読み込む
    model = load_model(model_dir)

    # 予測を行う
    y_pred = model.predict([X])

    return y_pred


# 保存したモデルがあるディレクトリのパスを指定する
model_dir = 'model/'

# 新しい画像データを用意する
X_new = ...

# 予測を行う
y_pred = predict(model_dir, X_new)
