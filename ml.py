import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

def generate_model(model):
    # モデルの保存先ディレクトリを指定する
    model_dir = 'model/'

    # ディレクトリが存在しない場合は作成する
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # モデルを保存するファイルパスを指定する
    model_file = os.path.join(model_dir, 'model.pkl')

    # モデルを保存する
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

# データの読み込み
X = []
y = []

for i in range(1, 11):
    for j in range(1, 6):
        img = cv2.imread(f"data/image_{i}_{j}.jpg")
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img.flatten())
        y.append(i)

X = np.array(X)
y = np.array(y)

# データの前処理
X = X / 255.0

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# SVMモデルの学習
model = SVC(kernel='rbf', C=10, gamma=0.1)
model.fit(X_train, y_train)

generate_model(model)

# テストデータに対する予測
y_pred = model.predict(X_test)

# モデルの評価
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
