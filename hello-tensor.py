import tensorflow as tf

msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

# MNIST 손글씨 데이터 package 가져오기
mnist = tf.keras.datasets.mnist

# MNIST 4분할 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)


# 이미지 데이터 원본 출력
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')
plt.show()

print('첫번째 학습용 데이터 입력값:', X_train[0])
print('첫번째 학습용 데이터 출력값:', Y_train[0])


# 이미지 데이터 [0,1] 스케일링
X_train = X_train / 255.0
X_test = X_test / 255.0

# 스케일링 후 데이터 확인
plt.imshow(X_train[0], cmap='gray')
plt.show()
print('첫번째 학습용 데이터 입력값:', X_train[0])





















