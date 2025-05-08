import tensorflow as tf

tensor = tf.constant( [3,4,5] )
tensor2 = tf.constant( [6,7,8] )
tensor3 = tf.constant( [ [1,2],
                         [3,4]])
tensor4 = tf.constant( [ [5,6],
                         [7,8]])
print(tensor + tensor2)
print(tf.matmul(tensor3, tensor4))

tensor5 = tf.zeros( [2,2,3] )
print(tensor5)

tensor6 = tf.constant( [3.0,4,5], tf.float32 )

print(tensor3.shape)
print(tensor6.shape)

# tf.cast

weight = tf.Variable(1.0)
weight.assign(2)
print(weight.numpy())
