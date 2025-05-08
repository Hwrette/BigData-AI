import tensorflow as tf

height = 170
footSize = 172

a = tf.Variable(0.1)
b = tf.Variable(0.5)


for i in range(100) :
    with tf.GradientTape() as tape: 
        guessFoot = height * a + b
        loss = (guessFoot - footSize)**2

    gradient = tape.gradient(loss, [a, b])
    a.assign_sub(gradient[0]*0.00001)
    b.assign_sub(gradient[1]*0.00001)
    print(a.numpy(), b.numpy())