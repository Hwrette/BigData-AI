import tensorflow as tf

height = [ 150, 160, 170, 180 ]
footSize = [ 152, 162, 172, 182 ]

a = tf.Variable(0.1)
b = tf.Variable(0.5)

opt = tf.keras.optimizers.Adam(learning_rate = 0.001 )

for i in range(1000) :
    with tf.GradientTape() as tape: 
        guessFoot = height * a + b
        loss = (guessFoot - footSize)**2
        # 데이터가 리스트라 얘네도 리스트로 나와서 밑에 grident에 바로 넣으면 이상해질 수 있어서 loss값을 평균내는거래
        loss = tf.reduce_mean(loss)

    gradient = tape.gradient(loss, [a, b])
    opt.apply_gradients([[gradient[0], a],[gradient[1], b] ]) 

    print(a.numpy(), b.numpy())