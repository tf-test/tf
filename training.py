import tensorflow as tf
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(2.0, name="bias")
feed_dict = {w1:4,w2:8}
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print(sess.run(w4,feed_dict))
saver.save(sess, "/output/abc.ckpt")
