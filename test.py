from edward.models import Normal

x = Normal(0.0, 1.0)

sess = ed.get_session()
sess.run(x)
## 0.088767856
x.eval()
## 1.0179992

x_ph = tf.placeholder(tf.float32, [])
y = Normal(x_ph, 1.0)

sess.run(y, feed_dict={x_ph: 100.0})
## 100.72381
y.eval(feed_dict={x_ph: 100.0})
## 101.87513
