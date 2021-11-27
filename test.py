from scipy.stats import norm
import traceback
try:
  print(norm.logpdf(0.0))
  print(norm.logpdf([0.0]))
  print(norm.logpdf([[0.0]]))
  print(norm.logpdf(0.0, loc=5))
  print(norm.logpdf(0.0, loc=[5]))
  ## -0.918938533205
  ## [-0.91893853]
  ## [[-0.91893853]]
  ## -13.4189385332
  ## [-13.41893853]

  from edward.stats import norm
  import tensorflow as tf
  sess = tf.InteractiveSession()

  print(norm.logpdf(0.0).eval())
  print(norm.logpdf([0.0]).eval())
  print(norm.logpdf([[0.0]]).eval())
  assert(norm.logpdf([[0.0]]).eval().shape == (1,1))
  print(norm.logpdf(0.0, loc=5).eval())
  print(norm.logpdf(0.0, loc=[5]).eval())
  ## edward with this pull request:
  ## -0.918939
  ## [-0.91893852]
  ## [[-0.91893852]]
  ## -13.4189
  ## [-13.41893864]
  ##
  ## edward without this pull request:
  ## -0.918939
  ## -0.918939
  ## -0.918939
  ## -13.4189
  ## -13.4189
except Exception as e:
  traceback.print_exc()
