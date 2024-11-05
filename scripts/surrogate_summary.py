import numpy as np
import tensorflow as tf

class DeepUQSurrogate:
    def __init__(self, D, L, d):
        self.D = D  # Input dimensionality
        self.L = L  # Number of layers in encoding
        self.d = d  # Encoding layer size
        
        # Define the layer sizes with exponential decay
        rho = np.exp((np.log(d) - np.log(D)) / L)
        self.ds = [int(D * (rho ** l)) for l in range(L + 1)]
        
        # Initialize weights, biases, and graph
        self.define_params()
        self.define_graph()
        self.define_session()
        
        # Call summary to print model structure
        self.summary()

    def define_params(self):
        # Initialize weight and bias matrices for each layer
        self.Ws = []
        self.bs = []
        
        for l in range(self.L):
            W = tf.Variable(
                tf.random.normal([self.ds[l], self.ds[l + 1]]) * np.sqrt(2.0 / self.ds[l]),
                name=f"W_{l}"
            )
            b = tf.Variable(
                tf.random.normal([self.ds[l + 1]]) * np.sqrt(2.0 / self.ds[l]),
                name=f"b_{l}"
            )
            self.Ws.append(W)
            self.bs.append(b)

    def define_graph(self):
        # Define input and output placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.D], name="x")
        self.ytrue = tf.placeholder(tf.float32, shape=[None, 1], name="ytrue")
        
        # Build the network layers
        h = self.x
        for l in range(self.L):
            h = tf.nn.relu(tf.matmul(h, self.Ws[l]) + self.bs[l])
        self.y = tf.matmul(h, self.Ws[-1]) + self.bs[-1]
        
        # Define loss with MSE and regularization
        self.mse = tf.reduce_mean(tf.square(self.y - self.ytrue))
        self.l1_reg = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in self.Ws])
        self.l2_reg = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.Ws])
        self.loss = self.mse + 0.001 * self.l1_reg + 0.001 * self.l2_reg
        
        # Define optimizer
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def define_session(self):
        # Start TensorFlow session and initialize variables
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def summary(self):
        # Custom summary function to visualize model structure
        print("Model Summary:")
        print("=" * 50)
        print(f"Input Shape: ({self.D}, )")
        
        # Print each layer's shape
        for i, (input_dim, output_dim) in enumerate(zip(self.ds[:-1], self.ds[1:])):
            print(f"Layer {i + 1}:")
            print(f"  Input Dim: {input_dim}")
            print(f"  Output Dim: {output_dim}")
        
        print("=" * 50)
        print(f"Output Shape: ({self.ds[-1]}, 1)")
        print("Total Layers:", self.L)
        print("=" * 50)

    def predict(self, x):
        # Predict function for inference
        return self.sess.run(self.y, feed_dict={self.x: x})
