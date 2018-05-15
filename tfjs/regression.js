// Data source: http://archive.ics.uci.edu/ml/datasets/Container+Crane+Controller+Data+Set

const dat = tf.tensor2d(
    [1.0, -5.0, 0.3,
    2.0, 5.0, 0.3, 
    3.0, -2.0, 0.5, 
    1.0, 2.0, 0.5, 
    2.0, 0.0, 0.7, 
    6.0, -5.0, 0.5, 
    7.0, 5.0, 0.5, 
    6.0, -2.0, 0.3, 
    7.0, 2.0, 0.3, 
    6.0, 0.0, 0.7, 
    8.0, -5.0, 0.5, 
    9.0, 5.0, 0.5, 
    10.0, -2.0, 0.3, 
    8.0, 2.0, 0.3, 
    9.0, 0.0, 0.5],
    [15, 3]
);

// Slice 
const y = dat.slice([0, 0], [15, 1]);
const x = dat.slice([0, 1], [15, 2]);

// Make a simple model with a single hidden layer
const model = tf.sequential();
model.add(tf.layers.dense({
    units: 8,
    inputShape: [2],
    useBias: true,
    activation: "relu",
    kernelInitializer: "randomNormal"
}));
model.add(tf.layers.dense({units: 1}));

// Use gradient descent
const learningRate = 0.005;
const optimizer = tf.train.sgd(learningRate);

// Compile model
model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError"
});

// Train model
model.fit(x, y, {
    batchSize: 4,
    epochs: 100
}).then(d => console.log(d.history.loss));

// Evaluate model
// model.evaluate(x, y).print();
