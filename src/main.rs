extern crate rand;
use rand::Rng;


const HIDDENSIZE: usize = 128;
const OUTPUTSIZE: usize = 10;
const INPUTSIZE: usize = 784; // 28*28 for MNIST sign recognition dataset

struct Layer {
    weights: Vec<f64>,
    biases: Vec<f64>,
    inputsize: usize,
    outputsize: usize,
}

struct Network {
    hidden: Layer,
    output: Layer,
}

// Softmax activation function
fn softmax(input: &mut Vec<f64>) {
    let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;

    for i in 0..input.len() {
        input[i] = (input[i] - max).exp();
        sum += input[i];
    }

    for i in 0..input.len() {
        input[i] /= sum;
    }
}

// Initialize layer with random weights and biases
fn initialize_layer(layer: &mut Layer, inputsize: usize, outputsize: usize) {
    let n = inputsize * outputsize;
    let scale = (2.0 / inputsize as f64).sqrt();
    layer.inputsize = inputsize;
    layer.outputsize = outputsize;
    layer.weights = vec![0.0; n];
    layer.biases = vec![0.0; outputsize];

    let mut rng = rand::thread_rng();
    for i in 0..n {
        layer.weights[i] = (rng.gen::<f64>() - 0.5) * 2.0 * scale;
    }

    for i in 0..outputsize {
        layer.biases[i] = (rng.gen::<f64>() - 0.5) * 2.0 * scale;
    }
}

// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Derivative of sigmoid function
fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

// Forward propagation
fn forward(layer: &Layer, input: &Vec<f64>, output: &mut Vec<f64>) {
    for i in 0..layer.outputsize {
        output[i] = layer.biases[i];
        for j in 0..layer.inputsize {
            output[i] += input[j] * layer.weights[j * layer.outputsize + i];
        }
    }
}

// Backward propagation
fn backward(
    layer: &mut Layer,
    input: &Vec<f64>,
    outputgrad: &Vec<f64>,
    inputgrad: &mut Vec<f64>,
    learningrate: f64,
) {
    for i in 0..layer.outputsize {
        for j in 0..layer.inputsize {
            let idx = j * layer.outputsize + i;
            let grad = outputgrad[i] * input[j];
            layer.weights[idx] -= learningrate * grad;
            if inputgrad.len() > 0 {
                inputgrad[j] += outputgrad[i] * layer.weights[idx];
            }
        }
        layer.biases[i] -= learningrate * outputgrad[i];
    }
}

// Cross-entropy loss function
fn cross_entropy_loss(predicted: &mut Vec<f64>, label: usize) -> f64 {
    -predicted[label].ln()
}

// Training function
fn train(net: &mut Network, input: &Vec<f64>, label: usize, learningrate: f64) {
    let mut hiddenoutput = vec![0.0; HIDDENSIZE];
    let mut finaloutput = vec![0.0; OUTPUTSIZE];
    let mut outputgrad = vec![0.0; OUTPUTSIZE];
    let mut hiddengrad = vec![0.0; HIDDENSIZE];

    forward(&net.hidden, input, &mut hiddenoutput);
    for i in 0..HIDDENSIZE {
        hiddenoutput[i] = sigmoid(hiddenoutput[i]);
    }

    forward(&net.output, &hiddenoutput, &mut finaloutput);
    softmax(&mut finaloutput);

    let loss = cross_entropy_loss(&mut finaloutput, label);
        println!("Loss: {}", loss); // Print or log the loss value
    for i in 0..OUTPUTSIZE {
        let flag = if i == label { 1.0 } else { 0.0 };
        outputgrad[i] = finaloutput[i] - flag;
    }

    backward(
        &mut net.output,
        &hiddenoutput,
        &outputgrad,
        &mut hiddengrad,
        learningrate,
    );

    for i in 0..HIDDENSIZE {
        if hiddenoutput[i] <= 0.0 {
            hiddengrad[i] *= sigmoid_derivative(hiddenoutput[i]);
        }
    }

    backward(&mut net.hidden, input, &hiddengrad, &mut vec![], learningrate);
}

// Prediction function
fn predict(net: &mut Network, input: &Vec<f64>) -> usize {
    let mut hiddenoutput = vec![0.0; HIDDENSIZE];
    let mut finaloutput = vec![0.0; OUTPUTSIZE];

    forward(&net.hidden, input, &mut hiddenoutput);
    for i in 0..HIDDENSIZE {
        hiddenoutput[i] = sigmoid(hiddenoutput[i]);
    }
    forward(&net.output, &hiddenoutput, &mut finaloutput);
    softmax(&mut finaloutput);

    let mut maxindex = 0;
    for i in 1..OUTPUTSIZE {
        if finaloutput[i] > finaloutput[maxindex] {
            maxindex = i;
        }
    }
    maxindex
}

fn main() {
    let input = vec![0.0; INPUTSIZE]; // Example input of 28x28 image (flattened)
    let label = 3; // Example label
    let mut net = Network {
        hidden: Layer {
            weights: vec![],
            biases: vec![],
            inputsize: INPUTSIZE,
            outputsize: HIDDENSIZE,
        },
        output: Layer {
            weights: vec![],
            biases: vec![],
            inputsize: HIDDENSIZE,
            outputsize: OUTPUTSIZE,
        },
    };

    initialize_layer(&mut net.hidden, INPUTSIZE, HIDDENSIZE);
    initialize_layer(&mut net.output, HIDDENSIZE, OUTPUTSIZE);

    let learningrate = 0.4;

    for _epoch in 0..20 {
        train(&mut net, &input, label, learningrate);
        
        //println!("Epoch {}: First weight in output layer: {}", epoch, net.output.weights[0]);
    }

    let predicted_label = predict(&mut net, &input);
    println!("Predicted Label: {}", predicted_label);
}

