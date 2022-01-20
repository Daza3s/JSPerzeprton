import { Netzwerk } from "./netzwerk/NeuronalesNetzwerk.js"
import { Matrix } from "./netzwerk/Matrix.js";
import { activations,loss } from "./netzwerk/functions.js";

let bn = new Netzwerk([2,3,3,2]);

let inps = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
];

let outs = [
    [1,0],
    [0,1],
    [0,1],
    [0,1]
];

let reluPair = [activations.relu,activations.reluA];
let softPair = [activations.softmax,activations.softmaxA];

bn.setActivations([reluPair,reluPair,softPair]);
bn.setErrorFunction(loss.crossEntropy);
bn.setLoss(loss.crossEntropyA);

bn.trainSet(inps, outs, 10000, false);

let x00 = new Matrix(1,2);
x00.init();

let x01 = new Matrix(1,2);
x01.werte[0] = 1;
x01.werte[1] = 0;

let x10 = new Matrix(1,2);
x10.werte[0] = 0;
x10.werte[1] = 1;

let x11 = new Matrix(1,2);
x11.werte[0] = 1;
x11.werte[1] = 1;

console.log("Weights");
console.dir(bn.weights, true);
console.log("Biases")
console.dir(bn.biases, true)

console.dir(bn.predict(x00).werte);
console.dir(bn.predict(x01).werte);
console.dir(bn.predict(x10).werte);
console.dir(bn.predict(x11).werte);

/*let nn = new Netzwerk([3,3,3,3]);

nn.lernRate = 0.4;

nn.setActivations([
    [activations.lrelu,activations.lreluA],
    [activations.lrelu,activations.lreluA],
    [activations.softmax,activations.softmaxA]
]);

nn.setErrorFunction(loss.crossEntropy);
nn.setLoss(loss.crossEntropyA);


console.dir(nn.weights[0].nest());
console.dir(nn.weights[1].nest());
console.dir(nn.weights[2].nest());

let og0 = nn.weights[0].werte.slice();
let og1 = nn.weights[1].werte.slice();
let og2 = nn.weights[2].werte.slice();

let inps = [];
let targets = [];


for(let i = 0;i < 10;i++) {
    let target = new Matrix(1,3);
    target.init();
    target.werte[parseInt(Math.random()*3)] = 1;
    let input = new Matrix(1,3);
    input.randomize();
    inps.push(input);
    targets.push(target)
}

let firstError = false;
let lastError = false;
let errorP = 0;
for(let i = 0;i < 500;i++) {
    errorP = 0;
    for(let j = 0;j < inps.length;j++) {
        errorP += nn.train(inps[j], targets[j]);
    }
    errorP /= inps.length;
    if(firstError === false) firstError = errorP;
    lastError = errorP;
    if(errorP === NaN) {
        console.log("fishy");
    }
}

console.dir(nn.weights[0].nest());
console.dir(nn.weights[1].nest());
console.dir(nn.weights[2].nest());

let diff0 = og0.map((x,i)=>x - nn.weights[0].werte[i]);
let diff1 = og1.map((x,i)=>x - nn.weights[1].werte[i]);
let diff2 = og2.map((x,i)=>x - nn.weights[2].werte[i]);

console.dir(diff0);
console.dir(diff1);
console.dir(diff2);

console.log("Diff: "+(firstError-errorP));

console.log(firstError);
console.log(errorP);*/

/*
let input = new Matrix(1,6);
let target = new Matrix(1,6);
let weightjk = new Matrix(6,6);


weightjk.randomize();
input.randomize();
target.init();
target.werte[0] = 1;

let netk = input.mult(weightjk);
let outk = activations.softmax(netk);
let Etotal = loss.crossEntropy(outk,target);
let dEdout = loss.crossEntropyA(outk, target);
let doutdnet = activations.softmaxA(netk);
let dnetdwjk = input;
let dEdWjk = dnetdwjk.dreh().mult( dEdout.dot(doutdnet) );

console.log(`Weight: ${weightjk.nest()}`);
console.log(`INPUT: ${input.nest()}`);
console.log(`NetK: ${netk.nest()}`);
console.log(`OutK: ${outk.nest()}`);
console.log(`Target: ${target.nest()}`);
console.log(`Sum: ${outk.sum()}`)
console.log(`Etotal: ${Etotal}`);
console.log(`dE/dOut: ${dEdout.nest()}`);
console.log(`dOut/dNet: `);
console.dir(doutdnet.nest());

console.log(`dNet/dWjk: ${dnetdwjk.nest()}`);
console.log(`dE/dWjk: ${dEdWjk.nest()}`);
*/

/*
console.time("Training");
for(let i = 0;i < 10000;i++) {
    nn.train(input, target);
    target.randomize();
    input.randomize();
}
console.timeEnd("Training");
input.randomize()
console.dir(nn.predict(input));
*/