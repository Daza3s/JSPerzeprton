import { Matrix } from "./Matrix.js";

class Netzwerk {
    /**
     * Creates new network with layers specified
     * @param {Array} layers 
     */
    constructor(layers) {
        this.sättigung = 0.1;
        this.lernRate = 0.3;
        this.d = 0.002;

        this.inputCount = layers[0];
        this.outputCount = layers[layers.length-1];
        this.layerCount = layers.length;
        this.layers = layers;
        this.hiddenLayerCount = this.layerCount-2;

        this.weights = [];
        this.biases = [];
        this.actFunc = [];

        for(let i = 0;i < this.layerCount-1;i++) {
            this.weights.push(new Matrix( this.layers[i] , this.layers[i+1] ));
            this.weights[i].randomize();
            this.biases.push(new Matrix( 1 , this.layers[i+1] ));
            this.biases[i].randomize();
            this.actFunc.push([this.act, this.actA]);
        }
    }
    
    setActivations(functions) {
        if(functions.length != this.weights.length) throw "Wrong amount of functions specified";
        
        if(functions[0].length != 2) throw "Did not specify activation and derivativ for each layer";
        
        this.actFunc = functions;
    }

    
    /**
     * 
     * @param {Number} n 
     * @returns {Number} Sigmoid(n)
     */
    sigmoid = function(n) {
        return 1 / (1 + Math.exp(-n));
    }

    /**
     * 
     * @param {Matrix} m 
     * @returns {Matrix} mapped with f_act
     */
    act = function(m) {
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x));
        return erg;
    }

    /**
     * 
     * @param {Matrix} m 
     * @returns {Matrix} mapped with f'act
     */
    actA = function(m) {
        let sättigung = this.sättigung || 0.1;
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x) * (1-sigmoid(x)) + sättigung);
        return erg;
    }
    
    /**
     * 
     * @param {Matrix} output 
     * @param {Matrix} target 
     * @returns {Matrix}
     */
    loss(output, target) {
        let erg = output.sub(target);
        return erg;
    }

    /**
     * 
     * @param {Function} func 
     */
    setLoss(func) {
        this.loss = func;
    }

    /**
     * 
     * @param {Matrix} output 
     * @param {Matrix} target 
     * @returns {Number}
     */
    errorTotal(output, target) {
        let Etotal = 0;

        for(let i = 0;i < output.werte.length;i++) {
            let wert = output.werte[i];
            let ziel = target.werte[i];
            Etotal += (ziel - wert)*(ziel - wert);
        }

        Etotal = Etotal / 2;
        return Etotal;
    }

    /**
     * 
     * @param {Function} func 
     */
    setErrorFunction(func) {
        this.errorTotal = func;
    }

    /**
     * 
     * @param {Matrix} input 
     * @returns {Matrix} output
     */
    predict = function(input) {
        let erg = input;
        for(let i = 0;i < this.layerCount-1;i++) {
            erg = erg.mult(this.weights[i]);
            erg = erg.add(this.biases[i]);
            erg = this.actFunc[i][0](erg);
        }
        return erg;
    }
    
    /**
     * 
     * @param {Matrix} input 
     * @param {Matrix} target 
     * @returns {Number} error
     */
    train = function(input, target, opts = false) {
        let net = [];
        let out = [];

        net[0] = input.mult(this.weights[0]).add(this.biases[0]);
        out[0] = this.actFunc[0][0](net[0]);

        for(let i = 1;i < this.layerCount-1;i++) {
            net.push(out[i-1].mult(this.weights[i]).add(this.biases[i]));
            out.push(this.actFunc[i][0](net[i]));
        }

        let output = out[out.length-1];

        /*Fehlerberechnung*/
        let Etotal = this.errorTotal(output, target);

        /**********/
        /*
            Deltaberechnung (d.h. der schwere Part)
        */
        let oDeltas = [];
        let wDeltas = [];

        let l = this.loss(output, target);
        let a = this.actFunc[this.actFunc.length-1][1](net[net.length-1]);
        let d = l.dot(a);
        let deltaO = d;
        let deltaW = (out[out.length-2] || input).dreh().mult( deltaO ).mult(this.lernRate);

        oDeltas.unshift(deltaO);
        wDeltas.unshift(deltaW);

    if(this.hiddenLayerCount > 0) {


        for(let i = 0;i < this.weights.length-2;i++) {
            let evIndex = net.length-1-2-i;

            //nullte 0 delta da das letzte geshifted wurde
            let weightedDelta = oDeltas[0].mult( this.weights[this.weights.length-1-i].dreh() );

            deltaO = this.actFunc[evIndex+1][1](net[evIndex+1]).dot( weightedDelta );  //????????
            deltaW = out[evIndex].dreh().mult( deltaO ).mult(this.lernRate);

            oDeltas.unshift(deltaO);
            wDeltas.unshift(deltaW);
        }

    

        let w = this.weights[1].dreh();

        //nullte 0 delta da das letzte geshifted wurde
        deltaO = this.actFunc[0][1](net[0]).dot( oDeltas[0].mult(w)); //??????
        deltaW = input.dreh().mult(deltaO).mult(this.lernRate);

        oDeltas.unshift(deltaO);
        wDeltas.unshift(deltaW);
    }
        /*
            Anpassen der Gewichte 
        */
        
        /*
            Checking special options
         */
        if(opts) {
            if(opts.gradientClipping) {
                for(let i = 0;i < wDeltas.length;i++) {
                    wDeltas[i].werte = wDeltas[i].werte.map((x)=> {
                        let erg = Math.min(x, opts.gradientClipping.max);
                        erg = Math.max(erg, opts.gradientClipping.min);
                        return erg;
                    });

                    oDeltas[i].werte = oDeltas[i].werte.map((x)=> {
                        let erg = Math.min(x, opts.gradientClipping.max/this.lernRate);
                        erg = Math.max(erg, opts.gradientClipping.min/this.lernRate);
                        return erg;
                    });
                }
            }
            if(opts.batch) {
                let bDeltas = [];
                for(let i = 0;i < oDeltas.length;i++) {
                    bDeltas.push(oDeltas[i].mult(this.lernRate));
                }

                return [Etotal, wDeltas, bDeltas];
            }
        }

        let gradMax = 0;

        for(let i = 0;i < wDeltas.length;i++) {
            gradMax = wDeltas[i].werte.reduce((a,b)=>Math.max(a,b));
        }


        for(let i = 0;i < this.weights.length;i++) {
            if(!wDeltas[i].werte[0] && wDeltas[i].werte[0] !== 0) {
                
                console.log("fishy");
            }
            this.weights[i] = this.weights[i].sub(wDeltas[i]);
            this.biases[i] = this.biases[i].sub(oDeltas[i].mult(this.lernRate));
        }

        return [Etotal, Math.abs(gradMax)];
        
    }
    
    /**
     * 
     * @param {Array} inputArray 
     * @param {Array}} targetArray 
     * @param {Number} rounds 
     * @param {Number} minError 
     * @returns {Number} error
     */
    trainSet = function(inputArray, targetArray, rounds = 1000, batch = false, minError = 0.001) {
        let rIndex = 0;
        let error = 0;

        if(inputArray[0].length !== this.weights[0].iDim) {
            throw `Input (${inputArray[0].length}) to Network (${this.weights[0].iDim}) missmatch`;
        }

        if(targetArray[0].length !== this.weights[this.weights.length-1].jDim) {
            throw `Network (${this.weights[this.weights.length-1].jDim}) to Target (${targetArray[0].length}) missmatch`;
        }

        let inputMatrix = new Matrix(1, inputArray[0].length);
        let targetMatrix = new Matrix(1, targetArray[0].length);

        let gradMax = 0;

        let opts = {
            gradientClipping: {
                max: 200,
                min: -200
            },
            batch: batch
        }

        let gradients = false;

        while(rIndex < rounds) {
            
            if(batch) gradients = [];


            for(let i = 0;i < inputArray.length;i++) {
                inputMatrix.werte = inputArray[i];
                targetMatrix.werte = targetArray[i];

                let ergebnis = this.train(inputMatrix, targetMatrix, opts);
                error += ergebnis[0];


                if(batch) {
                    gradients.push([ergebnis[1], ergebnis[2]]);
                    continue;
                }
                
                
                
                gradMax = gradMax < ergebnis[1]? ergebnis[1]:gradMax;
            }
            //console.dir(gradients[0]);
            //console.log(error/(rIndex+1));
            if(batch) {
                for(let gradientIndex = 0;gradientIndex < gradients.length;gradientIndex++) {
                    for(let weightIndex = 0;weightIndex < this.weights.length;weightIndex++) {
                        this.weights[weightIndex] = this.weights[weightIndex].sub(gradients[gradientIndex][0][weightIndex]);
                        this.biases[weightIndex] = this.biases[weightIndex].sub(gradients[gradientIndex][1][weightIndex]);
                    }
                }
            }

            if(error < minError) {
                console.log(gradMax);
                return error/rIndex;
            }
            rIndex++;
        }
        //console.log(gradMax);
        return error/rounds;
    }

    safeFast() {

    }
}

export { Netzwerk }