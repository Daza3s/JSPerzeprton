class Matrix {
    /**
     * 
     * @param {int} i_Dimension  
     * @param {int} j_Dimension 
     */
    constructor(iDim, jDim) {
        this.iDim = iDim;
        this.jDim = jDim;
        this.length = iDim*jDim;
        this.werte = new Array(this.length);
        this.j = false;
    }
    /**
     * Randomizes all values in range from -1 to 1
     */
    randomize() {
        for(let i = 0;i < this.length;i++) {
            this.werte[i] = Math.random()*2-1;
        }
    }

    init() {
        for(let i = 0;i < this.length;i++) {
            this.werte[i] = 0;
        }
    }

    /**
     * 
     * @param {int} i 
     * @param {int} j 
     * @returns {int} Wert
     */
    at(i, j) {
        return this.werte[i * this.jDim + j];
    }

    /**
     * 
     * @param {int} i 
     * @param {int} j 
     * @param {double} wert
     */
     set(i, j, wert) {
        this.werte[i * this.jDim + j] = wert;
    }

    dreh() {
        let erg = new Matrix(this.jDim, this.iDim);
        //Forwärts über j
        for(let j = 0;j < this.jDim;j++) {
            //Rückwärts über i
            for(let i = this.iDim-1;i >= 0;i--) {
                erg.set(j, this.iDim-1-i, this.at(i,j));
            }
        }
        return erg;
    }

    /**
     * 
     * @param {Matrix} m
     * @return {Matrix} Punktprodukt 
     */
    dot(m) {
        if(m.j) return this.mult(m);
        if(m.iDim !== this.iDim || m.jDim !== this.jDim) {
            throw "Invalid multiplication scale";
        }
        let erg = new Matrix(this.iDim,this.jDim);
        for(let index = 0;index < this.length;index++) {
            erg.werte[index] = this.werte[index] * m.werte[index];
        }

        return erg;
    }

    /**
     * 
     * @param {Matrix} m
     * @return {Matrix} Summe
     */
    add(m) {
        if(m.iDim !== this.iDim || m.jDim !== this.jDim) {
            throw "Invalid addition scale";
        }
        let erg = new Matrix(this.iDim,this.jDim);
        for(let index = 0;index < this.length;index++) {
            erg.werte[index] = this.werte[index] + m.werte[index];
        }

        return erg;
    }

    /**
     * 
     * @param {Matrix} m
     * @return {Matrix} Differenz
     */
    sub(m) {
        if(m.iDim !== this.iDim || m.jDim !== this.jDim) {
            throw "Invalid subtraction scale";
        }
        let erg = new Matrix(this.iDim,this.jDim);
        for(let index = 0;index < this.length;index++) {
            erg.werte[index] = this.werte[index] - m.werte[index];
        }

        return erg;
    }
    
    /**
     * 
     * @param {Matrix} m
     * @return {Matrix} Produkt
     */
    mult(m) {
        if(typeof m === "number") {
            let erg = new Matrix(this.iDim,this.jDim);
            for(let index = 0;index < this.length;index++) {
                erg.werte[index] = this.werte[index] * m;
            }
            return erg;
        }else if(this.jDim !== m.iDim) {
            throw "Invalid multiplication scale";
        }
        
        let erg = new Matrix(this.iDim, m.jDim);
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < m.jDim;j++) {
                erg.set(i,j,0);
                for(let sumIndex = 0;sumIndex < this.jDim;sumIndex++) {
                    erg.set(i,j,erg.at(i,j) + this.at(i,sumIndex) * m.at(sumIndex,j));
                }
            }
        }
        return erg;
    }

    /**
     * 
     * @returns Nested array
     */
    nest() {
        let erg = [];
        for(let i = 0;i < this.iDim;i++) {
            erg.push([]);
            for(let j = 0;j < this.jDim;j++) {
                erg[i].push(this.at(i,j));
            }
        }
        return erg;
    }

    sum(func = n=>n) {
        let erg = 0;
        for(let i = 0;i < this.werte.length;i++) {
            erg += func(this.werte[i]);
        }
        return erg;
    }
}

export { Matrix }