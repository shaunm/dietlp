import solver from "javascript-lp-solver";
import data from "./foods.json";

let foods = {};
let clonedFoods;

class LP {

    constructor() {
    }

    process() {

        this.result();
    }

    result() {

        const shuffled = Object.keys(data).sort(() => 0.5 - Math.random());

        // Get sub-array of first n elements after shuffled
        let selected = shuffled.slice(0, 100).forEach(item => foods[item] = data[item]);


        clonedFoods = JSON.parse(JSON.stringify(foods));

        console.log(foods)
        
        let intVars = Object.keys(clonedFoods).forEach(v => clonedFoods[v] = 1);
        let model = {
            "opType": "min",
            "optimize": "quantity",
            "constraints": {
                "calories": { "max": 2200, "min": 1800 },
                "quantity": { "max": 5, "min": 3 }
            },
            "variables": foods,
            "ints": clonedFoods,
            "options": {
                "tolerance": 0.05
            }
        }
        console.log(model)
        let ans = solver.Solve(model)

        console.log(ans);
    }


}

export default LP;