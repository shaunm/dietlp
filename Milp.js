let solver = require("javascript-lp-solver");
let data = require("./foods.json");

class DietOptimizer {
    goals;
    solutions = [];
    days;
    solved = false;
    constructor(options, days) {
        this.goals = options;
        this.days = days;
    }
    solveDay = () => {

        let foods = {};
        let intVars = {};
        let servingConstr = {}
        const shuffled = Object.keys(data).sort(() => 0.5 - Math.random());
        // Get sub-array of first n elements after shuffled then propagate empty objects
        shuffled.slice(0, 100).forEach((item) => {
            foods[item] = data[item];
            intVars[item] = 1;
            servingConstr[item] = { "max": 1.1 }; //no "priority"
        });

        let model = {

            "opType": "min",
            "optimize": "quantity",
            "constraints": {
                ...this.goals,
                //"quantity": { "max": 5, "min": 3 }
                ...servingConstr
            },
            "variables": foods,
            "ints": intVars,
            "options": {
                "tolerance": 0.05,
                "keep_solutions": true
            }

        };
        let ans = solver.Solve(model, 1e-8, true);
        //console.log(model)
        let result = ans.solutionSet;
        //console.log(ans._tableau.model.solutions);

        let satisfied = true;

        for (const key in result) {
            if (result[key] > 1) {
                console.log("The solution is not binary, let's try again")
                satisfied = false;
                break;
            }
        }

        if (!satisfied) {
            this.solveDiet() //let's try again
        }
        else {
            this.solutions.push(result);
            if (this.solutions.length >= this.days){
                this.solved = true;
            }
            console.log(result);
        }

    }

    gatherDays = () => {
        let n = this.days;
        while (n > 0){
            this.solveDay();
            n--;
        }
    }

}

opts = {
    "calories": { "max": 2000, "min": 1500 },
    "carbohydrates": { "min": 270, "max": 370 },
    "protein": { "min": 70, "max": 90 },
    "fat": { "min": 50, "max": 70 },
}
plan = new DietOptimizer(opts)
plan.gatherDays();
console.log(plan.solutions)
