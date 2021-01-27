# DietOptimizer
Flask app that runs on Google App Engine for diet linear programming.

## Example

POST http://dietlp.appspot.com
```
 {
        "goal": "MAX",
        "priority": "protein",
        "meals": [
            3,
            6
        ],
        "calories": [
            1000,
            2000
        ],
        "days": 7,
        "protein": [
            50,
            75
        ],
        "tastes": "ice cream | (salty && Chinese)",
        "allergens": "peanut|sesame|lobster"
 }
 ```
 
## Documentation
TODO + refactoring
