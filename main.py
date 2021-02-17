import json

from mip import *
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
import os
from flask import Flask
from flask import request
from flask_cors import CORS
# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)
cors = CORS(app)

app.config['PROPAGATE_EXCEPTIONS'] = False

class Recommender:
    def __init__(self):
        # Reading recipes file. Then select attributes and generate the dataframe for calculation
        recipes = pd.read_csv('recommender.csv')
        recipes = recipes[['recipe_id', 'recipe_name', 'servings',
                           'group_name', 'calories', 'protein',
                           'carbohydrates', 'fat', 'ingredients', 'recipe_source']]

        pattern = re.compile(".+?'(.+?)'")
        ingredients = np.zeros(recipes.shape[0])
        ingredients = pd.Series(ingredients)
        for i in range(recipes.shape[0]):
            temp = pattern.findall(recipes['ingredients'][i])
            temp = '|'.join(temp)
            ingredients[i] = temp
        recipes['ingredients'] = ingredients

        # Break up the big string into a string array
        feature = recipes['group_name'].str.split('|') + recipes['ingredients'].str.split('|')

        # Convert to string value
        feature = feature.fillna("").astype('str')

        # Build a 1-dimensional array with recipe names
        self.recipes = recipes
        self.recipe_name = recipes['recipe_name']
        self.recipe_source = recipes['recipe_source']
        self.indices = pd.Series(recipes.index, index=recipes['recipe_name'])
        self.feature = feature

    def feature_recommendations(self, like, allergy, num_sample, num_recipe):
        # User's favourite feature
        recipe_id = range(len(self.feature))
        recipe_length = len(self.feature)
        sample_id = random.sample(recipe_id, num_sample)
        sample = self.feature.iloc[sample_id]
        sample_length = len(sample)
        user_like_feature = sample
        temps = like.split('|')

        real_length = len(self.feature)
        for i in range(len(temps)):
            user_like = pd.Series(str(temps[i].split('&&')))
            user_like_feature[recipe_length + i] = user_like[0]
            real_length = real_length + 1

        # Convert to string value
        user_like_feature_str = user_like_feature.fillna("").astype('str')
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        user_like_tfidf_matrix = tf.fit_transform(user_like_feature_str)
        user_like_cosine_sim = linear_kernel(user_like_tfidf_matrix, user_like_tfidf_matrix)

        # The index of user_like vector
        union = []
        for i in range(len(temps)):
            idx = len(user_like_feature) - (i + 1)
            sim_scores = list(enumerate(user_like_cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:(1 + num_recipe)]
            recipe_indices = [user_like_feature.index[i[0]] for i in sim_scores]
            recipe_indices = list(filter(lambda x: x <= recipe_length, recipe_indices))

            # Remove recieps that the user is allergic to
            user_allergy = allergy.split('|')
            removed_idx = []
            for i in recipe_indices:
                for j in range(len(user_allergy)):
                    if user_allergy[j] in self.recipes['ingredients'][i] and user_allergy[j] != '':
                        print('The user is allergic to this recipe ' + self.recipe_name.iloc[i])
                        removed_idx.append(i)
                        break
            for i in range(len(removed_idx)):
                if removed_idx[i] in recipe_indices:
                    recipe_indices.remove(removed_idx[i])
            union.append(recipe_indices)
        return union


class IPSolver():
    def __init__(self, options, df):
        # We should only do this once
        self.food_items = list(df['recipe_id'])
        self.cals = dict(zip(self.food_items, df['calories']))
        self.protein = dict(zip(self.food_items, df['protein']))
        self.carbs = dict(zip(self.food_items, df['carbohydrates']))
        self.fats = dict(zip(self.food_items, df['fat']))
        self.options = options
        self.build()

    def build(self):
        # Initialize the model
        if (self.options["goal"] == "MAX"):
            self.m = Model(sense=MAXIMIZE, solver_name=CBC)
        else:
            self.m = Model(sense=MINIMIZE, solver_name=CBC)
        self.m.verbose = 1
        self.m.emphasis = 0

        # The solver uses a branching algorithm, so order matters. Let's shuffle it to get new combinations
        food_vars = [self.m.add_var(name="_" + str(food), var_type=INTEGER, lb=0, ub=1) for food in self.food_items]

        # We always want to make this our main constraint. Having less than 3 meals or more than 5 would be a worse outcome then a few calories more or less
        meals_arr = []
        cals_arr = []
        carbs_arr = []
        protein_arr = []
        fats_arr = []
        # Lets set up our constraint variables in one loop instead of 6 lol
        for i, key in enumerate(self.food_items):
            meals_arr.append(food_vars[i])
            cals_arr.append(int(1 * self.cals[key]) * food_vars[i])
            carbs_arr.append(int(1 * self.carbs[key]) * food_vars[i])
            protein_arr.append(int(1 * self.protein[key]) * food_vars[i])
            fats_arr.append(int(1 * self.fats[key]) * food_vars[i])

        self.m += xsum(meals_arr) >= self.options["meals"][0], "SumMinimum"
        self.m += xsum(meals_arr) <= self.options["meals"][1], "SumMaximum"

        self.m += xsum(cals_arr) >= self.options["calories"][
            0], "CalorieMinimum"
        self.m += xsum(cals_arr) <= self.options["calories"][
            1], "CalorieMaximum"

        objective = self.options["priority"]

        if objective == "calories":
            print("Objective: " + objective)
            self.m.objective = xsum(cals_arr)
        elif objective == "protein" and "protein" in self.options:
            print("Objective: " + objective)
            self.m.objective = xsum(protein_arr)
        elif objective == "carbohydrates" and "carbohydrates" in self.options:
            print("Objective: " + objective)
            self.m.objective = xsum(carbs_arr)
        elif objective == "fats" and "fats" in self.options:
            print("Objective: " + objective)
            self.m.objective = xsum(fats_arr)
        else:
            print("Objective: total meals")
            self.m.objective = xsum(meals_arr)

        ## Optional settings
        if "carbohydrates" in self.options:
            self.m += xsum(carbs_arr) >= self.options["carbohydrates"][
                0], "CarbsMinimum"
            self.m += xsum(carbs_arr) <= self.options["carbohydrates"][
                0], "carbohydratesMaximum"

        if "protein" in self.options:
            self.m += xsum(protein_arr) >= \
                      self.options["protein"][
                          0], "ProteinMinimum"
            self.m += xsum(protein_arr) <= \
                      self.options["protein"][
                          0], "ProteinMaximum"

        if "fats" in self.options:
            self.m += xsum(fats_arr) >= self.options["fats"][
                0], "FatsMinimum"
            self.m += xsum(fats_arr) <= self.options["fats"][
                0], "FatsMaximum"

    def solve(self):
        status = self.m.optimize(max_seconds=10, relax=False)
        solutions = []
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            for i in range(self.m.num_solutions):
                solution = []
                for v in self.m.vars:
                    if abs(v.xi(i)) > 1e-6:  # only printing non-zeros
                        solution.append(v.name)
                solutions.append(solution)
        return solutions


class DietOptimizer:
    # The recipe_id here is equal to the index
    pool = []

    def __init__(self, content, foods_csv="modified.csv"):
        union = Recommender().feature_recommendations(content["tastes"],content["allergens"], 2500, 1000)
        flat_union = [item for sublist in union for item in sublist]
        print("Number in recommended selection:" + str(len(flat_union)))
        df = pd.read_csv(foods_csv,
                         usecols=["recipe_name", "recipe_id", "calories", "protein", "carbohydrates", "fat",
                                  "ingredients"])
        self.df = df
        self.select = df.iloc[flat_union]
        # Clean up the constraints
        defaults = {"goal": "MAX", "priority": "", "meals": [3, 5], "calories": [1500, 2000], "days": 7}
        self.options = {**defaults, **content}

    def display_results(self, menu):
        object = {"query": {**self.options}, "results": []}
        for day in menu:
            total_calories = 0
            total_fat = 0
            total_carbs = 0
            total_protein = 0
            meals = []

            for food in day:
                entry = {}
                id = int(food.replace("_", ""))
                df_obj = self.df.iloc[id]

                entry["id"] = id;
                entry["name"] = df_obj["recipe_name"]
                entry["ingredients"] = df_obj["ingredients"]


                entry["calories"] = df_obj["calories"]
                total_calories += df_obj["calories"]

                entry["carbohydrates"] = df_obj["carbohydrates"]
                total_carbs += df_obj["carbohydrates"]

                entry["fats"] = df_obj["fat"]
                total_fat += df_obj["fat"]

                entry["protein"] = df_obj["protein"]
                total_protein += df_obj["protein"]
                meals.append(entry)



            total_calories = round(total_calories, 1)
            total_carbs = round(total_carbs, 1)
            total_fat = round(total_fat, 1)
            total_protein = round(total_protein, 1)

            if "calories" in self.options and (
                    self.options["calories"][0] > total_calories or self.options["calories"][1] < total_calories):
                total_calories = str(total_calories) + "(unmet)"
            if "fats" in self.options and (self.options["fats"][0] > total_fat or self.options["fats"][1] < total_fat):
                total_fat = str(total_fat) + "(unmet)"
            if "carbohydrates" in self.options and (
                    self.options["carbohydrates"][0] > total_carbs or self.options["carbohydrates"][1] < total_carbs):
                total_carbs = str(total_carbs) + "(unmet)"

            if "protein" in self.options and (
                    self.options["protein"][0] > total_protein or self.options["protein"][1] < total_protein):
                total_protein = str(total_protein) + "(unmet)"

            object["results"].append({"calories": total_calories, "carbohydrates": total_carbs, "fats": total_fat, "protein": total_protein, "meals": meals})
        return (json.dumps(object, indent=4, sort_keys=False))

    def optimize(self):
        pool = []
        days = self.options["days"]
        print("Optimizing with options:")
        print(self.options)
        print("\n")
        while (len(pool) < days):
            pool += IPSolver(self.options, self.select.sample(frac=1)).solve()
        return pool

@app.route('/', methods=['POST', 'OPTIONS'])
def run():
    content = request.get_json()
    print(content)
    d = DietOptimizer(content)
    results = d.optimize()
    return d.display_results(results), 200, {'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*'}

if __name__ == '__main__':

    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
    #run()

