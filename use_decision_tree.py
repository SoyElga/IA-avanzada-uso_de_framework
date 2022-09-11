from iris_dataset_clean import *
from framework_implementation import *
import os

path = input("Path for the Graphviz bin carpet in your computer (Example: C:/Program Files/Graphviz/bin/): ")
os.environ["PATH"] += os.pathsep + path

option = 1

def print_menu():
    print("""Options:
    1. Predict data
    2. Plot Decision Tree
    3. Print Metrics
    4. Exit
    """)
    return int(input("Select and option: "))

model = DecisionTreeModel()
model.fit(X_train, y_train)

print("--- NOTE: Iris Dataset already loaded and fitted in model for usage ---\n")
y_pred = []

while True:
    option = print_menu()
    if option == 1:
        y_pred = model.predict(X_test)
        print("Predictions:", y_pred)
        print("Predicted data with Success!")
    elif option == 2:
        model.plot_graph(feature_names = X_train.columns, class_names = species)
    elif option == 3:
        if not len(y_pred) == 0:
            model.print_metrics(y_test, class_names = species)
        else:
            print("No predictions to load metrics from")
    elif option == 4:
        break
    else:
        print("That's not a valid option")