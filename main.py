# Machine Learning Classifier

# There are many types of classifiers - we will be using a Decision Tree Classifier
# Decision Tree Classifier = a supervised learning algorithm that can be used for both
# classification and regression.

# Whatever classifier you use, it should be able to take in the same set of features
# and output the same class label.


# Import libraries
from sklearn import tree  # DecisionTreeClassifier from the 'tree' module


# The features are the characteristics of the data that are being classified.
# The labels (aka targets) are the classes that are having data classified into.


# Feature List:  Wing Span  and  Fuselage Length
# NOTE: The order of the features is important & these are NOT LABELS.
# ORDER of features: (Wing Span, Fuselage Length)

aircraft_features = [[42, 60], [35, 55], [29, 35], [23, 24], [26, 28], [22, 25]]

# Now that the features have been defined, the labels (aka targets) must be defined!
# The labels are the classes that are having data classified into.

# ORDER IS CRITICAL! The order of the labels must match the order of the features.
# Label List: Bomber, Bomber, Bomber, Fighter, Fighter, Fighter

aircraft_labels = [0, 0, 0, 1, 1, 1]  # 0 = Bomber, 1 = Fighter


# Now that both features & labels are defined, create the DT classifier!
aircraft_DT_classifier = tree.DecisionTreeClassifier()


# ********** TRAINING THE CLASSIFIER **********
# With the classifier created, it's time to train it - using the fit() method!

aircraft_DT_classifier = aircraft_DT_classifier.fit(aircraft_features, aircraft_labels)


# OUTPUT the training process to the console
def training_output():
    # Title
    print("######################\n"
          "### Training Data: ###\n"
          "######################\n"
          "\n. . . . . . . . . . . .\n\n")

    # Output the features
    print("Aircraft Features (ft.) :\n")
    print("\t\t\tWing Span:\t\tFuselage Length:")

    # Initialize counter (for the 3 Bombers vs. 3 Fighters)
    counter = 0
    # Iterate over 'aircraft_features' list (retrieving each tuple as vars ['wingspan', 'fuselage_length'])
    for wingspan, fuselage_length in aircraft_features:
        counter += 1  # Increment counter by 1
        if counter > 3:  # If the counter is greater than 3 (4-6), output "Fighter"
            print("Fighter", end="\t|\t")  # last 3 items in the table
        else:  # Otherwise, output "Bomber" ( if the counter is 3 or less (1-3) )
            print(" Bomber", end="\t|\t")  # first 3 items in the table

        # Output formatted table of 'aircraft_features'
        print("\t" + str(wingspan) + "\t\t\t\t\t" + str(fuselage_length))

    print("\n\n. . . . . . . . . . . .\n\n\n\n\n\n")


# Prediction function - outputs formatted prediction (given a tuple of both data values)
def prediction(data):
    print("########################################")
    print("Given the following data...\n")
    for wingspan, fuselage_length in data:
        # Output the data to predict from
        print("Wingspan: " + str(wingspan) + "ft.")
        print("Fuselage Length: " + str(fuselage_length) + "ft.")
        # Prediction output intro
        print("\n\nThe type of aircraft predicted by the algorithm is:")
        print("PREDICTION:", end=" ")

        # Output the prediction - ( "Bomber" if it's 0, "Fighter" if otherwise (1) )
        print("Bomber" if aircraft_DT_classifier.predict([[wingspan, fuselage_length]]) == 0 else "Fighter")
        # print(aircraft_DT_classifier.predict([[wingspan, fuselage_length]]))

    print("########################################\n")
    print("\n\n")


# **********


# Output training information to user
training_output()


# ********** MAKING PREDICTIONS **********
# Using the predict() method, predictions can be made on new data that hasn't been seen before!

# Bomber PREDICTION
prediction([[35, 45]])

# Fighter PREDICTION
prediction([[22, 26]])


# Due to the nature of this program as an experimental delve into how ML decision-making operates,
# I decided to not take user input for predictions, but rather to output the process of both the
# training and predictions in an easy-to-read format, to demonstrate the entire process to the user.

# Basically, I chose to make it like a demonstration, rather than an interactive program  :)
