from TonEd_util import *

print ""
print "##########################"
print "### Feature extraction ###"
print "##########################"
print ""

#get full features
full_features, full_classes = features_by_tone("Full")
plt.figure(figsize=(20, 4))
for feat in full_features:
    plt.plot(feat)
    plt.title("Full Features")
plt.show()

#get thin features
thin_features, thin_classes = features_by_tone("Thin")
plt.figure(figsize=(20, 4))
for feat in thin_features:
    plt.plot(feat)
    plt.title("Thin Features")
plt.show()

#get crunchy features
crunchy_features, crunchy_classes = features_by_tone("Crunchy")
plt.figure(figsize=(20, 4))
for feat in crunchy_features:
    plt.plot(feat)
    plt.title("Crunchy Features")
plt.show()

print ""
print "##########################"
print "#### Cross Validation ####"
print "##########################"
print ""

#concatenate all features together
all_features = np.concatenate([full_features, thin_features, crunchy_features])
all_classes = np.concatenate([full_classes, thin_classes, crunchy_classes])
#shuffle the order!!
all_features, all_classes = shuffle(all_features, all_classes)

#run different models
cross_validation(all_features, all_classes, "Decision Tree", 18)
cross_validation(all_features, all_classes, "Nearest Neighbor", 18)
cross_validation(all_features, all_classes, "Stochastic Gradient Descent", 18)
#cross_validation(all_features, all_classes, "SVM", 18)
cross_validation(all_features, all_classes, "Multi-layer Perceptron", 18)

print ""
print "###########################"
print "##### Classify Scales #####"
print "###########################"
print ""


## Let's classify some specific scales now...
all_file_paths = []

for file in os.listdir("samples/"):
    if file.endswith(".wav"):
        all_file_paths.append(os.path.join("samples/", file))

name = raw_input("Please enter the name you'd like to test: ")
test_name = name.replace(" ", "")
test_model = raw_input("Please enter the model you'd like to use: ")
print ""

all_features = []
all_classes = []

test_paths = [path for path in all_file_paths if path.startswith("samples/" + test_name)]
file_paths = [path for path in all_file_paths if not path.startswith("samples/" + test_name)]

for f in file_paths:
    tone = f.split("_")[1].split(".")[0]
    features, classes = features_by_file(f, tone)
    all_features.append(features)
    all_classes.append(classes)

all_features = np.concatenate(all_features)
all_classes = np.concatenate(all_classes)
#shuffle the order!!
all_features, all_classes = shuffle(all_features, all_classes)

#get each scale that I played
test_features = []
test_classes = []
for test_path in test_paths:
    tone = test_path.split("_")[1].split(".")[0]
    test_feature, test_class = features_by_file(test_path, tone)
    test_features.append(test_feature)
    test_classes.append(tone)

print "~~~ Scale Classification Results for " + name + " using " + test_model + " ~~~"

for i in range(len(test_features)):
    expected_class = test_classes[i]
    test_results = run_model(all_features, all_classes, test_features[i], test_model)
    print_most_common_class(test_results, expected_class)