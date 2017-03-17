from TonEd_util import *
import sys

def run_cross_validation_test():
    print ""
    print "#################################"
    print "##### Cross Validation Test #####"
    print "#################################"
    print ""

    print "Extracting features..."

    full_features, full_classes = features_by_tone("Full")
    thin_features, thin_classes = features_by_tone("Thin")
    crunchy_features, crunchy_classes = features_by_tone("Crunchy")

    #concatenate all features together
    all_features = np.concatenate([full_features, thin_features, crunchy_features])
    all_classes = np.concatenate([full_classes, thin_classes, crunchy_classes])
    #shuffle the order!!
    all_features, all_classes = shuffle(all_features, all_classes)

    print "Features extracted."
    print ""
    print "Running cross validation tests..."

    dtree_precision = 0
    neighbor_precision = 0
    sgd_precision = 0
    mlp_precision = 0

    for i in range(100):
        dtree_precision = dtree_precision + cross_validation(all_features, all_classes, "Decision Tree", 10, display=False)
        neighbor_precision = neighbor_precision + cross_validation(all_features, all_classes, "Nearest Neighbor", 10, display=False)
        sgd_precision = sgd_precision + cross_validation(all_features, all_classes, "Stochastic Gradient Descent", 10, display=False)
        mlp_precision = mlp_precision + cross_validation(all_features, all_classes, "Multi-layer Perceptron", 10, display=False)
        sys.stdout.write('#')
        sys.stdout.flush()

    print ""
    print "Finished running cross validation tests."
    print ""
    print "~~~ Cross Validation Test Results ~~~"
    print "Decision Tree Precision: " + str(dtree_precision/100.)
    print "Nearest Neighbor Precision: " + str(neighbor_precision/100.)
    print "Stochastic Gradient Descent Precision: " + str(sgd_precision/100.)
    print "Multi-layer Perceptron Precision: " + str(mlp_precision/100.)

def run_scale_classification_test():
    print ""
    print "################################"
    print "##### Classify Scales Test #####"
    print "################################"
    print ""

    all_file_paths = []

    for file in os.listdir("samples/"):
        if file.endswith(".wav"):
            all_file_paths.append(os.path.join("samples/", file))

    names = ["Matthew Niemer", "Emily Ott", "Ann Duchow", "Kaitlin Moran", "Jon Huang", "Megan Renner"]
    models = {"Decision Tree" : 0, "Nearest Neighbor" : 0, "Stochastic Gradient Descent" : 0, "Multi-layer Perceptron" : 0}
    total = 18

    for name in names:
        print "Classifying scales for " + name + "..."
        test_name = name.replace(" ", "")          
        test_paths = [path for path in all_file_paths if path.startswith("samples/" + test_name)]
        file_paths = [path for path in all_file_paths if not path.startswith("samples/" + test_name)]

        all_features = []
        all_classes = []
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

        for test_model in list(models.keys()): 
            for i in range(len(test_features)):
                expected_class = test_classes[i]
                test_results = run_model(all_features, all_classes, test_features[i], test_model)
                result = most_common_class(test_results, expected_class, display=False)
                if result == expected_class:
                    models[test_model] = models[test_model] + 1

    print ""
    for test_model in list(models.keys()):
        print "~~~ Scale Classification Results for " + test_model + " ~~~"
        print str(models[test_model]) + " scales out of " + str(total) + " classified correctly."
        print ""

if len(sys.argv) <= 1:
    run_cross_validation_test()
    run_scale_classification_test()

elif len(sys.argv) == 2 and sys.argv[1] == "-cv":
    run_cross_validation_test()

elif len(sys.argv) == 2 and sys.argv[1] == "-sc":
    run_scale_classification_test()
