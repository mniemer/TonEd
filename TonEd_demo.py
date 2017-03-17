from TonEd_util import *

def run_demo():
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

    if len(test_paths) != 3:
        print name + " is not a valid name. Please try again!"
        return

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
        most_common_class(test_results, expected_class)

run_demo()