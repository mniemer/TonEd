from TonEd_util import *

def run_demo():
    all_file_paths = []

    for file in os.listdir("samples/"):
        if file.endswith(".wav"):
            all_file_paths.append(os.path.join("samples/", file))

    test_path = ""
    demo = raw_input("Please enter the demo sample you'd like to test: ")
    path = demo.replace(" ", "")
    for file in os.listdir("demo/"):
        if file.startswith(path):
            test_path = os.path.join("demo/", file)

    if test_path == "":
        print demo + " is not a valid demo. Please try again."
        return

    test_model = raw_input("Please enter the model you'd like to use: ")
    print ""

    all_features = []
    all_classes = []


    for f in all_file_paths:
        tone = f.split("_")[1].split(".")[0]
        features, classes = features_by_file(f, tone)
        all_features.append(features)
        all_classes.append(classes)

    all_features = np.concatenate(all_features)
    all_classes = np.concatenate(all_classes)
    #shuffle the order!!
    all_features, all_classes = shuffle(all_features, all_classes)

    tone = test_path.split("_")[1].split(".")[0]
    test_features, test_classes = features_by_file(test_path, tone)
    test_results = run_model(all_features, all_classes, test_features, test_model)

    print "~~~ Tone Reccomendation for " + demo + " using " + test_model + " ~~~"
    result = most_common_class(test_results, tone, display=False)
    print "This scale has a " + result + " tone."
    if result == "Full":
        print "Good Job!"
    elif result == "Thin":
        print "Try using more weight with your bow."
    elif result == "Crunchy":
        print "Try using less weight with your bow."

run_demo()