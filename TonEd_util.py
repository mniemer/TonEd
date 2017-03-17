import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, math, os, sys
#import librosa.display as display
#from IPython.display import Audio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree, linear_model
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from collections import Counter

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)[1:14, :]
    mean_mfccs = np.mean(mfccs, axis = 1)
    var_mfccs = np.var(mfccs, axis = 1)
    delta_mfccs = np.diff(mfccs, axis = 1)
    mean_delta_mfccs = np.mean(delta_mfccs, axis = 1)
    var_delta_mfccs = np.var(delta_mfccs, axis = 1)
    #return np.concatenate([var_mfccs, var_delta_mfccs])
    return np.concatenate([mean_mfccs, var_mfccs, mean_delta_mfccs, var_delta_mfccs])
    
def analyze_results(results):
    true_positives = sum([r[0] == r[1] for r in results])
    false_positives = sum([r[0] != r[1] for r in results])
    precision = true_positives / float(true_positives + false_positives)
    return precision

def print_results(method, results):
    print "~~~ " + method + " Results ~~~"
    print "Precision: " + str(analyze_results(results))
    print "Confusion matrix:"
    print confusion_matrix([r[0] for r in results], [r[1] for r in results])
    print ""
    
def features_by_name(name):
    fullstr = "samples/" + name + "_Full.wav"
    crunchystr = "samples/" + name + "_Crunchy.wav"
    thinstr = "samples/" + name + "_Thin.wav"
    pathstrings = [fullstr, thinstr, crunchystr]
    
    features = []
    classes = []
    for path in pathstrings:
        if (path.endswith("Full.wav")):
            tone = "Full"
        elif (path.endswith("Crunchy.wav")):
            tone = "Crunchy"
        elif (path.endswith("Thin.wav")):
            tone = "Thin"
        sr = 44100.
        sample, sr = librosa.load(path, sr)
        frames = librosa.util.frame(sample, 26460, hop_length = 26460)
        for frame in range(frames.shape[-1]):
            feature = extract_features(frames[:, frame], sr)
            features.append(feature)
            classes.append(tone)
            
    return features, classes

def features_by_file(path, tone):
    features = []
    classes = []
    sr = 44100.
    sample, sr = librosa.load(path, sr)
    frames = librosa.util.frame(sample, 26460, hop_length = 26460)
    for frame in range(frames.shape[-1]):
            feature = extract_features(frames[:, frame], sr)
            features.append(feature)
            classes.append(tone)
            
    return features, classes

def features_by_tone(tone):
    features = []
    classes = []
    for file in os.listdir("samples/"):
        if file.endswith(tone + ".wav"):
            sr = 44100.
            pathstr = os.path.join("samples/", file)
            sample, sr = librosa.load(pathstr, sr)
            frames = librosa.util.frame(sample, 26460, hop_length = 26460)
            for frame in range(frames.shape[-1]):
                feature = extract_features(frames[:, frame], sr)
                features.append(feature)
                classes.append(tone)
    
    return features, classes

def cross_validation(features, classes, model, folds, display=True):
    results = []
    for i in range(folds):
        start = i*features.shape[0]/folds
        stop = (i+1)*features.shape[0]/folds
        cross_validation_set = np.concatenate((features[0:start, :], features[stop:, :]))
        cross_validation_classes = np.concatenate((classes[0:start], classes[stop:]))

        if model == "SVM":
            clf = sklearn.svm.SVC()
        elif model == "Nearest Neighbor":
            clf = KNeighborsClassifier(n_neighbors=15, algorithm='auto')
        elif model == "Decision Tree":
            clf = tree.DecisionTreeClassifier()
        elif model == "Stochastic Gradient Descent":
            clf = linear_model.SGDClassifier()
        elif model == "Multi-layer Perceptron":
            clf = MLPClassifier()
        else:
            print model +" is not a valid model."
            return
        
        clf.fit(cross_validation_set, cross_validation_classes)  
        for j in range(start, stop):        
            result = clf.predict(features[j, :].reshape(1, -1))
            results.append((classes[j], result))

    if display:
        print_results(model, results)
        
    return analyze_results(results)[0]
    
def run_model(train_features, train_classes, test_features, model):
    results = []
    if model == "SVM":
        clf = sklearn.svm.SVC()
    elif model == "Nearest Neighbor":
        clf = KNeighborsClassifier(n_neighbors=15, algorithm='auto')
    elif model == "Decision Tree":
        clf = tree.DecisionTreeClassifier()
    elif model == "Stochastic Gradient Descent":
        clf = linear_model.SGDClassifier()
    elif model == "Multi-layer Perceptron":
        clf = MLPClassifier()
    else:
        print model +" is not a valid model."
        return

    clf.fit(train_features, train_classes)
    for feature in test_features:
        result = clf.predict(feature.reshape(1,-1))
        results.append(result[0])
    
    return results

def most_common_class(results, expected, display=True):
    data = Counter(results)
    most_common = data.most_common(1)
    if display:
        print "Expected Tone Class: " + expected
        print "The most common class was " + most_common[0][0] + " (Predicted " + str(most_common[0][1]) + " times, " +  str(len(results)) + " total)."
        print ""
    return most_common[0][0]