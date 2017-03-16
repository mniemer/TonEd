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

#get features by name
emily_features, emily_classes = features_by_name("EmilyOtt")
jon_features, jon_classes = features_by_name("JonHuang")
megan_features, megan_classes = features_by_name("MeganRenner")
ann_features, ann_classes = features_by_name("AnnDuchow")
kaitlin_features, kaitlin_classes = features_by_name("KaitlinMoran")
matt_features, matt_classes = features_by_name("MatthewNiemer")

#concatenate all features together (leave mine out)
all_features = np.concatenate([emily_features, megan_features, ann_features, jon_features, kaitlin_features])
all_classes = np.concatenate([emily_classes, megan_classes, ann_classes, jon_classes, kaitlin_classes])
#shuffle the order!!
all_features, all_classes = shuffle(all_features, all_classes)

#get each scale that I played
full_features = features_by_file('samples/MatthewNiemer_Full.wav')
crunchy_features = features_by_file('samples/MatthewNiemer_Crunchy.wav')
thin_features = features_by_file('samples/MatthewNiemer_Thin.wav')

#full_results = run_model(all_features, all_classes, full_features, "Nearest Neighbor")
#crunchy_results = run_model(all_features, all_classes, crunchy_features, "Nearest Neighbor")
#thin_results = run_model(all_features, all_classes, thin_features , "Nearest Neighbor")
full_results = run_model(all_features, all_classes, full_features, "Decision Tree")
crunchy_results = run_model(all_features, all_classes, crunchy_features, "Decision Tree")
thin_results = run_model(all_features, all_classes, thin_features , "Decision Tree")
print_most_common_class(full_results, "Full")
print_most_common_class(crunchy_results, "Crunchy")
print_most_common_class(thin_results, "Thin")