import json
import glob
import numpy
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz



def files_to_dataset(files):
	pos_data = []
	neg_data = []
	for file in files:
		#print(file)
		with open(file, "r") as f:
			jason = json.load(f)
			pos = jason["component"]["mark sav"] == "Pseudofehler"
			if pos:
				pos_data.append(numpy.array([e["value"] for e in jason["component"]["features"]]))
			else:
				neg_data.append(numpy.array([e["value"] for e in jason["component"]["features"]]))
	X = pos_data + neg_data
	y = [1]*len(pos_data) + [0]*len(neg_data)
	return X,y
	
def train(X,y,Xv,yv):	
	#clf = RandomForestClassifier(criterion="entropy",n_estimators=100, max_depth=16, max_leaf_nodes=128)#class_weight="balanced_subsample")#, max_leaf_nodes=200),class_weight="balanced_subsample")
	#clf.fit(X,y)
	from joblib import dump, load
	#dump(clf, 'RF_classifier.joblib') 
	clf = load('RF_classifier.joblib') 
	prediction = clf.predict(Xv)
	return prediction, clf

train_files = glob.glob("../AOI_json/train/*.json")
val_files = glob.glob("../AOI_json/validation/*.json")
X,y = files_to_dataset(train_files)
Xv,yv = files_to_dataset(val_files)
predictions, model=train(X,y,Xv,yv)

export_graphviz(model.estimators_[1], out_file='tree.dot',
	feature_names = [e for e in range(128)],
	class_names = ["P","F"],
	rounded = True, proportion = False, 
	precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

with open("predictions_AOI_RF_json.csv", "w") as out:
	#out.write("image_filename,Result\n")
	for i in range(len(yv)):
		out.write(val_files[i].split("\\")[-1] + "," + ("P" if predictions[i] == 1 else "F") + "\n")
with open("groundtruth_AOI_json.csv", "w") as out:
	for i in range(len(yv)):
		out.write(val_files[i].split("\\")[-1] + "," + ("P" if yv[i] == 1 else "F") + "\n")