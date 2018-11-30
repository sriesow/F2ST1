import json
import glob
import numpy
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

#from sklearn.datasets import make_classification



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
	clf = RandomForestClassifier(criterion="entropy",n_estimators=100, max_depth=16, max_leaf_nodes=128)#class_weight="balanced_subsample")#, max_leaf_nodes=200),class_weight="balanced_subsample")
	#clf = BaggingClassifier(n_estimators=15,max_samples=0.8, max_features=1.0)
	clf.fit(X,y)
	#for e in range(len(Xv)):
	prediction = clf.predict(Xv)
	#print(sklearn.metrics.f1_score(yv,prediction,average="weighted"))
	#tp = sum([1 if (prediction[i] == 0 and yv[i] == 0) else 0 for i in range(len(yv))])
	fp = sum([1 if (prediction[i] == 0 and yv[i] == 1) else 0 for i in range(len(yv))])
	#tn = sum([1 if (prediction[i] == 1 and yv[i] == 1) else 0 for i in range(len(yv))])
	fn = sum([1 if (prediction[i] == 1 and yv[i] == 0) else 0 for i in range(len(yv))])
	print(fp,fn)
	#p = tp/(tp+fp) if (tp+fp) > 0 else -1
	#r = tp/(tp+fn) 
	#f = 2*p*r/(p+r)
	#print("tp:\t",tp,"| fp:\t",fp)
	#print("-------------------------------")
	#print("tn:\t",tn,"| fn:\t",fn)
	return prediction, clf#tp,fp,tn,fn,p,r,f

#tps = []
#fps = []
#tns = []
#fns = []
#ps = []
#rs = []
#fs = []
#y = [1]
#for i in y:
#	tp,fp,tn,fn,p,r,f=train(i)
#	tps.append(tp)
#	fps.append(fp)
#	tns.append(tn)
#	fns.append(fn)
#	fs.append(f)
#plt.plot(y,tps, label='true positives')
#plt.plot(y,fps, label='false positives')
#plt.plot(y,tns, label='true negatives')
#plt.plot(y,fns, label='false negatives')
#plt.plot(y,fs, label='f score')
#plt.legend()
#plt.show()
train_files = glob.glob("../AOI_json/train/*.json")
val_files = glob.glob("../AOI_json/train/*.json")
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