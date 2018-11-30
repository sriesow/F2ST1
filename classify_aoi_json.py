import json
import glob
import numpy
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
	
def train(weight):	
	clf = RandomForestClassifier(criterion="entropy",n_estimators=100, max_depth=16)
	train_files = glob.glob("AOI_json/train/*.json")
	val_files = glob.glob("AOI_json/train/*.json")
	X,y = files_to_dataset(train_files)
	Xv,yv = files_to_dataset(val_files)
	clf.fit(X,y)
	#for e in range(len(Xv)):
	prediction = clf.predict(Xv)
	tp = sum([1 if (prediction[i] == 0 and yv[i] == 0) else 0 for i in range(len(yv))])
	fp = sum([1 if (prediction[i] == 0 and yv[i] == 1) else 0 for i in range(len(yv))])
	tn = sum([1 if (prediction[i] == 1 and yv[i] == 1) else 0 for i in range(len(yv))])
	fn = sum([1 if (prediction[i] == 1 and yv[i] == 0) else 0 for i in range(len(yv))])
	p = tp/(tp+fp) if (tp+fp) > 0 else -1
	r = tp/(tp+fn)
	f = 2*p*r/(p+r)
	print("tp:\t",tp,"| fp:\t",fp)
	print("-------------------------------")
	print("tn:\t",tn,"| fn:\t",fn)
	return tp,fp,tn,fn,p,r,f

tps = []
fps = []
tns = []
fns = []
ps = []
rs = []
fs = []
y = [1]
for i in y:
	tp,fp,tn,fn,p,r,f=train(i)
	tps.append(tp)
	fps.append(fp)
	tns.append(tn)
	fns.append(fn)
	fs.append(f)
#plt.plot(y,tps, label='true positives')
plt.plot(y,fps, label='false positives')
#plt.plot(y,tns, label='true negatives')
plt.plot(y,fns, label='false negatives')
#plt.plot(y,fs, label='f score')
plt.legend()
plt.show()