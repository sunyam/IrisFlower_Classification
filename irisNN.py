import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder

df = pandas.read_csv('dataset/iris.csv', header=None)
data = df.values

X = data[:,0:4].astype(float)
Y = data[:,4]

#print len(X)
#print len(Y)
# 150 rows

seed = 7
numpy.random.seed(seed)

# Convert string-outputs to one-hot vector
enc = LabelEncoder()
enc.fit(Y)
enc_Y = enc.transform(Y)
#print enc_Y    # [0,0,0,0,..,1,1,1,1,..,2,2,2,2,..]

y = np_utils.to_categorical(enc_Y)
#print y    # [[ 1.  0.  0.]
#              [ 1.  0.  0.]
#              [ 1.  0.  0.]
#              [ 1.  0.  0.]....]

# Function to pass to KerasClassifier
def create_nn():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=create_nn, nb_epoch=200, batch_size=5, verbose=True)

kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)

result = cross_val_score(model, X, y, cv=kfold)
print "\n"
print result.mean()*100
print "\n"
print result.std()*100