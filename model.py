from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from hmmlearn import hmm


# Step 2: Specify the initial parameters
data = np.genfromtxt('data/binary_data.csv', delimiter=',')  # Replace this with your actual data
data = pd.read_csv('data/binary_data.csv')
data_NDVI = data["NDVI"]

def fitHMM(Q, test, test_label, nSamples):
    # fit Gaussian HMM to Q
    model = hmm.GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q,[len(Q),1]))
     
    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(test,[len(test),1]))
 
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
 
    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q,[len(Q),1]))
 
    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)
 
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    accuracy = accuracy_score(test_label, hidden_states)
    conf_matrix = confusion_matrix(test_label, hidden_states)
    classification_rep = classification_report(test_label, hidden_states)
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', classification_rep)

    return hidden_states, mus, sigmas, P, logProb, samples

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
NDVI_train = train_data["NDVI"]
NDVI_train_values = NDVI_train.values
NDVI_test = test_data["NDVI"]
NDVI_test_values = NDVI_test.values
label_test = test_data["description"]

# log transform the data and fit the HMM
hidden_states, mus, sigmas, P, logProb, samples = fitHMM(NDVI_train_values, NDVI_test_values, label_test, len(NDVI_train_values))
print(f"Hidden states: {hidden_states}")
print(f"mus: {mus}")
print(f"sigmas: {sigmas}")
print(f"P: {P}")
print(f"logProb: {logProb}")
print(f"samples: {samples}")


# figure out from true labels

df_train_ndvi = train_data[["description", "NDVI"]]
# Mean for not-flooded label, based on our train data
df_train_0 = df_train_ndvi[df_train_ndvi["description"] == 0]
mean_0 = df_train_0["NDVI"].mean()

print("mean for not flooded", mean_0)
# Mean for flooded label, based on our train data
df_train_1 = df_train_ndvi[df_train_ndvi["description"] == 1]
mean_1 = df_train_1["NDVI"].mean()
print("mean for flooded", mean_1)

from matplotlib import pyplot as plt
import seaborn as sns
 
# Plot the final results, compare with original truth
def plotTimeSeries(Q, hidden_states, ylabel, filename):
 
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    xs = np.arange(len(Q))+1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], Q[masks], c='r', label='Not flooded')
    masks = hidden_states == 1
    ax.scatter(xs[masks], Q[masks], c='b', label='Flooded')
     
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()
 
    return None
 
plt.switch_backend('agg') # turn off display when running with Cygwin
plotTimeSeries(NDVI_train_values, hidden_states, 'description', 'img.png')

plot = sns.scatterplot(data=train_data, x="date", y="NDVI", hue="description")
fig = plot.get_figure()
fig.savefig("true.png")

