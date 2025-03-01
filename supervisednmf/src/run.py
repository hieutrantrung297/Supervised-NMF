import sys
sys.path.append("src/function_help")
from import_library_and_function import *
from metric import *
start = time.time()

# input

feature_path = str(sys.argv[1]) # path to feature file
meta_path = str(sys.argv[2]) # path to meta data file
output_path = str(sys.argv[3]) # path to output_dir

nmf_init_mode = str(sys.argv[4]) # how to init nmf
loss_type = str(sys.argv[5]) # how to init nmf

if loss_type == 'LR':
    from SNMF_with_LR_loss import *
elif loss_type == 'SVM':
    from SNMF_with_SVM_loss import *
elif loss_type == 'LDA':
    from SNMF_with_LDA_loss import *

feature_name = str(sys.argv[6]) # feature name
rank = int(sys.argv[7]) # number of new features

iter = int(sys.argv[8]) # number of iteractions
tolerance = float(sys.argv[9]) # cutoff check improve or not
patience = int(sys.argv[10]) # number of iteractions to check improvement
alpha = float(sys.argv[11]) # alpha in ADADELTA
epsilon = float(sys.argv[12]) # epsilon in ADADELTA


epsStab = 2*2.220446049250313e-16

os.system('mkdir -p {}'.format(output_path))

### main ###################################################################################################

### read data

meta = read_meta(meta_path)
meta['Label'].value_counts()

feature = read_feature(feature_path, meta)
feature.shape

X_train = pd.merge(meta[['SampleID']], feature).drop('SampleID', axis = 1)
y_train = meta['Label']


### training snmf ###################################################################################################
fig, ax = plt.subplots(1, 1, figsize=[7, 5])

summary = {}

max_ylim = 0
    
# Feature
Y = X_train.copy().values

# Label
u = y_train.copy().values
u = encode_y(u)

# Training SNMF
X, loss_list, loss_list1, loss_list2, weight = SNMF(Y, u, rank, iter, tolerance, patience, epsStab, alpha, epsilon, nmf_init_mode)

# Save result
summary['X'] = X
summary['weight'] = weight
summary['loss'] = loss_list
summary['loss1'] = loss_list1
summary['loss2'] = loss_list2

# Update ylim to plot
if np.max(loss_list2) > max_ylim:
    max_ylim = np.max(loss_list2)

ax.plot(loss_list1, lw=1, color = 'red', label = 'NMF')
ax.plot(loss_list2, lw=1, color = 'green', label = 'Classification model')
    
ax.set_title('{}: Fit loss'.format(feature_name), fontsize=16)
ax.tick_params(labelsize=12)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
plt.ylim([0, max_ylim * 102/100])
plt.grid()
plt.legend(title="Loss")
plt.savefig('{}/Fit_loss.png'.format(output_path))
plt.show()


### transform snmf ###################################################################################################

fig, ax = plt.subplots(1, 1, figsize=[7, 5])

# Feature
Y = X_train.copy().values

# Label
u = y_train.copy().values
u = encode_y(u)

# Transform SNMF
X = summary['X'].copy()
K, loss_list, loss_list1, loss_list2 = SNMF_transform(Y, X, iter, tolerance, patience, epsStab, alpha, epsilon)

# save
summary['loss_transform_train'] = loss_list
summary['loss1_transform_train'] = loss_list1
summary['loss2_transform_train'] = loss_list2

ax.plot(loss_list1, lw=1, color = 'red', label = 'NMF')
ax.plot(loss_list2, lw=1, color = 'green', label = 'Classification model')
    
ax.set_title('{}: Train transform loss'.format(feature_name), fontsize=16)
ax.tick_params(labelsize=12)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
plt.ylim([0, max_ylim * 102/100])
plt.grid()
plt.legend(title="Loss")
plt.savefig('{}/Train_transform_loss.png'.format(output_path))
plt.show()


### final loss ###################################################################################################

loss, loss1, loss2 = cost_function(Y, K, X, u, weight)

summary['loss_transform_train_final'] = loss
summary['loss1_transform_train_final'] = loss1
summary['loss2_transform_train_final'] = loss2


### save summary ###################################################################################################
# Open a file in binary write mode
with open('{}/summary.pkl'.format(output_path), 'wb') as file:
    # Serialize and save the data to the file
    pickle.dump(summary, file)