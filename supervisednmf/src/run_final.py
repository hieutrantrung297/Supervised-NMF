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


set_data_list = meta['Set'].unique().tolist()
set_data_list.remove('train')
set_data_list = ['train'] + set_data_list
print(set_data_list)


dict_meta = {}
dict_X = {}
dict_y = {}

for set_data in set_data_list:
    
    meta_set_data = meta[meta['Set'] == set_data].reset_index(drop = True)

    X_set_data = pd.merge(meta_set_data[['SampleID']], feature).drop('SampleID', axis = 1)
    y_set_data = meta_set_data['Label']

    print(set_data, X_set_data.shape, y_set_data.shape)
    
    dict_meta[set_data] = meta_set_data.copy()
    dict_X[set_data] = X_set_data.copy()
    dict_y[set_data] = y_set_data.copy()


with open('{}/summary.pkl'.format(output_path), 'rb') as file:
    # Serialize and save the data to the file
    summary = pickle.load(file)
    
    
### transform snmf ###################################################################################################
for set_data in set_data_list:
    
    # Feature
    Y = dict_X[set_data].copy().values

    # Label
    u = dict_y[set_data].copy().values
    u = encode_y(u)

    # Transform SNMF
    X = summary['X'].copy()
    K, loss_list, loss_list1, loss_list2 = SNMF_transform(Y, X, iter, tolerance, patience, epsStab, alpha, epsilon)

    dict_X[set_data] = pd.DataFrame(K).copy()

    # save
    summary['loss_transform_{}'.format(set_data)] = loss_list
    summary['loss1_transform_{}'.format(set_data)] = loss_list1
    summary['loss2_transform_{}'.format(set_data)] = loss_list2


### save summary final ###################################################################################################
for set_data in set_data_list:

    # Open a file in binary write mode
    with open('{}/X_{}_transformed.pkl'.format(output_path, set_data), 'wb') as file:
        # Serialize and save the data to the file
        pickle.dump(dict_X[set_data], file)

    # Open a file in binary write mode
    with open('{}/y_{}.pkl'.format(output_path, set_data), 'wb') as file:
        # Serialize and save the data to the file
        pickle.dump(dict_y[set_data], file)

# Open a file in binary write mode
with open('{}/summary_final.pkl'.format(output_path), 'wb') as file:
    # Serialize and save the data to the file
    pickle.dump(summary, file)
    
    
### save SNMF as csv ###################################################################################################
df  = pd.DataFrame()

for set_data in set_data_list:
    sample_df = dict_meta[set_data][['SampleID']]
    data = pd.concat([sample_df, dict_X[set_data]], axis = 1)

    df = pd.concat([df, data], axis = 0)
    
df.to_csv('{}/feature.csv'.format(output_path), index = None)