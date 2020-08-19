def warn(*args,**kwargs): #Keras spits out a bunch of junk
    pass
import warnings
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Tensorflow also spits out a bunch of junk
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.compat.v2.random.set_seed(0)
import random
random.seed(0)

#these two are included scripts
import preprocess
import utilities

#getModel and getPrediction are the two main functions. Build the model and load the parameters,
#then evaluate by supplying two lists, one of the depth1/2 smiles and one of the depth0 smiles
#I've shown an example below
def getModel(weights):
    params = {
        'n_layers'  :3,
        'n_nodes'   :256,
        'fp_length' :256,
        'fp_depth'  :3,
        'conv_width':30,
        'L2_reg'    :0.0004, #The rest of the parameters don't really do anything outside of training
        'batch_normalization':1,
        'learning_rate':1e-4,
        'input_shape':2
    }

    from gcnn_model import build_fp_model
    
    predictor_MLP_layers = []
    for l in range(params['n_layers']):
        predictor_MLP_layers.append(params['n_nodes'])

    model = build_fp_model(
        fp_length = params['fp_length'],
        fp_depth = params['fp_depth'],
        conv_width=params['conv_width'],
        predictor_MLP_layers=predictor_MLP_layers,
        L2_reg=params['L2_reg'],
        batch_normalization=params['batch_normalization'],
        lr = params['learning_rate'],
        input_size = params['input_shape']
    )

    model.load_weights(weights)
    return model
    
def getPrediction(smiles,R0_smiles):
    X_eval = (smiles,R0_smiles)
    processed_eval = preprocess.neuralProcess(X_eval)
    predictions = np.squeeze(model.predict_on_batch(x=processed_eval))
    return predictions

model = getModel('nfp.h5')

pnk_smiles = []
with open('smiles','r') as f:
    for line in f:
        pnk_smiles.append(line.strip())

pnk_alt = []
with open('R0-smiles','r') as f:
    for line in f:
        pnk_alt.append(line.strip())

preds = getPrediction(['CC(c1ccccc1)C'],['c1ccccc1'])
print(preds)
exit()
preds = getPrediction(pnk_smiles,pnk_alt)
y_test = np.genfromtxt('hf_diff_298k')
mae = np.mean(np.squeeze(np.absolute(preds-y_test)))
medae = np.median(np.squeeze(np.absolute(preds-y_test)))

print('Final mae: {}'.format(mae))
print('Final median: {}'.format(medae))
