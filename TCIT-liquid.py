# This is file is write to 
# 1. Indentify whether this is a minimal structure or not, if so, append it into database
# 2. use Taffi component theory to get prediction of Hf
# Author: Qiyuan Zhao, Nicolae C. Iovanac

def warn(*args,**kwargs): #Keras spits out a bunch of junk
    pass
import warnings
warnings.warn = warn

import sys,os,argparse,subprocess
import numpy as np
from fnmatch import fnmatch
import tensorflow as tf
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Tensorflow also spits out a bunch of junk
np.random.seed(0)
tf.compat.v2.random.set_seed(0)
random.seed(0)

# import taffi related functions
sys.path.append('utilities')
from taffi_functions import * 
from deal_ring import get_rings,return_smi

sys.path.append('SIMPOL')
from SIMPOL import calculate_PandHvap

# import Machine learning related functions
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/ML-package')
import preprocess
import utilities

def main(argv):

    parser = argparse.ArgumentParser(description='This script predicts the enthalpy of formation for given target compounds based '+\
                                                 'on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component '+\
                                                 'Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. Further ring correction is added' +\
                                                 'distributed with the paper "Ring correction"' +\
                                                 'The script operates on .xyz files, prints the components that it uses for each prediction, and will return the '+\
                                                 '0K and 298K enthalpy of formation. No ring corrections are supplied in this version '+\
                                                 'of the script, and compounds that require missing CAVs are skipped.' )

    #optional arguments                                                                                                                   
    parser.add_argument('-t', dest='Itype', default='xyz',
                        help = 'Controls the input type, either xyz of smiles (default: "xyz")')

    parser.add_argument('-i', dest='input_name',
                        help = 'If input type is xyz, the program loops over all of the .xyz files in this input folder and makes Hf predictions for them (default: "input_xyz",\
                                If input type is smiles, the program loops over all of the smiles string in the given file (default: "input.txt")')

    parser.add_argument('-o', dest='outputname', default='result',
                        help = 'Controls the output file name for the results (default: "results")')

    parser.add_argument('-db', dest='dbfile', default='database/TAFFI_HF.db',
                        help = 'The directory path of TCIT CAVs database. (default: "TAFFI_Hf.db"')

    parser.add_argument('-g4db', dest='g4dbfile', default='database/Hf_G4.db',
                        help = 'The directory path of TCIT CAVs database. (default: "TAFFI_Hf.db"')

    parser.add_argument('-rcdb', dest='rcdbfile', default='database/depth0_RC.db',
                        help = 'The directory path of TCIT CAVs database. (default: "TAFFI_Hf.db"')

    # parse configuration dictionary (c)                                                                                                   
    print("parsing calculation configuration...")
    args=parser.parse_args()    
    # set default value
    if args.Itype not in ['xyz','smiles']:
        print("Warning! input_type must be either xyz or smiles, use default xyz...")
        args.Itype = 'xyz'

    if args.Itype == 'xyz' and args.input_name == None:
        args.input_name = 'input_xyz'

    if args.Itype == 'smiles' and args.input_name == None:
        args.input_name = 'input.txt'
    
    ##################################################                                                                                    

    # Initializing dictionary

    ##################################################                                                                                   

    kj2kcal = 0.239006
    H2kcal =  627.509

    # Tabulated enthalpies of formation for gaseous atoms, taken from Curtiss et al. J Chem Phys, 1997
    Hf_atom_0k = { "H":51.63, "Li":37.69, "Be":76.48, "B":136.2, "C":169.98, "N":112.53, "O":58.99, "F":18.47, "Na":25.69, "Mg":34.87, "Al":78.23, "Si":106.6, "P":75.42, "S":65.66, "Cl":28.59 ,"Br":26.77}

    # G4 Atom Energy table, taken from Curtiss et al. J Chem Phys, 2007                                                                   
    Atom_G4 = { "H":-0.501420, "Li":-7.46636, "Be":-14.65765, "B":-24.64665, "C":-37.834168, "N":-54.57367, "O":-75.0455, "F":-99.70498,\
                "Na":-162.11789, "Mg":-199.91204, "Al":-242.22107, "Si":-289.23704, "P":-341.13463, "S":-397.98018, "Cl":-460.01505,\
                "Br":-2573.5854}

    # H(298K) - H(0K) values for gaseous atoms, taken from Curtiss et al. J Chem Phys, 1997
    H298km0k   = { "H":1.01,  "Li":1.10,  "Be":0.46,  "B":0.29,  "C":0.25,   "N":1.04,   "O":1.04,  "F":1.05,  "Na":1.54,  "Mg":1.19,  "Al":1.08,  "Si":0.76,  "P":1.28,  "S":1.05,  "Cl":1.10, "Br":1.48}

    # Initialize periodic table
    periodic = { "H": 1,  "He": 2,\
                 "Li":3,  "Be":4,                                                                                                      "B":5,    "C":6,    "N":7,    "O":8,    "F":9,    "Ne":10,\
                 "Na":11, "Mg":12,                                                                                                     "Al":13,  "Si":14,  "P":15,   "S":16,   "Cl":17,  "Ar":18,\
                  "K":19, "Ca":20,   "Sc":21,  "Ti":22,  "V":23,  "Cr":24,  "Mn":25,  "Fe":26,  "Co":27,  "Ni":28,  "Cu":29,  "Zn":30, "Ga":31,  "Ge":32,  "As":33, "Se":34,  "Br":35,   "Kr":36,\
                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}    

    invert_periodic = {}
    for p in periodic.keys():
        invert_periodic[periodic[p]]=p

    # load database
    FF_dict = parse_HF_database(args.dbfile)    
    G4_dict = parse_G4_database(args.g4dbfile)
    ring_dict = parse_ringcorr(args.rcdbfile)
    
    # create similarity match dictionary 
    # For new conformers which not included in G4 database, find alternative conformer for instead
    similar_match = {}
    for inchi in G4_dict["HF_0"].keys():
        similar_match[inchi[:14]]=inchi
    
    sys.stdout = Logger(args.outputname)

    # find all xyz files in given folder
    if args.Itype == 'xyz':
        target_xyzs=[os.path.join(dp, f) for dp, dn, filenames in os.walk(args.input_name) for f in filenames if (fnmatch(f,"*.xyz"))]
        items      = sorted(target_xyzs)
    else:
        target_smiles = []
        with open(args.input_name,"r") as f:
            for line in f:
                target_smiles.append(line.strip())
        items = target_smiles
        
    # load in ML model
    base_model = getModel()

    # create result dict
    TCITresult = {}

    # loop for target xyz files
    for i in items:
        print("Working on {}...".format(i))
        if args.Itype == 'xyz':
            E,G = xyz_parse(i)
            name  = i.split('/')[-1]
            
        else:
            E,G = parse_smiles(i)
            smiles = i
            name = i
            
        if True in [element.lower() not in ['h','c','n','o','f','s','cl','br','p'] for element in E]:
            print("can't deal with some element in this compounds")
            continue

        adj_mat = Table_generator(E,G)
        # if input type is xyz, generate smiles string
        if args.Itype == 'xyz':
            smiles= return_smi(E,G,adj_mat)
                        
        atom_types = id_types(E,adj_mat,2)
        # replace "R" group by normal group
        atom_types = [atom_type.replace('R','') for atom_type in atom_types]

        # identify ring structure
        ring_inds     = [ring_atom(adj_mat,j) for j,Ej in enumerate(E)] 
        ring_corr_0K  = 0
        ring_corr_298K= 0
        ring_flag = False
        
        if True in ring_inds: # add ring correction to final prediction
            RC0,RC2=get_rings(E,G,gens=2,return_R0=True) 

            if len(RC0.keys()) > 0:
                print("Identify rings! Add ring correction to final predictoon")
                ring_flag = True
                for key in RC0.keys():
                    depth0_ring=RC0[key]
                    depth2_ring=RC2[key] 

                    NonH_E = [ele for ele in E if ele is not 'H']
                    ring_NonH_E = [ele for ele in depth2_ring["elements"] if ele is not 'H']

                    if len(depth2_ring["ringsides"]) == 0:
                        print("\n{} can not use TGIT to calculate, the result comes from G4 result".format(i.split('/')[-1]))

                    if float(depth0_ring["hash_index"]) in ring_dict["HF_0"].keys():
                        # predict difference at 0K
                        base_model.load_weights('ML-package/zero_model.h5')
                        diff_0K = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)
                        # predict difference at 298K
                        base_model.load_weights('ML-package/roomT_model.h5')
                        diff_298= getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)
                        RC_0K   = ring_dict["HF_0"][float(depth0_ring["hash_index"])]  + diff_0K
                        RC_298  = ring_dict["HF_298"][float(depth0_ring["hash_index"])]+ diff_298

                        ring_corr_0K  +=RC_0K 
                        ring_corr_298K+=RC_298
                
                        print("Add ring correction {}: {} kcal/mole into final prediction (based on depth=0 ring {})".format(depth2_ring["hash_index"],RC_298,depth0_ring["hash_index"]))
                    
                    else:
                        print("Information of ring {} is missing, the final prediction might be not accurate, please update ring_correction database first".format(depth0_ring["hash_index"]))

            else: 
                print("Identify rings, but the heavy atom number in the ring is greater than 12, don't need add ring correction")
        
        # remove terminal atoms                                                                                                           
        B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

        # Apply pedley's constrain 1                                                                                                      
        H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
        P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]

        group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]
        Unknown = [j for j in group_types if j not in FF_dict["HF_0"].keys()]

        # indentify whether this is a minimal structure or not
        min_types = [ j for j in group_types if minimal_structure(j,G,E,gens=2) is True ]

        if ring_flag is False: # deal with linear strcuture
            if len(Unknown) == 0: 
                print("\n"+"="*120)
                print("="*120)
                print("\nNo more information is needed, begin to calculate enthalpy of fomation of {}".format(i.split('/')[-1]))
                Hf_0,Hf_298 = calculate_CAV(E,G,name,FF_dict,Hf_atom_0k,Atom_G4,H298km0k,periodic,ring_corr_0K,ring_corr_298K)
                
                # predict liquid phase enthalpy of formation (at room temperature)
                P,H_vap = calculate_PandHvap(smiles,T=298)
                print("Prediction of Hf_298 for {} in liquid pahse is {} kJ/mol\n\n".format(name, Hf_298-H_vap))
                TCITresult[smiles]=Hf_298-H_vap

                if len(group_types) < 2: 
                    print("\n{} is too small for TCIT prediction, the result comes directly from a G4 calculation".format(i.split('/')[-1]))

            else:
                print("\n"+"="*120)
                print("Unknown CAVs are required for this compound, skipping...\n") 
                print("\n"+"="*120)

        else: # deal with ring structure
            if len(Unknown) == 0:
                print("\n"+"="*120)
                print("="*120)
                print("\nNo more information is needed, begin to calculate enthalpy of fomation of {}".format(i.split('/')[-1]))
                Hf_0,Hf_298 = calculate_CAV(E,G,name,FF_dict,Hf_atom_0k,Atom_G4,H298km0k,periodic,ring_corr_0K,ring_corr_298K)
                P,H_vap = calculate_PandHvap(smiles,T=298)
                print("Prediction of Hf_298 for {} in liquid pahse is {} kJ/mol\n\n".format(name, Hf_298-H_vap))
                TCITresult[smiles]=Hf_298-H_vap

            else:
                print("\n"+"="*120)
                print("Unknown CAVs are required for this compound, skipping...\n") 
                print("\n"+"="*120)
                
    with open("TCIT_result.txt","w") as f:
        for smiles in sorted(TCITresult.keys()):
            f.write("{:<60s} {:<10.2f}\n".format(smiles,TCITresult[smiles]))

    quit()

# function to calculate Hf based on given TCIT database
def calculate_CAV(E,G,name,FF_dict,Hf_atom_0k,Atom_G4,H298km0k,periodic,ring_corr_0K,ring_corr_298K):

    kj2kcal = 0.239006
    H2kcal =  627.509

    adj_mat = Table_generator(E,G)
    atom_types = id_types(E,adj_mat,2)
    atom_types = [atom_type.replace('R','') for atom_type in atom_types]

    # remove terminal atoms                                                                                                           
    B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

    # Apply pedley's constrain 1                                                                                                      
    H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
    P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and\
                len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]
    group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]

    Hf_target_0  = ring_corr_0K
    Hf_target_298= ring_corr_298K

    for j in group_types:
        Hf_target_0 += FF_dict["HF_0"][j]
        Hf_target_298 += FF_dict["HF_298"][j]

    for j in P1_inds:
        NH = len([ind_j for ind_j,adj_j in enumerate(adj_mat[j,:]) if adj_j == 1 and ind_j in H_inds])
        if NH == 3:
            Hf_target_0 += FF_dict["HF_0"]["[6[6[1][1][1]][1][1][1]]"]
            Hf_target_298 += FF_dict["HF_298"]["[6[6[1][1][1]][1][1][1]]"]
        elif NH == 2:
            Hf_target_0 += FF_dict["HF_0"]["[6[6[1][1]][1][1]]"]
            Hf_target_298 += FF_dict["HF_298"]["[6[6[1][1]][1][1]]"]
        elif NH == 1:
            Hf_target_0 += FF_dict["HF_0"]["[6[6[1]][1]]"]
            Hf_target_298 += FF_dict["HF_298"]["[6[6[1]][1]]"]
        else:
            print("Error, no such NH = {} in Constrain 1".format(NH))
            print("{} shouldn't appear here".format([atom_types[Pind] for Pind in P1_inds]))
            quit()
    
    print("Prediction are made based on such group types:")
    for j in group_types:
        print("{:30s}: {}".format(j,FF_dict["HF_298"][j]))
    print("Prediction of Hf_0 for {} is {} kJ/mol".format(name, Hf_target_0/kj2kcal))
    print("Prediction of Hf_298 for {} is {} kJ/mol\n\n".format(name, Hf_target_298/kj2kcal))

    return Hf_target_0/kj2kcal,Hf_target_298/kj2kcal

# Function that take smile string and return element and geometry
def parse_smiles(smiles):
    # load in rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # construct rdkir object
    m = Chem.MolFromSmiles(smiles)
    m2= Chem.AddHs(m)
    AllChem.EmbedMolecule(m2)
    
    # parse mol file and obtain E & G
    lines = Chem.MolToMolBlock(m2).split('\n')
    E = []
    G = []
    for line in lines:
        fields = line.split()
        if len(fields) > 5 and fields[0] != 'M' and fields[-1] != 'V2000':
            E  += [fields[3]]
            geo = [float(x) for x in fields[:3]]
            G  += [geo]

    G = np.array(G)
    return E,G

# load in TCIT CAV database
def parse_HF_database(db_files,FF_dict={"HF_0":{},"HF_298":{}}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0].lower() == "hf_gav": 
                FF_dict["HF_0"][fields[1]] = float(fields[2])
                FF_dict["HF_298"][fields[1]] = float(fields[3])
    return FF_dict

# load in depth=0 ring correction database
def parse_ringcorr(db_file,RC_dict={"HF_0":{},"HF_298":{}}):
    with open(db_file,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue 
            if len(fields) >= 3 : 
                RC_dict["HF_0"][float(fields[0])] = float(fields[1])
                RC_dict["HF_298"][float(fields[0])] = float(fields[2])
    return RC_dict

# load in G4 database
def parse_G4_database(db_files,FF_dict={"HF_0":{},"HF_298":{}}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue
            if len(fields)>=4:
                FF_dict["HF_0"][fields[0]] = float(fields[2])
                FF_dict["HF_298"][fields[0]] = float(fields[3])
    return FF_dict

# Description:   Checks if the supplied geometry corresponds to the minimal structure of the molecule
# 
# Inputs:        atomtype:      The taffi atomtype being checked
#                geo:           Geometry of the molecule
#                elements:      elements, indexed to the geometry 
#                adj_mat:       adj_mat, indexed to the geometry (optional)
#                atomtypes:     atomtypes, indexed to the geometry (optional)
#                gens:          number of generations for determining atomtypes (optional, only used if atomtypes are not supplied)
# 
# Outputs:       Boolean:       True if geo is the minimal structure for the atomtype, False if not.
def minimal_structure(atomtype,geo,elements,adj_mat=None,gens=2):

    # If required find the atomtypes for the geometry
    if adj_mat is None:
        if len(elements) != len(geo):
            print("ERROR in minimal_structure: While trying to automatically assign atomtypes, the elements argument must have dimensions equal to geo. Exiting...")
            quit()

        # Generate the adjacency matrix
        # NOTE: the units are converted back angstroms
        adj_mat = Table_generator(elements,geo)

        # Generate the atomtypes
        atom_types= id_types(elements,adj_mat,gens)
        atomtypes = [atom_type.replace('R','') for atom_type in atom_types] 

    # Check minimal conditions
    count = 0
    for count_i,i in enumerate(atomtypes):

        # If the current atomtype matches the atomtype being searched for then proceed with minimal geo check
        if i == atomtype:
            count += 1

            # Initialize lists for holding indices in the structure within "gens" bonds of the seed atom (count_i)
            keep_list = [count_i]
            new_list  = [count_i]
            
            # Carry out a "gens" bond deep search
            for j in range(gens):

                # Id atoms in the next generation
                tmp_new_list = []                
                for k in new_list:
                    tmp_new_list += [ count_m for count_m,m in enumerate(adj_mat[k]) if m == 1 and count_m not in keep_list ]

                # Update lists
                tmp_new_list = list(set(tmp_new_list))
                if len(tmp_new_list) > 0:
                    keep_list += tmp_new_list
                new_list = tmp_new_list
            
            # Check for the minimal condition
            keep_list = set(keep_list)
            if False in [ elements[j] == "H" for j in range(len(elements)) if j not in keep_list ]:
                minimal_flag = False
            else:
                minimal_flag = True
        
    return minimal_flag

#getModel and getPrediction are the two main functions. Build the model and load the parameters,
def getModel():
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

    return model
    
def getPrediction(smiles,R0_smiles,model):
    X_eval = (smiles,R0_smiles)
    processed_eval = preprocess.neuralProcess(X_eval)
    predictions = np.squeeze(model.predict_on_batch(x=processed_eval))
    return predictions

# Logger object redirects standard output to a file.
class Logger(object):
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open("{}.log".format(filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

if __name__ == "__main__":
    main(sys.argv[1:])

