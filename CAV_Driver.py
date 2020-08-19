# This is file is write to apply TCIT based on existing database
# Author: Qiyuan Zhao (Based on taffi written by Prif. Brett Savoie)
import sys,os,argparse
from scipy.spatial.distance import cdist
import numpy as np
from fnmatch import fnmatch

def main(argv):

    parser = argparse.ArgumentParser(description='This script predicts the enthalpy of formation for given target compounds based '+\
                                                 'on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component '+\
                                                 'Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. The script '+\
                                                 'operates on .xyz files, prints the components that it uses for each prediction, and will return the '+\
                                                 '0K and 298K enthalpy of formation. No ring corrections are supplied in this version '+\
                                                 'of the script, and compounds that require missing CAVs are skipped.' )

    #optional arguments                                                                                                                   
    parser.add_argument('-i', dest='input_folder', default='input_xyz',
                        help = 'The program loops over all of the .xyz files in this input folder and makes Hf predictions for them (default: "input_xyz"')

    parser.add_argument('-o', dest='outputname', default='results',
                        help = 'Controls the output file name for the results (default: "results")')

    parser.add_argument('-db', dest='dbfile', default='TAFFI_HF.db',
                        help = 'The directory path of TCIT CAVs database. (default: "TAFFI_Hf.db"')


    # parse configuration
    print("parsing calculation configuration...")
    args=parser.parse_args()    
    
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
    sys.stdout = Logger(args.outputname)

    # find all xyz files in given folder
    if args.input_folder is None:
        target_xyzs=[os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if (fnmatch(f,"*.xyz"))]

    else:
        target_xyzs=[os.path.join(dp, f) for dp, dn, filenames in os.walk(args.input_folder) for f in filenames if (fnmatch(f,"*.xyz"))]
    
    # loop for target xyz files
    for i in target_xyzs:
        print("\nWorking on {}...".format(i))
        E,G = xyz_parse(i)
        adj_mat = Table_generator(E,G)
        atom_types = id_types(E,adj_mat,2)
        
        # remove terminal atoms                                                                                                           
        B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

        # Apply pedley's constrain 1                                                                                                      
        H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
        P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]

        group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]
        Unknown = [j for j in group_types if j not in FF_dict["HF_0"].keys()]
        
        # indentify whether this is a minimal structure or not
        min_types = [ j for j in group_types if minimal_structure(j,G,E,adj_mat,atomtypes=atom_types,gens=2) is True ]

        # only when all of CAVs are known can this program give a prediction
        if len(Unknown) == 0:
            print("\n"+"="*120)
            print("="*120)
            print("\nAll required CAVs are known, beginning enthalpy of formation calculation for {}".format(i.split('/')[-1]))
            calculate_GAV(i,FF_dict,Hf_atom_0k,Atom_G4,H298km0k,periodic)
            if len(min_types) > 0:
                print("{} is too small for TCIT prediction, the result comes directly from a G4 calculation".format(i.split('/')[-1]))

        else:
            print("Unknown CAVs are required for this compound, skipping...\n")
            
# Description: function to calculate enthalpy of formation
def calculate_GAV(input_xyz,FF_dict,Hf_atom_0k,Atom_G4,H298km0k,periodic):

    kj2kcal = 0.239006
    H2kcal =  627.509

    E,G = xyz_parse(input_xyz)
    adj_mat = Table_generator(E,G)
    atom_types = id_types(E,adj_mat,2)
        
    # remove terminal atoms                                                                                                           
    B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

    # Apply pedley's constrain 1                                                                                                      
    H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
    P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and\
                len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]
    group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]

    Hf_target_0 = 0
    Hf_target_298 = 0

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
            print("Error, no such NH = {} in Constraint 1".format(NH))
            print("{} shouldn't appear here".format([atom_types[Pind] for Pind in P1_inds]))
            quit()
    
    print("\nPrediction is based on the following component types:\n")
    for j in set(group_types):
        print("\t{:30s}: {}".format(j,FF_dict["HF_298"][j]))
    print("\nPrediction of Hf_0 for {} is {} KJ/mol".format(input_xyz.split('/')[-1], Hf_target_0/kj2kcal))
    print("Prediction of Hf_298 for {} is {} KJ/mol".format(input_xyz.split('/')[-1], Hf_target_298/kj2kcal))
    return

# Description: parse TCIT CAVs database
#
# Inputs      dbfile path
# Output      a dictionary of CAV database, key is CAV name
#
def parse_HF_database(db_files,FF_dict={"HF_0":{},"HF_298":{}}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0].lower() == "hf_gav": 
                FF_dict["HF_0"][fields[1]] = float(fields[2])
                FF_dict["HF_298"][fields[1]] = float(fields[3])
    return FF_dict

# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types

# Generates the adjacency matrix based on UFF bond radii
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 array holding the geometry of the molecule
def Table_generator(Elements,Geometry):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print("ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print("Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if i == "H":  print("WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                if i == "C":  print("WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                if i == "Si": print("WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                if i == "F":  print("WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                if i == "Cl": print("WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                if i == "Br": print("WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                if i == "I":  print("WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                if i == "O":  print("WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                if i == "N":  print("WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                if i == "B":  print("WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                
        print("")

    return Adj_mat

# identifies the taffi atom types from an adjacency matrix/list (A) and element identify. 
def id_types(elements,A,gens=2,which_ind=None,avoid=[],geo=None,hybridizations=[]):

    # On first call initialize dictionaries
    if not hasattr(id_types, "mass_dict"):

        # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
        id_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
    masses = [ id_types.mass_dict[i] for i in elements ]
    atom_types = [ "["+taffi_type(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

    # Add ring atom designation for atom types that belong are intrinsic to rings 
    # (depdends on the value of gens)
    for count_i,i in enumerate(atom_types):
        if ring_atom(A,count_i,ring_size=(gens+2)) == True:
            atom_types[count_i] = "R" + atom_types[count_i]            

    return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ count_i for count_i,i in enumerate(adj_mat[ind]) if i == 1 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type(i,elements,adj_mat,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])

def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False


# hashing function for canonicalizing geometries on the basis of their adjacency matrices and elements
# ind  : index of the atom being hashed
# A    : adjacency matrix
# M    : masses of the atoms in the molecule
# gens : depth of the search used for the hash   
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

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
def minimal_structure(atomtype,geo,elements,adj_mat=None,atomtypes=None,gens=2):

    # If required find the atomtypes for the geometry
    if atomtypes is None or adj_mat is None:
        if len(elements) != len(geo):
            print("ERROR in minimal_structure: While trying to automatically assign atomtypes, the elements argument must have dimensions equal to geo. Exiting...")
            quit()

        # Generate the adjacency matrix
        # NOTE: the units are converted back angstroms
        adj_mat = adj.Table_generator(elements,geo)

        # Generate the atomtypes
        atom_types = id_types(elements,adj_mat,gens)
        atomtypes=[atom_type.replace('R','') for atom_type in atom_types] 

    # Check if this is a ring type, if not and if there are rings
    # in this geometry then it is not a minimal structure. 
    if "R" not in atomtype:
        if True in [ "R" in i for i in atomtypes ]:
            return False
        
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

# Logger object redirects standard output to a file.
class Logger(object):
    def __init__(self,outputname):
        self.terminal = sys.stdout
        self.log = open("{}.out".format(outputname), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass
        
if __name__ == "__main__":
    main(sys.argv[1:])
