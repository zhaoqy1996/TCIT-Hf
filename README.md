(Notes: this package is just used for reproducing [paper1](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092?casa_token=J-tbN5mxhiAAAAAA:KaJcTVzRs0t3M3kkwdSpvg5LQkAD6iSyzpUEjzNg_MmwqNGdmah57E_NSlwBlJ81p8ROOqibqUN8NEs5) and [paper2](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092?casa_token=J-tbN5mxhiAAAAAA:KaJcTVzRs0t3M3kkwdSpvg5LQkAD6iSyzpUEjzNg_MmwqNGdmah57E_NSlwBlJ81p8ROOqibqUN8NEs5) results. An up-to-date version can be found in TCIT-thermo folder in https://github.com/zhaoqy1996/TCIT_thermo. We'll continue update new component additivity values (CAVs) and ring corrections (RCs) and also add TCIT predictions for other properties and radicals species in TCIT_thermo project in a near future.

## TCIT

**TCIT**, the short of Taffi component increment theory, is a powerful tool to predict thermochemistry properties, like enthalpy of formation.

This script implemented TCIT which performs on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "[A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092)" by Zhao and Savoie. Further ring correction is added distributed with the paper "[Transferable Ring Corrections for Predicting Enthalpy of Formation of Cyclic Compounds](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00367)" by Zhao, Iovanac and Savoie.

The script operates on either a folder of xyz files or a list of smiles strings, prints out the Taffi components and corresponding CAVs that are used for each prediction, and returns the 0K and 298K enthalpy of formation. 

## Software requirement
1. openbabel 2.4.1 or higher
2. anaconda

## Set up an environment if needed
* conda create -n TCIT -c conda-forge python=3.7 rdkit
* source activate TCIT
* pip install alfabet

## Usage
If your input type a xyz file:

1. Put xyz files of the compounds with research interest in one folder (default: inputxyz)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT" and the program will take all xyz files in "input_xyz" and return a prediction result in result.log

If your input type is smiles string:

1. Make a list of smiles string (default: input.txt)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT -t smiles" and the program will take all smiles string in input.txt and return a prediction result in result.log

## Notes
1. Make sure the bin folder of openbabel is added in the environment setting, or 'alias obabel=' to that bin folder 
