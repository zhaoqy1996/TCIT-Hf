## TCIT

**TCIT**, the short of Taffi component increment theory, is a powerful tool to predict thermochemistry properties, like enthalpy of formation.

This script implemented TCIT which performs on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. (The paper can be found [here](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092?casa_token=J-tbN5mxhiAAAAAA:KaJcTVzRs0t3M3kkwdSpvg5LQkAD6iSyzpUEjzNg_MmwqNGdmah57E_NSlwBlJ81p8ROOqibqUN8NEs5).) Further ring correction is added distributed with the paper "Ring correction XXX"' by Zhao, Iovanac and Savoie. 

The script operates on .xyz files, prints out the taffi components and corresponding CAVs that are used for each prediction, and returns the 0K and 298K enthalpy of formation. 

## Software requirement
1. openbabel 2.4.1 
2. python 3.5 or higher
3. tensorflow 2.X
4. numpy 1.17 or higher

## Usage
1. Put xyz files of the compounds with research interest in one folder (default: input_xyz)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
