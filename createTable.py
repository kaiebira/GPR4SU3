import numpy as np
import os
import re
import latqcdtools.physics.polyakovTools as pT

from latqcdtools.base.readWrite import writeTable
from latqcdtools.legacy import jackknife


Nt = 6
Ns = 30


# Parsing through outfiles
def parse_files(directory):
    results = {}
    file_pattern = re.compile(r'^Nt6_Ns30_b\d+\.?\d*\.out$')
    polyakov_pattern = re.compile(r'Polyakov loop = \(\s*(-?\d+\.?\d*e?[+-]?\d*),\s*(-?\d+\.?\d*e?[+-]?\d*)\s*\)')
    beta_value_pattern = re.compile(
        r'# PARAM :: beta = (\d+\.?\d*)')  # Pattern to capture beta value from the file content

    # Gather all files
    files = [f for f in os.listdir(directory) if file_pattern.match(f)]
    print(files)

    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            beta_value = None
            for line in file:
                if 'PARAM :: beta =' in line:
                    beta_match = beta_value_pattern.search(line)
                    if beta_match:
                        beta_value = beta_match.group(1)
                        if beta_value not in results:
                            results[beta_value] = []
                        continue  # Continue to read further for Polyakov loop values

                if beta_value and 'Polyakov loop' in line:
                    numbers = polyakov_pattern.search(line)
                    if numbers:
                        real_part, imag_part = map(float, numbers.groups())
                        results[beta_value].append((real_part, imag_part))

    # Convert lists to numpy arrays and prepare sorted beta values list
    beta_values_sorted = sorted(results.keys(), key=float)
    for beta in beta_values_sorted:
        results[beta] = np.array(results[beta], dtype=[('real', 'f4'), ('imag', 'f4')])

    return results, beta_values_sorted


polReIm, betas = parse_files('.')

# calculate <|P|> Err_<|P|> X Err_X
absPloops = []
Errs_absPloop = []
Xs = []
Errs_X = []
Nconfs = []

pTools = pT.polyakovTools(Nsigma=Ns, Ntau=Nt)

for beta in betas:
    Nconfs.append(len(polReIm[beta]))
    real_parts = polReIm[beta]['real']  # Extract real parts
    imag_parts = polReIm[beta]['imag']  # Extract imaginary parts
    combined_re_im = (real_parts, imag_parts)

    absPTemp, errPTemp = jackknife(pTools.absPLoop, combined_re_im)
    absPloops.append(absPTemp)
    Errs_absPloop.append(errPTemp)

    XTemp, errXTemp = jackknife(pTools.Suscept, combined_re_im)
    Xs.append(XTemp)
    Errs_X.append(errXTemp)

# Convert to numpy arrays
absPloops = np.array(absPloops)
Errs_absPloop = np.array(Errs_absPloop)
Xs = np.array(Xs)
Errs_X = np.array(Errs_X)
Nconfs = np.array(Nconfs)


writeTable('Nt' + str(Nt) + '_Ns' + str(Ns) + '.txt', betas, absPloops, Errs_absPloop, Xs, Errs_X, Nconfs,
           header=['Beta', '<|P|>', 'Err <|P|>', 'X', 'Err X', 'NConf'])
