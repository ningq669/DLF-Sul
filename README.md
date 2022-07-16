# DLF-Sul
Python Requirement:
Python >= 3.6

Package Requirement:
numpy >= 1.19.5
pytorch >= 1.10.0
sklearn >= 0.0
tqdm >= 4.62.3
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Steps:

1. Put the predicted amino acid sequence into text.csv.

2. Run DLF sul py.

3. From probability_ of_ Sul.txt to view the predicted results.

The following is the explanation and matters needing attention for all text files:
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test.csv:
1. The input format includes three columns, the first column is protein IDs, the second column is site positions, 
and the third column is the sequence of s-sulfation site or non-s-sulfation site.
2. Please ensure: (1) each single S-sulfinylation sequence or non-S-sulfinylation sequence only occupy one line in the txt file.
                  (2) Sequence string is in upper case.
4. The predicted result could be unauthentic if the number of sequence is not equal to 31.
5. It could take a long time if sequence is too long.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
probability_of_Sul.txt
1.The output format includes three columns: the first column is protein IDs, the second column is site positions, and the third column is probability values.
2.This file is one of the output files after predicting, that probability of The cysteine in the middle of the sequence is S-sulfinylation site" for 
each sequence in test.csv will be written in the file.
3. Each probability only occupy one line in the csv file, and it corresponds to the test.csv.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AAI.txt
1.This is about the physicochemical characteristics of amino acids.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BE.txt
1.Here are the features about the amino acid position.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BLOSUM62.txt
1.This is a feature about the substitution of amino acid positions by related amino acids.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test.csv example:
A0A075B759 62 RYKGSCFHRIIPGFMCQGGDFTRPNGTGDKS
A0A0A0MY14 27 ARVTKVLGRTGSQGQCTQVRVEFMDDTSRSI
A0A0B4J2A2 62 RYKGSCFHRIIPGFMCQGGDFTRPNGTGDKS
A0A0G2JVI3 523 EKEEFEHQQKELEKVCNPIITKLYQSAGGMP

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
probability_of_Sul.txt example:
A0A075B767 62 0.999865
A0A0B4J2A2 62 0.999810
A0A0G2JVI3 523 0.999963
A0A0G2JWU1 222 0.999418
