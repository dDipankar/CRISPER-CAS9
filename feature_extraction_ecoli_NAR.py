#import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import *
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import GC
import re
import sys

def one_hot_coding_single_nu(input_df):
    SEQ = zeros((len(input_df), 20*4), dtype=int)
    for l in range(len(input_df)):
        seq = input_df.iat[l]
        seq = seq[4:24]
        for i in range(20):
            if seq[i] in "Aa":
                SEQ[l, 4*i+0] = 1
            elif seq[i] in "Cc":
                SEQ[l, 4*i +1] = 1
            elif seq[i] in "Gg":
                SEQ[l, 4*i + 2] = 1
            elif seq[i] in "Tt":
                SEQ[l, 4*i + 3] = 1
    return(SEQ)

def one_hot_coding_di_nu(input_df):
    SEQ = zeros((len(input_df), 19*16), dtype=int)
    for l in range(len(input_df)):
        seq = input_df.iat[l]
        seq = seq[4:24]
        for i in range(19):
            if seq[i:i+2] == "AA":
                SEQ[l, 16*i+0] = 1
            elif seq[i:i+2] == "AC":
                SEQ[l, 16*i +1] = 1
            elif seq[i:i+2] == "AT":
                SEQ[l, 16*i + 2] = 1
            elif seq[i:i+2] == "AG":
                SEQ[l, 16*i + 3] = 1
	    elif seq[i:i+2] == "CA":
                SEQ[l, 16*i +4] = 1
            elif seq[i:i+2] == "CC":
                SEQ[l, 16*i + 5] = 1
            elif seq[i:i+2] == "CT":
                SEQ[l, 16*i + 6] = 1
	    elif seq[i:i+2] == "CG":
                SEQ[l, 16*i + 7] = 1
	    elif seq[i:i+2] == "TA":
                SEQ[l, 16*i +8] = 1
            elif seq[i:i+2] == "TC":
                SEQ[l, 16*i + 9] = 1
            elif seq[i:i+2] == "TT":
                SEQ[l, 16*i + 10] = 1
	    elif seq[i:i+2] == "TG":
                SEQ[l, 16*i + 11] = 1
	    elif seq[i:i+2] == "GA":
                SEQ[l, 16*i +12] = 1
            elif seq[i:i+2] == "GC":
                SEQ[l, 16*i + 13] = 1
            elif seq[i:i+2] == "GT":
                SEQ[l, 16*i + 14] = 1
	    elif seq[i:i+2] == "GG":
                SEQ[l, 16*i + 15] = 1	
    return(SEQ)

def nucleotide_count(input_df):
	count_nu = zeros((len(input_df), 4), dtype=int)
	for l in range(len(input_df)):
        	seq = input_df.iat[l]
        	seq = seq[4:24]
		a = len(re.findall('(?=A)', seq))
		c = len(re.findall('(?=C)', seq))
		t = len(re.findall('(?=T)', seq))
		g = len(re.findall('(?=G)', seq))
		count_nu[l,0] = a
		count_nu[l,1] = c
		count_nu[l,2] = t
		count_nu[l,3] = g
	return count_nu

def dinucleotide_count(input_df):
	count_dnu = zeros((len(input_df), 16), dtype=int)
	for l in range(len(input_df)):
        	seq = input_df.iat[l]
        	seq = seq[4:24]
		aa = len(re.findall('(?=AA)', seq))
		ac = len(re.findall('(?=AC)', seq))
		at = len(re.findall('(?=AT)', seq))
		ag = len(re.findall('(?=AG)', seq))
		ca = len(re.findall('(?=CA)', seq))
		cc = len(re.findall('(?=CC)', seq))
		ct = len(re.findall('(?=CT)', seq))
		cg = len(re.findall('(?=CG)', seq))
		ta = len(re.findall('(?=TA)', seq))
		tc = len(re.findall('(?=TC)', seq))
		tt = len(re.findall('(?=TT)', seq))
		tg = len(re.findall('(?=TG)', seq))
		ga = len(re.findall('(?=GA)', seq))
		gc = len(re.findall('(?=GC)', seq))
		gt = len(re.findall('(?=GT)', seq))
		gg = len(re.findall('(?=GG)', seq))
		count_dnu[l,0] = aa
		count_dnu[l,1] = ac
		count_dnu[l,2] = at
		count_dnu[l,3] = ag
		count_dnu[l,4] = ca
		count_dnu[l,5] = cc
		count_dnu[l,6] = ct
		count_dnu[l,7] = cg
		count_dnu[l,8] = ta
		count_dnu[l,9] = tc
		count_dnu[l,10] = tt
		count_dnu[l,11] = tg
		count_dnu[l,12] = ga
		count_dnu[l,13] = gc
		count_dnu[l,14] = gt
		count_dnu[l,15] = gg
	return count_dnu

def GC_features(input_df):
	# G count + C count , if GC<=9, if GC>9
	# total 3 features
	count_gc = zeros((len(input_df), 1), dtype=int)
	for l in range(len(input_df)):
		seq = input_df.iat[l]
		# Protospacer sequence (20mer)
		seq = seq[4:24]
		count_gc[l,0] = float(GC(seq))/100
	return count_gc

def melting_tmp(input_df):
	mtmp = zeros((len(input_df), 1), dtype=float)	
	for l in range(len(input_df)):
		seq = input_df.iat[l]
		# Protospacer sequence (20mer)
		seq = seq[4:24]
		mtmp[l,0] =  mt.Tm_staluc(seq)
	return mtmp		


def melting_tmp_T5(input_df):
	mtmp = zeros((len(input_df), 1), dtype=float)	
	for l in range(len(input_df)):
		seq = input_df.iat[l]
		# Protospacer sequence (20mer)
		seq = seq[19:24]
		mtmp[l,0] =  mt.Tm_staluc(seq)
	return mtmp		
	
def melting_tmp_T8(input_df):
	mtmp = zeros((len(input_df), 1), dtype=float)	
	for l in range(len(input_df)):
		seq = input_df.iat[l]
		# Protospacer sequence (20mer)
		seq = seq[11:19]
		mtmp[l,0] =  mt.Tm_staluc(seq)
	return mtmp		
	
def melting_tmp_T7(input_df):
	mtmp = zeros((len(input_df), 1), dtype=float)	
	for l in range(len(input_df)):
		seq = input_df.iat[l]
		# Protospacer sequence (20mer)
		seq = seq[4:11]
		mtmp[l,0] =  mt.Tm_staluc(seq)
	return mtmp		

def one_hot_coding_di_nu_NGGN(input_df):
    SEQ = zeros((len(input_df), 1*16), dtype=int)
    for l in range(len(input_df)):
        seq = input_df.iat[l]
        N1 = seq[24]
        N2 = seq[27]
        NN = N1 + N2
        for i in range(1):
			if NN == 'AA':
				SEQ[l,0] = 1
			elif NN == 'AC':
				SEQ[l,1] =1
			elif NN == 'AT':
				SEQ[l,2] = 1
			elif NN == 'AG':
				SEQ[l,3] = 1
			elif NN == 'CA':
				SEQ[l,4] = 1
			elif NN == 'CC':
				SEQ[l,5] =1
			elif NN == 'CT':
				SEQ[l,6] =1	
			elif NN == 'CG':
				SEQ[l,7] =1
			elif NN == 'TA':
				SEQ[l,8] = 1
			elif NN == 'TC':
				SEQ[l,9] = 1
			elif NN == 'TT':
				SEQ[l,10] =1
			elif NN == 'TG':
				SEQ[l,11] = 1
			elif NN == 'GA':
				SEQ[l,12] = 1
			elif NN == 'GC':
				SEQ[l,13] = 1
			elif NN == 'GG':
				SEQ[l,14] =1 
			elif NN =='GT':
				SEQ[l,15] = 1
	return SEQ										
						
											
	
if __name__ == '__main__':
	
	input_df = pd.read_table('data_ecoli_mod.csv',sep=',', header = 'infer')
	input_data = input_df.ix[:, '30mers']
	#string = 'CATGCTGAACCAGTTGGCCATTAGGGCGGGTAGA'
	#print(input_data)
	ss = one_hot_coding_single_nu(input_data)
	#print(ss)
	sss = one_hot_coding_di_nu(input_data)
	#print(sss)
	combined = concatenate((ss, sss), axis = 1)
	#print(combined.shape)
	cc = nucleotide_count(input_data)
	#print(cc)
	combined = concatenate((combined, cc), axis = 1)
	dcc = dinucleotide_count(input_data)
	#print(dcc)
	combined = concatenate((combined, dcc), axis = 1)
	#print(combined.shape)
	gcc = GC_features(input_data)
	#print(gcc)
	combined = concatenate((combined, gcc), axis = 1)
	print(combined.shape)
	mtmp = melting_tmp(input_data)
	#print(mtmp)
	combined = concatenate((combined, mtmp), axis = 1)
	print(combined.shape)
	mtmp = melting_tmp_T5(input_data)
	combined = concatenate((combined, mtmp), axis = 1)
	print(combined.shape)
	mtmp = melting_tmp_T8(input_data)
	combined = concatenate((combined, mtmp), axis = 1)
	print(combined.shape)	
	mtmp = melting_tmp_T7(input_data)
	combined = concatenate((combined, mtmp), axis = 1)
	print(combined.shape)
	
	NGGN = one_hot_coding_di_nu_NGGN(input_data)
	combined = concatenate((combined, NGGN), axis = 1)
	print(combined.shape)	
	cutting_score = input_df.ix[:, 'sgRNA Score']
	cutting_score = cutting_score.values
	cutting_score = cutting_score.reshape(len(cutting_score),1)
	print(cutting_score.shape)
	combined = concatenate((combined,cutting_score), axis = 1)
	print(combined.shape)
	savetxt("features_ecoli_NAR.csv", combined, delimiter=",")
