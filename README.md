# CRISPER-CAS9

For pretraining run cnn_aenc_genome_pt.py which uses autoencoder
After that to load the pretrained weight run cnn_aenc_genome_tr_ld.py

# Groups:

- Group 1 (47124): All sgRNAs (might be the same as before)
- Group 2 (46234): All except manual exclusions (ME, described in our initial submission)
- Group 3 (45503): All except ME and End of Chromosome sgRNAs (as described in initial submission)
- Group 4 (43716): All except ME, EOC, and those containing polyT motif 
- Group 5 (43651): All except ME, EOC, polyT, and lowest 500 abundance sgRNAs in untransformed library
- Group 6 (43181): All except ME, EOC, polyT, and lowest 1000
- Group 7 (42238): All except ME, EOC, polyT, and lowest 2000  
- Group 8 (40337): All except ME, EOC, polyT, and lowest 4000  
- Group 9 (36566): All except ME, EOC, polyT, and lowest 8000 
- Group 10 (29041): All except ME, EOC, polyT, and lowest 1600

# Classification of Guides:
- Excellent: min cutting score -10.003659369999999 and max cutting score  -5.0002097399999998. Less the cutting score more                  excellent the guide is
- Moderate: min cutting score -4.999916121 and max cutting score -1.000090874 
- Poor: min cutting score -0.999961177 and max cutting score 1.999324657
for binary classification, we have labeled 'Excellent' as 1 ; 'Moderate' and 'Poor' as 0. So cutting score<= -5.00 is positive label threshold>-5.00 is negative label. So threshold is -5.00

# Experimental Result on Cpf1 Data

|  | Spearman R | Pearson R |  
| --- | --- | --- |
| `CNN (w/o pretrain)` | 0.582 |  0.592 |
| `CAE (pre-train + w/o fine-tune)` | 0.527 | 0.537 |
| `CAE (pre-train + fine-tune)` | 0.655 |  0.664 |
| `Re-trained DeepCpf1` | 0.614 |  0.628 |

# Experimental Result on Cas9 Data

|  | Spearman R | Pearson R |  
| --- | --- | --- |
| `CNN (w/o pretrain)` | 0.413 |  0.491 |
| `CAE (pre-train + w/o fine-tune)` | 0.335 | 0.375 |
| `CAE (pre-train + fine-tune)` | 0.416 | 0.492  |
| `CAE (pre-train + fine-tune + w/o NU)` | 0.262 | 0.312  |

# Ablation analysis on Cpf1 Data
filename: Seq_cpf1_autoenc_final-Copy1
## Fine-tuned
Pool + flatten = 0.637

Pool + flatten + 1 FC(dropout) = 0.649

Pool + flatten + 2 FC(dropout) = 0.653

Pool + flatten + 3 FC = 0.653

## Not Fine-tuned
Pool + flatten = 0.517

Pool + flatten + 1 FC(dropout) = 0.527

Pool + flatten + 2 FC(dropout) = 0.523

Pool + flatten + 3 FC = 0.515

netstat -tnlp | grep :88
