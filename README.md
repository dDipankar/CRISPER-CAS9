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
- for cpf1, if cutting score<-2.2 then good guide otherwise bad guide

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
| `CAE (pre-train + fine-tune + w/o NU)` | 0.314 | 0.373  |

# Ablation analysis on Cpf1 Data
filename: Seq_cpf1_autoenc_final-Copy1
## Fine-tuned
Pool + flatten = 0.637(Sp), 0.641(Pr)

Pool + flatten + 1 FC(dropout) = 0.649(Sp), 0.658(Pr)

Pool + flatten + 2 FC(dropout) = 0.653(Sp), 0.660(Pr)

Pool + flatten + 3 FC = 0.653

## Not Fine-tuned
Pool + flatten = 0.521(Sp), 0.532(Pr)

Pool + flatten + 1 FC(dropout) = 0.527(Sp), 0.533(Pr)

Pool + flatten + 2 FC(dropout) = 0.505(Sp), 0.517(Pr)

Pool + flatten + 3 FC = 0.501(sp), 0.514(pr)

netstat -tnlp | grep :88

# Ablation analysis on Cas9 Data
filename: cas9_training_28bp
## Fine-tuned
Pool + flatten = 0.347(Sp), 0.409(Pr)

Pool + flatten + 1 FC(dropout) = 0.364(Sp), 0.424(Pr)

Pool + flatten + 2 FC(dropout) = 0.357(Sp), 0.414(Pr)

Pool + flatten + 3 FC =  0.357(Sp), 0.414(Pr)

Pool + flatten + 4 FC (mult)=  0.428(Sp), 0.503(Pr)

## Not Fine-tuned
Pool + flatten = 0.316(Sp), 0.353(Pr)

Pool + flatten + 1 FC(dropout) = 0.273(Sp), 0.310(Pr)

Pool + flatten + 2 FC(dropout) = 0.261(Sp), 0.291(Pr)

Pool + flatten + 3 FC =  0.269(Sp), 0.305(Pr)

Pool + flatten + 4 FC (mult)=  0.345(Sp), 0.388(Pr)

# DeepCRISPR
       
|  | Spearman R | Pearson R |  
| --- | --- | --- |
| `CAE on cpf1 data` | 0.655 |  0.664 |
| `CAE on cas9 data` | 0.41 |  0.490 |
| `Tested on cas9 data` | 0.200 |  0.211 |
| `Retrained on cas9 data` | 0.261 |  0.282 |
| `Tested on cpf1 data` | 0.047 |  0.040 |
| `Retrained on cpf1 data` | 0.277 |  0.277 |

# DeepCPf1
       
|  | Spearman R | Pearson R |  
| --- | --- | --- |
| `CAE on cpf1 data` | 0.655 |  0.664 |
| `CAE on cas9 data` | 0.41 |  0.490 |
| `Tested on CPf1 data` | 0.250 |  0.243 |
| `Retrained on cpf1 data` | 0.614 |  0.628 |
| `Tested on cas9 data` | 0.200 |  0.218 |
| `Retrained on cas9 data` | 0.370 |  0.450 |

* Model for cpf1: Cpf1_32nt_seq_wtt.h5
* Note: training file for cas9: cas9_training_28nt.ipynb(28 bp guide),cas9_training_40bp (40 bp), cas9_training_seq_28nt.ipynb(28 bp seq)
cas9_loading_28nt.ipynb(for loading)
* for cpf1: Seq_cpf1_autoenc_final_training.ipynb
* Note: loading model file for cpf1: Seq_cpf1_autoenc_load_final.ipynb
* flask: https://github.com/krishnaik06/Deployment-flask
* Grep command: grep -i -r 'Cpf1_CS_wins_nu_new.csv' *
