# CRISPER-CAS9

For pretraining run cnn_aenc_genome_pt.py which uses autoencoder
After that to load the pretrained weight run cnn_aenc_genome_tr_ld.py

Group 1 (47124): All sgRNAs (might be the same as before)
Group 2 (46234): All except manual exclusions (ME, described in our initial submission)
Group 3 (45503): All except ME and End of Chromosome sgRNAs (as described in initial submission)
Group 4 (43716): All except ME, EOC, and those containing polyT motif 
Group 5 (43651): All except ME, EOC, polyT, and lowest 500 abundance sgRNAs in untransformed library
Group 6 (43181): All except ME, EOC, polyT, and lowest 1000
Group 7 (42238): All except ME, EOC, polyT, and lowest 2000  
Group 8 (40337): All except ME, EOC, polyT, and lowest 4000  
Group 9 (36566): All except ME, EOC, polyT, and lowest 8000 
Group 10 (29041): All except ME, EOC, polyT, and lowest 1600
