from scipy.stats import pearsonr, spearmanr

simtheobert=[ 0.7397847752417287,0.8490720142920812,0.5752032314028058,0.6760935889823096,0.8965948537776345,0.43027432858943937,0.7204429116742365,0.7680894924748328,0.6557986704926742, 0.9359762555076963,0.5298174836418845,0.1850460090420463,0.9307105541229248,0.7886220514774323,0.751273113489151,0.8999340659693668,0.8706924946684587, 0.12233721017837525,0.7065569992576327,0.6793529114552906,0.33316148072481155,0.8726344406604767,0.7365939761030262,0.6571067944169044,0.814725108685032,0.5958791375160217,0.333962773734873,0.91340012135713,0.607046643892924,0.3789167155822118,0.9002644910531885,0.7945306607194849,0.46147525661131916,0.9567035585641861,0.6443207349096026,0.33266868645494635,0.9030077051032673,0.826835644872565,0.2593914568424225,0.936783458505358,0.5593209796481662,0.23743645598491034,0.9550576372580090,0.3878824961812873,0.4269414782524100,0.962462248802185,0.4911256631215413,0.4582169413566589,0.9817622860272726,0.927590529123942,0.7649328967799311, 0.9195111186608024,0.5453143691023191,0.48650278828360816,0.990923456076918,0.5266686603426933,0.09750449180603027,0.9589551311952097,0.8496543616056442,0.482453852891922]
simchuyengia=[0.8333333333333334, 0.8333333333333334, 0.0, 0.5, 0.8333333333333334, 0.0, 1.0, 0.5, 0.0, 0.8333333333333334, 0.5, 0.0, 1.0, 0.3333333333333333, 0.0, 0.6666666666666666, 0.5, 0.0, 0.8333333333333334, 0.3333333333333333, 0.0, 0.8333333333333334, 0.3333333333333333, 0.0, 0.8333333333333334, 0.6666666666666666, 0.0, 1.0, 0.6666666666666666, 0.0, 0.6666666666666666, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.3333333333333333, 0.16666666666666666, 0.8333333333333334, 0.5, 0.0, 0.8333333333333334, 0.3333333333333333, 0.0, 1.0, 0.5, 0.0, 0.8333333333333334, 0.5, 0.0, 0.8333333333333334, 0.5, 0.16666666666666666, 1.0, 0.6666666666666666, 0.0, 0.8333333333333334, 0.5, 0.0]

pearson = pearsonr(simtheobert, simchuyengia)
spearman = spearmanr(simtheobert, simchuyengia)
print(simtheobert)
print(simchuyengia)
print("Tương Quan Theo Phương Pháp PearSon : ", pearson[0])
print("Tương Quan Theo Phương Pháp SpearMan : ", spearman[0])