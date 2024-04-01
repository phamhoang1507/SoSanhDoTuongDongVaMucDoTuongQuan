from matplotlib import pyplot as pp
import  numpy as np
danhgiasim=[0.36134502, 0.97691363, 0.7782171, 0.0026801974, 0.5724145, 0.8060971, 0.42013377, 0.1644185, 0.8669199, 0.5649719, 0.566566, 0.3437384, 0.8762062, 0.65606534, 0.798301, 0.123349205, 0.9350787, 0.8599137, 0.49031332, 0.91453254, 0.9129046, -0.11639181, 0.8379861, 0.19822744, 0.88243717, 0.22350022, 0.0483137, 0.930925, 0.25061494, 0.7943748, 0.80853057, 0.24818613, 0.23024243, 0.7428888, 0.4479093, 0.87279975, 0.16428807, 0.097130924, 0.9056369, 0.6833609, 0.092074215, 0.9529132, 0.99999994, 0.8143792, 0.21033624, 0.91413707, 0.18795149, 0.107804015, 0.22138691, 0.9212607, 0.96031356, 1.0, 0.2683732, 0.47269142, 0.9865465, 0.93689036, 0.4521916, 0.7583457, 0.9342556, 0.4233382, 0.94190896, 0.14550059, 0.1514945, 0.30772814, 0.13418274, 0.19974, 0.10444739, 0.21392348, 0.30604845, 0.20387071, 0.17287919, 0.12885016, -0.026748132, 0.29692626, 0.12713358, 0.08061991, 0.007994041, 0.045957867, 0.24508926, 0.8517155, 0.24545138, 0.7786953, 0.29450428, 0.24367559, 0.3375784, 0.93136823, 0.99999994, 0.41534847, 0.33431268, 0.3025333, 0.48552796, 0.8772117, 0.9559585, 0.9650632, 0.075881295, 0.10124323, 0.8915459, 0.9148405, 0.32602853, -0.08619696, 0.84463584, 0.70848393, 0.86000764, 0.2865041, 0.57436943, 0.697997, 0.6168547, 0.36889884, 0.753806, 0.7970978, 0.70868033, 0.9136084, 0.70950294, 0.7323642, 0.56096697, 0.76536816, 0.7981273, 0.7608258, 0.65746295, 0.47977695, 0.56909966, 0.24856955, 0.49879086, 0.5934603, 0.9139949, 0.9135988, 0.6846551, 0.6172992, 0.5661779, 0.2833612, 0.5665242, 0.4856345, 0.57234704, 0.7952125, 0.2876134, 0.6269859, 0.36889884, 0.40725502, 0.112221174, 0.8173777, 0.5692357, 0.44368505, 0.4985658, 0.26997808, 0.5956047, 0.84457594, 0.7923958, 0.7352562, 0.3542905, 0.15184045, 0.73173594, 0.6520002, 0.5599977, 0.11360915, 0.6144506, 0.52878606, 0.4089105, 0.572324, 0.5037043, 0.24402788, 0.34281665, 0.3949472, 0.41571215, 0.544255, 0.25425184, 0.05353243, 0.61026925, 0.115389355, 0.20925444, 0.21405596, 0.05337552, 0.2865041, 0.35155693, 0.49995804, 0.3012544, -0.03070338, 0.113624655, 0.68106306, 0.33344266, 0.87926865, 0.8792847, 0.6454904, 0.9497789, 0.87923616, 0.3915543, 0.7548917, 0.66169506, 0.7517973, 0.7843595, 0.6530906, 0.8337709, 0.3623583, 0.53652465, 0.28096035, 0.084676355, 0.047550127, 0.04605304, 0.20807588, -0.056369286, 0.6847186, 0.82563484, 0.44090778, 0.54280424, 0.25802433, 0.48792756, 0.06871571, 0.41710472, -0.0041386113, -0.061838306, 0.7246891, 0.59034073, 0.23164997, 0.22253917, 0.3330865, 0.11343617, 0.3085792, 0.29071134, 0.04023148, 0.045936584, 0.425679, 0.47250557, 0.14930207, 0.56739134, 0.42425328, 0.71338665, 0.18598193, 0.17217565, 0.2838089, -0.037590723, 0.20019278, 0.16179138, -0.034203067, 0.023503765, 0.09710126, 0.11384697, -0.017515454, 0.08845502, -0.012445547, 0.43995684, 0.68161094, 0.33000582, 0.392394, 0.32504863, 0.66321576, 0.284644, 0.50872076, 0.3176988, -0.008755751, 0.1881102, -0.010161057, -0.06754338, 0.37468007, -0.034203067, 0.2579666, 0.114885546, 0.13784026, 0.15977922, 0.10695804, 0.3654846, 0.5398227, 0.72468394, 0.22767016, 0.44900393, 0.3547172, 0.37039888, 0.0014097095, 0.2429291, 0.05340469, 0.0014097095, 0.20171422, 0.7098036, 0.49823532, 0.44900393, 0.375629, 0.37039888, -0.017515454, 0.2429291, 0.13956147, 0.11616954, 0.7246891, 0.59034073, 0.23405877, 0.56739134, 0.09710126, 0.11384697, -0.038769893, 0.36462831, 0.2838089, -0.037590723, 0.65361214, 0.54938436, 0.26040566, 0.2439576, 0.21167003, 0.07945899, -0.038769893, 0.36462831, 0.01630678, -0.03884157, 1.0, 0.35969257, 0.6842312, 0.29513505, 0.4092834, 0.3330865, 0.20391515, 0.3085792, 0.29071134, 0.04023148, 0.045936584, 0.83288366, 0.52093726, 0.3702811, 0.56739134, 0.09710126, 0.11384697, 0.13095528, 0.09296964, 0.009034209, -0.037590723, 0.45584092, 0.5398469, 0.58935297, 0.38324505, 0.23404439, 0.54671407, 0.17410585, 0.1883443, -0.026473675, 0.035597175, 0.6753937, 0.56147456, 0.7688447, 0.30196962, 0.27048585, 0.15867928, 0.024873996, 0.38258085, 0.103456624, -0.025382213, 0.838073, 0.7996786, 0.13378131, 0.49290833, 0.13439289, 0.07880342, 0.11141479, 0.28162694, 0.26510614, -0.053290643, 0.71573275, 0.4759018, 0.3105751, 0.39863336, 0.36431897, 0.5995548, 0.0014097095, 0.04194212, -0.025382213, 0.015367806, 0.6530906, 0.8337709, 0.3623583, 0.53652465, 0.28096035, 0.18838586, 0.047550127, 0.09720268, 0.20807588, -0.056369286, 0.6597943, 0.26868132, 0.5570213, 0.3689355, 0.3190836, 0.31933516, 0.27645093, 0.09753797, 0.1189733, 0.061607935, 0.20019278, 0.31522843, 0.23252097, 0.023503765, 0.09710126, 0.11384697, -0.017515454, 0.08845502, -0.012445547, 0.43995684, 0.425679, 0.5398469, 0.27828854, 0.56739134, 0.42425328, 0.71338665, 0.18598193, 0.17217565, 0.2838089, -0.037590723, 0.36568326, 0.65147024, 0.2979663, 0.26580706, 0.115034565, 0.34713322, 0.18789548, 0.14713423, 0.0833704, 0.12998205, 0.64019156, 0.5958382, 0.3074816, 0.51014584, 0.17423114, 0.51927835, 0.5255238, 0.09717448, 0.6659145, -0.028526127, 0.11584082, 0.4271413, 0.4986722, 0.30021414, 0.09770483, 0.37769836, 0.46425098, 0.48645186, 0.55102277, 0.73493916, 0.26321238, -0.01830262, -0.01947774, 0.27810776, 0.7569791, 0.5516987, 0.35490233, 0.048223436, 0.16291927, 0.55104417, 0.019071713, 0.34971964, 0.41422677, 0.614409, 0.24028742, 0.67490745, 0.35586998, 0.46249384, 0.60346437, 0.44987872, 0.34619433, 0.4241287, 0.3756426, 0.5691221, 0.35947827, 0.08532668, 0.4694684, 0.6595968, 0.66855323, 0.78486943, 0.2058526, 0.2091307, 0.60698277, 0.31740946, 0.1358731, 0.29785454, 0.43218577, 0.6744267, 0.7283129, 0.5049832, 0.40991616, 0.24227707, 0.5959675, 0.6742599, 0.5068691, 0.7480922, 0.13473752, 0.43014994, 0.41334915, 0.39148194, 0.15183382, 0.4178492, 0.54184985, 0.6742418, 0.53520274, 0.43977898, 0.45134482, 0.31229478, 0.94947374, 0.52024204, 0.8126134, 0.35126662, 0.30864513, 0.1798602, 0.52078825, 0.7043505, -0.072655596, 0.4347353, 0.61647487, 0.37701145, 0.24123454, 0.80414706, 0.74672437, 0.8351811, 0.93463886, 0.5414232, 0.32739785, 0.7746117, 0.1259739, 0.6502773, 0.7109401, 0.5597235, 0.51580256, 0.9653897, 0.78936696, 0.5933434, 1.0, 0.3971719, 0.41050434, -0.10807815, 0.40282238, -0.08437647, 0.1520761, 0.62323403, 0.1473397, 0.9999999, 0.6649166, 0.47562003, 0.5333497, 0.76837015, 0.8246984, 0.41575605, 1.0, 0.12679763, 0.8072025, 0.7527665, 0.6664799, 0.30387455, 0.19171491, 1.0, 0.5494079, 0.40092728, 0.26025066, 0.8344779, 0.509655, 0.3882088, 0.045273293, 1.0000001, 0.67510235, 0.20505303, 0.5121501, 0.40179884, 0.5840781, 0.013229348, 1.0000001, 0.5332004, 0.45103407, 0.44107103, 0.80851465, 0.8604594, 0.33438542, 0.4785356, 0.16416258, 1.0, 0.44357347, 0.8327395, 0.9431208, 0.79262006, 0.22822918, 0.9310982, 0.42535427, 0.6676611, 0.701594, 0.0655586, 0.70730686, 0.6453576, 0.850573, 0.27357247, 0.04999528, 0.34648672, 0.54381514, 0.8058342, 0.73593456, 0.36516735, 0.68037015, 1.0, 0.5233661, 0.5605658, 0.61525005, 1.0, 0.5514031, 0.70329225, 0.5244669, 0.9999999, 0.58348125, 0.63622075, 0.6374111, 0.014809623, 0.4143563, 0.41816372]

danhgiachuyengia=[0.75, 1.0, 0.5, 0.0, 0.5, 0.75, 0.0, 0.0, 0.5, 0.75, 0.25, 0.0, 1.0, 0.5, 0.75, 0.0, 1.0, 0.75, 0.25, 0.75, 0.5, 0.25, 1.0, 0.25, 0.75, 0.0, 0.25, 1.0, 0.25, 0.75, 1.0, 0.25, 0.0, 0.75, 0.5, 1.0, 0.25, 0.0, 1.0, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 1.0, 0.25, 0.25, 0.0, 1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 0.0, 0.75, 0.75, 0.25, 1.0, 1.0, 0.5, 1.0, 0.75, 0.75, 0.25, 0.75, 0.75, 0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0, 0.25, 0.75, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 0.75, 0.0, 0.25, 1.0, 0.75, 0.25, 0.5, 1.0, 0.25, 1.0, 0.25, 0.25, 1.0, 1.0, 0.25, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.75, 1.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5, 0.75, 0.0, 0.5, 0.75, 0.5, 0.75, 0.5, 0.25, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.75, 0.0, 0.0, 0.25, 0.5, 0.0, 0.0, 1.0, 1.0, 0.75, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.5, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25,
0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.25, 0.75, 0.25, 0.0, 0.5, 0.0, 0.0, 0.25, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.5, 0.0, 0.5, 0.75, 0.25, 0.0, 0.25, 0.75, 0.0, 0.25, 0.5, 0.75, 0.5, 1.0, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.75, 0.5, 0.0, 0.25, 0.75, 0.5, 1.0, 0.5, 0.25, 1.0, 0.5, 0.75, 0.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.75, 0.75, 1.0, 0.75, 1.0, 0.25, 1.0, 0.75, 0.0, 0.5, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.5, 0.25, 0.5, 1.0, 1.0, 0.0, 0.75, 0.75, 0.75, 0.25, 0.75, 0.75, 0.5, 1.0, 0.75, 0.75, 0.75, 0.0, 0.75, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.0, 0.5, 0.25, 0.25, 0.75, 0.0, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.0, 1.0, 0.75, 0.5, 0.25, 0.25, 1.0, 0.5, 0.25, 0.5, 0.5, 0.75, 0.75, 0.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 1.0, 0.75, 0.75, 0.0, 1.0, 1.0, 0.75, 0.75, 0.25, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 0.75, 0.0, 1.0, 0.5, 1.0, 0.75, 0.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.5, 1.0, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.0, 0.5, 0.75]

causo = np.arange(1, 601)

pp.plot(causo,danhgiasim,color='purple')
pp.plot(causo,danhgiachuyengia,color='red')

pp.show()
