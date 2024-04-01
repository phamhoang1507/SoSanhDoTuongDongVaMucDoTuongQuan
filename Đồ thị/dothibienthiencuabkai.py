import matplotlib.pyplot as plt
import numpy as np
# Dữ liệu cho biểu đồ 1
causo = np.arange(1, 601)
danhgiachuyengia=[0.75, 1.0, 0.5, 0.0, 0.5, 0.75, 0.0, 0.0, 0.5, 0.75, 0.25, 0.0, 1.0, 0.5, 0.75, 0.0, 1.0, 0.75, 0.25, 0.75, 0.5, 0.25, 1.0, 0.25, 0.75, 0.0, 0.25, 1.0, 0.25, 0.75, 1.0, 0.25, 0.0, 0.75, 0.5, 1.0, 0.25, 0.0, 1.0, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 1.0, 0.25, 0.25, 0.0, 1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 0.0, 0.75, 0.75, 0.25, 1.0, 1.0, 0.5, 1.0, 0.75, 0.75, 0.25, 0.75, 0.75, 0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0, 0.25, 0.75, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 0.75, 0.0, 0.25, 1.0, 0.75, 0.25, 0.5, 1.0, 0.25, 1.0, 0.25, 0.25, 1.0, 1.0, 0.25, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.75, 1.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5, 0.75, 0.0, 0.5, 0.75, 0.5, 0.75, 0.5, 0.25, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.75, 0.0, 0.0, 0.25, 0.5, 0.0, 0.0, 1.0, 1.0, 0.75, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.5, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25,
0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.25, 0.75, 0.25, 0.0, 0.5, 0.0, 0.0, 0.25, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.5, 0.0, 0.5, 0.75, 0.25, 0.0, 0.25, 0.75, 0.0, 0.25, 0.5, 0.75, 0.5, 1.0, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.75, 0.5, 0.0, 0.25, 0.75, 0.5, 1.0, 0.5, 0.25, 1.0, 0.5, 0.75, 0.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.75, 0.75, 1.0, 0.75, 1.0, 0.25, 1.0, 0.75, 0.0, 0.5, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.5, 0.25, 0.5, 1.0, 1.0, 0.0, 0.75, 0.75, 0.75, 0.25, 0.75, 0.75, 0.5, 1.0, 0.75, 0.75, 0.75, 0.0, 0.75, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.0, 0.5, 0.25, 0.25, 0.75, 0.0, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.0, 1.0, 0.75, 0.5, 0.25, 0.25, 1.0, 0.5, 0.25, 0.5, 0.5, 0.75, 0.75, 0.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 1.0, 0.75, 0.75, 0.0, 1.0, 1.0, 0.75, 0.75, 0.25, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 0.75, 0.0, 1.0, 0.5, 1.0, 0.75, 0.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.5, 1.0, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.0, 0.5, 0.75]


# Dữ liệu cho biểu đồ 2
danhgiabkai=[0.5439903140068054, 0.9824670553207397, 0.7632510662078857, 0.384459912776947, 0.7249449491500854, 0.6211104393005371, 0.3669186234474182, 0.3381597697734833, 0.8028558492660522, 0.6576968431472778, 0.6664958596229553, 0.5992695093154907, 0.9676824808120728, 0.7934331297874451, 0.8833308219909668, 0.45990103483200073, 0.9742900729179382, 0.9600070714950562, 0.35151779651641846, 0.9593842029571533, 0.8890537619590759, 0.3340131640434265, 0.9341583847999573, 0.37168246507644653, 0.9329047799110413, 0.5994445085525513, 0.2585727572441101, 0.9786743521690369, 0.2971124053001404, 0.8299465179443359, 0.8387912511825562, 0.3383392095565796, 0.5125221014022827, 0.930274486541748, 0.37953102588653564, 0.9161006212234497, 0.42464762926101685, 0.19981534779071808, 0.9329423904418945, 0.8170396089553833, 0.35185837745666504, 0.9807971119880676, 0.9999999403953552, 0.6372196078300476, 0.44750750064849854, 0.988625168800354, 0.5936383008956909, 0.424858421087265, 0.3717629909515381, 0.9581164121627808, 0.9523190259933472, 1.0000001192092896, 0.3698809742927551, 0.35180437564849854, 0.9785940051078796, 0.9565679430961609, 0.6133036613464355, 0.9247951507568359, 0.9635896682739258, 0.4812438189983368, 0.9868229627609253, 0.4229338765144348, 0.3113655745983124, 0.5482472777366638, 0.4314941167831421, 0.23551712930202484, 0.23229530453681946, 0.5168493986129761, 0.4717070162296295, 0.1762995719909668, 0.1788213849067688, 0.10969170182943344, 0.13132864236831665, 0.3934035897254944, 0.24758897721767426, 0.14085611701011658, 0.3576997220516205, 0.22250019013881683, 0.3285525441169739, 0.84247887134552, 0.06531259417533875, 0.8626643419265747, 0.5588637590408325, 0.07684148848056793, 0.4892733693122864, 0.9457626938819885, 0.9999999403953552, 0.5176491141319275, 0.39712846279144287, 0.3814338147640228, 0.42001819610595703, 0.903180718421936, 0.9609096050262451, 0.9475287199020386, 0.4403630793094635, 0.4273360073566437, 0.951907217502594, 0.9236833453178406, 0.2981511950492859, 0.22123129665851593, 0.8797031044960022, 0.8612779378890991, 0.9646605253219604, 0.36296743154525757, 0.7579798698425293, 0.6983912587165833, 0.6985776424407959, 0.5910332202911377, 0.8246738314628601, 0.6976886987686157, 0.7460925579071045, 0.9085119366645813, 0.8284525871276855, 0.7667051553726196, 0.7576962113380432, 0.7059666514396667, 0.9218282699584961, 0.8599871397018433, 0.7198992967605591, 0.8116457462310791, 0.7100375890731812, 0.4698788523674011, 0.6185353398323059, 0.7605035305023193, 0.9385246634483337, 0.942002534866333, 0.7125649452209473, 0.8742138147354126, 0.7440133094787598, 0.6410825252532959, 0.5391021370887756, 0.6459993124008179, 0.7796480059623718, 0.6951523423194885, 0.6028042435646057, 0.6671749353408813, 0.5910332202911377, 0.6357412338256836, 0.45529890060424805, 0.8043279051780701, 0.6709564924240112, 0.6925325393676758, 0.6440668106079102, 0.5996628999710083, 0.8508001565933228, 0.8213025331497192, 0.8248344659805298, 0.8801112174987793, 0.6494263410568237, 0.3195062279701233, 0.88319331407547, 0.6099095940589905, 0.7478438019752502, 0.3500290513038635, 0.7963035702705383, 0.639083981513977, 0.5294409990310669, 0.7818130850791931, 0.5898722410202026, 0.5352956056594849, 0.5527212619781494, 0.4246034622192383, 0.6151714324951172, 0.6220531463623047, 0.4653671979904175, 0.2760314643383026, 0.6452887058258057, 0.44276517629623413, 0.3456302285194397, 0.22161245346069336, 0.5338409543037415, 0.36296743154525757, 0.4087978005409241, 0.3395143151283264, 0.6050375699996948, 0.35358762741088867, 0.4329291880130768, 0.7004445195198059, 0.6301263570785522, 0.9366423487663269, 0.9307421445846558, 0.7050338983535767, 0.8819911479949951, 0.9245790243148804, 0.7809242010116577, 0.5053845643997192, 0.6245696544647217, 0.5465129017829895, 0.7736048698425293, 0.8047674298286438, 0.818651020526886, 0.7330234050750732, 0.6730706095695496, 0.1399603933095932, 0.3212074041366577, 0.18440204858779907, 0.26362109184265137, 0.0955282673239708, 0.06259848177433014, 0.9433977603912354, 0.9622185826301575, 0.5726509094238281, 0.7606027126312256, 0.3787924647331238, 0.4909461736679077, 0.04578311741352081, 0.4156929552555084, 0.11576002091169357, -0.011495627462863922, 0.857193112373352, 0.8115222454071045, 0.8801175951957703, 0.6874899864196777, 0.5622632503509521, 0.8728024959564209, 0.35237836837768555, 0.5532597303390503, 0.3179497718811035, 0.13611823320388794, 0.6738401651382446, 0.6435986161231995, 0.5363299250602722, 0.8145754337310791, 0.4708758592605591, 0.7324935793876648, 0.29497015476226807, 0.589893102645874, -0.002416455652564764, -0.06996767222881317, 0.49561697244644165, 0.36774787306785583, 0.2722824215888977, 0.30443739891052246, 0.6004177331924438, 0.008942369371652603, 0.22710031270980835, 0.5281587839126587, 0.27784591913223267, 0.7995613813400269, 0.928217887878418, 0.7156880497932434, 0.8748815059661865, 0.8027387857437134, 0.7885900139808655, 0.42852258682250977, 0.5605113506317139, 0.36385631561279297, 0.027560850605368614, 0.09577743709087372, 0.42562130093574524, 0.4553179144859314, 0.49963998794555664, 0.46112585067749023, 0.29928648471832275, 0.19067247211933136, 0.1747359037399292, 0.2557923197746277, 0.11741673201322556, -0.05080823227763176, 0.9415273070335388, 0.9466438293457031, 0.6830106973648071, 0.5020808577537537, 0.378598153591156, 0.629547655582428, 0.29030001163482666, 0.5261112451553345, 0.20787550508975983, 0.11980728805065155, 0.9208575487136841, 0.9521660208702087, 0.5460668802261353, 0.5265624523162842, 0.4509185254573822, 0.629547655582428, 0.2959222197532654, 0.5261112451553345, 0.45548540353775024, 0.26319098472595215, 0.857193112373352, 0.8115222454071045, 0.6217374801635742, 0.8145754337310791, 0.6004177331924438, 0.008942369371652603, 0.6747102737426758, 0.41015636920928955, -0.002416455652564764, -0.05568014085292816, 0.7762260437011719, 0.7486578226089478, 0.6996703743934631, 0.7122870087623596, 0.46493107080459595, 0.1007484495639801, 0.6747102737426758, 0.41015636920928955, 0.20953857898712158, 0.17023508250713348, 0.9999996423721313, 0.634182333946228, 0.835365891456604, 0.5629492998123169, 0.5845184326171875, 0.5622632503509521, 0.29457518458366394, 0.35237836837768555, 0.5532597303390503, 0.3179497718811035, 0.13611823320388794, 0.9014869332313538, 0.8720817565917969, 0.5547256469726562, 0.8145754337310791, 0.6004177331924438, 0.008942369371652603, 0.507237434387207, 0.3723706007003784, 0.2510526180267334, -0.05568014085292816, 0.7126696109771729, 0.8203172087669373, 0.6979824304580688, 0.5846154689788818, 0.5293603539466858, 0.5798336267471313, 0.4547601044178009, 0.678727924823761, 0.08034588396549225, 0.16449546813964844, 0.7873185873031616, 0.8850707411766052, 0.774045467376709, 0.7105637788772583, 0.29641708731651306, 0.3783307671546936, 0.32195863127708435, 0.3163434863090515, 0.058814406394958496, 0.10609492659568787, 0.8199038505554199, 0.8936753273010254, 0.789760410785675, 0.7069003582000732, 0.19524109363555908, 0.15478761494159698, 0.3942665457725525, 0.19968223571777344, 0.2196037471294403, -0.05715832859277725, 0.7458291053771973, 0.8345954418182373, 0.5302045345306396, 0.5478054285049438, 0.34282082319259644, 0.4811512231826782, 0.3628426790237427, 0.3082960844039917, 0.018828438594937325, 0.16409297287464142, 0.8047674298286438, 0.818651020526886, 0.7330234050750732, 0.6730706095695496, 0.1399603933095932, 0.5241079330444336, 0.18440204858779907, 0.2735992670059204, 0.0955282673239708, 0.06259848177433014, 0.8654931783676147, 0.6112843751907349, 0.8348686695098877, 0.6814275979995728, 0.560088038444519, 0.5338475704193115, 0.34411880373954773, 0.35619252920150757, 0.09748269617557526, 0.31627410650253296, 0.49561697244644165, 0.5449216961860657, 0.44884076714515686, 0.30443739891052246, 0.6004177331924438, 0.008942369371652603, 0.22710031270980835, 0.5281587839126587, 0.27784591913223267, 0.7995613813400269, 0.6737024784088135, 0.8203172087669373, 0.6173673868179321, 0.8145754337310791, 0.4708758592605591, 0.7324935793876648, 0.29497015476226807, 0.589893102645874, -0.002416455652564764, -0.06996767222881317, 0.5679163932800293, 0.8072496056556702, 0.6346778869628906, 0.543418824672699, 0.35042649507522583, 0.4732253849506378, 0.23050743341445923, 0.3859891891479492, 0.291732519865036, 0.25950419902801514, 0.8844072818756104, 0.6062978506088257, 0.589777946472168, 0.8411175608634949, 0.9012485146522522, 0.7410637140274048, 0.6926289796829224, 0.34940940141677856, 0.8413457870483398, 0.4497417211532593, 0.406791627407074, 0.42085450887680054, 0.5291475057601929, 0.27256691455841064, 0.43740665912628174, 0.8847455978393555, 0.7125158309936523, 0.7634234428405762, 0.6981967687606812, 0.6255365610122681, 0.582480788230896, 0.3209061026573181, 0.515483021736145, 0.39843976497650146, 0.8254892826080322, 0.7941740155220032, 0.6213005185127258, 0.14109787344932556, 0.45976829528808594, 0.8427033424377441, 0.18071693181991577, 0.5327767133712769, 0.6884109377861023, 0.8537890315055847, 0.6694983839988708, 0.8155953884124756, 0.8106105923652649, 0.7317180037498474, 0.6812708973884583, 0.8005139231681824, 0.5107835531234741, 0.5145164728164673, 0.41291534900665283, 0.6182771921157837, 0.6498735547065735, 0.07175882160663605, 0.5119078755378723, 0.7709054350852966, 0.7579792737960815, 0.9362593293190002, 0.33622798323631287, 0.29388049244880676, 0.8722807168960571, 0.3840743601322174, 0.7240675687789917, 0.47033506631851196, 0.7501565217971802, 0.9290096759796143, 0.9143946766853333, 0.5678805708885193, 0.6018129587173462, 0.5367793440818787, 0.6771585941314697, 0.8413479924201965, 0.8471894264221191, 0.8910967111587524, 0.48588770627975464, 0.7430126667022705, 0.7390061020851135, 0.3855402171611786, 0.5332331657409668, 0.48948684334754944, 0.8609219789505005, 0.7515661120414734, 0.5489101409912109, 0.7990075349807739, 0.9326000213623047, 0.7236951589584351, 0.9581690430641174, 0.8203161358833313, 0.9429208040237427, 0.618841290473938, 0.43054482340812683, 0.4072178900241852, 0.8337953686714172, 0.7762273550033569, 0.16387933492660522, 0.6598907113075256, 0.5815857648849487, 0.6627011895179749, 0.3074225187301636, 0.8774250745773315, 0.7666386365890503, 0.90785813331604, 0.9518516063690186, 0.540003776550293, 0.5221155881881714, 0.8601498603820801, 0.30614185333251953, 0.7285184860229492, 0.8635721206665039, 0.4767618775367737, 0.7211344242095947, 0.9932765960693359, 0.9201443195343018, 0.7998038530349731, 0.9999997615814209, 0.7610560655593872, 0.3976367115974426, 0.06873781979084015, 0.6873095035552979, 0.27702799439430237, 0.47262030839920044, 0.7922520637512207, 0.18128913640975952, 1.0, 0.7162245512008667, 0.7929531335830688, 0.8303439617156982, 0.7501969933509827, 0.8260020017623901, 0.6257110834121704, 1.0000001192092896, 0.07496073842048645, 0.7160342931747437, 0.8163009881973267, 0.8029499053955078, 0.5284003615379333, 0.5599251985549927, 0.9999998807907104, 0.8047661781311035, 0.7445207834243774, 0.6486691832542419, 0.8035018444061279, 0.6260110139846802, 0.6626813411712646, 0.2935670018196106, 1.0, 0.8071597218513489, 0.25931671261787415, 0.5703620910644531, 0.6053341627120972, 0.7496916055679321, 0.7450507879257202, 0.9999998807907104, 0.6521091461181641, 0.6652661561965942, 0.7671791315078735, 0.7975074648857117, 0.8218831419944763, 0.5758101344108582, 0.5267261266708374, 0.4559324383735657, 1.0, 0.7769900560379028, 0.7078540921211243, 0.9345269203186035, 0.7400854825973511, 0.6092758178710938, 0.9274488687515259, 0.6517101526260376, 0.6719158887863159, 0.806643009185791, 0.14879204332828522, 0.8141355514526367, 0.7498306035995483, 0.8931335210800171, 0.7690484523773193, 0.22359226644039154, 0.7611812353134155, 0.7255719304084778, 0.8381288051605225, 0.7885558605194092, 0.5697663426399231, 0.807682991027832, 1.0000001192092896, 0.9654145240783691, 0.7195734977722168, 0.8026400804519653, 1.0, 0.7399145364761353, 0.8555713891983032, 0.5250991582870483, 1.0, 0.6881746053695679, 0.7977681756019592, 0.6105761528015137, 0.29163122177124023, 0.6293396949768066, 0.6953205466270447]


# Tạo và cấu hình subplot
plt.figure(figsize=(10, 5))  # Kích thước của subplot
plt.subplot(1, 2, 1)  # Chia màn hình thành 1 hàng và 2 cột, và chọn subplot thứ 1
plt.plot(causo,danhgiabkai,color='purple') # Vẽ biểu đồ cột cho dữ liệu 1
plt.title('Biểu đồ 1')  # Đặt tiêu đề cho subplot đầu tiên

plt.subplot(1, 2, 2)  # Chọn subplot thứ 2
plt.plot(causo,danhgiachuyengia,color='red') # Vẽ biểu đồ cột cho dữ liệu 2
plt.title('Biểu đồ 2')  # Đặt tiêu đề cho subplot thứ hai

plt.tight_layout()  # Tùy chỉnh khoảng cách giữa các subplot
plt.show()  # Hiển thị biểu đồ