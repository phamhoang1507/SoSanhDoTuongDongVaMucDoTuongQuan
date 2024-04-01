import matplotlib.pyplot as plt
import numpy as np
# Dữ liệu cho biểu đồ 1
causo = np.arange(1, 601)
danhgiachuyengia=[0.75, 1.0, 0.5, 0.0, 0.5, 0.75, 0.0, 0.0, 0.5, 0.75, 0.25, 0.0, 1.0, 0.5, 0.75, 0.0, 1.0, 0.75, 0.25, 0.75, 0.5, 0.25, 1.0, 0.25, 0.75, 0.0, 0.25, 1.0, 0.25, 0.75, 1.0, 0.25, 0.0, 0.75, 0.5, 1.0, 0.25, 0.0, 1.0, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 1.0, 0.25, 0.25, 0.0, 1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 0.0, 0.75, 0.75, 0.25, 1.0, 1.0, 0.5, 1.0, 0.75, 0.75, 0.25, 0.75, 0.75, 0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0, 0.25, 0.75, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 0.75, 0.0, 0.25, 1.0, 0.75, 0.25, 0.5, 1.0, 0.25, 1.0, 0.25, 0.25, 1.0, 1.0, 0.25, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.75, 1.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5, 0.75, 0.0, 0.5, 0.75, 0.5, 0.75, 0.5, 0.25, 0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.0, 0.0, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.25, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.75, 0.0, 0.0, 0.25, 0.5, 0.0, 0.0, 1.0, 1.0, 0.75, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.5, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25,
0.25, 0.0, 0.0, 1.0, 0.75, 0.75, 0.5, 0.75, 0.5, 0.75, 0.25, 0.75, 0.25, 0.0, 0.5, 0.0, 0.0, 0.25, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.5, 0.0, 0.5, 0.75, 0.25, 0.0, 0.25, 0.75, 0.0, 0.25, 0.5, 0.75, 0.5, 1.0, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.75, 0.5, 0.0, 0.25, 0.75, 0.5, 1.0, 0.5, 0.25, 1.0, 0.5, 0.75, 0.0, 0.75, 0.75, 0.75, 0.5, 0.75, 0.75, 0.75, 1.0, 0.75, 1.0, 0.25, 1.0, 0.75, 0.0, 0.5, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.5, 0.25, 0.5, 1.0, 1.0, 0.0, 0.75, 0.75, 0.75, 0.25, 0.75, 0.75, 0.5, 1.0, 0.75, 0.75, 0.75, 0.0, 0.75, 0.5, 0.5, 0.5, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.0, 0.5, 0.25, 0.25, 0.75, 0.0, 1.0, 1.0, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.0, 1.0, 0.75, 0.5, 0.25, 0.25, 1.0, 0.5, 0.25, 0.5, 0.5, 0.75, 0.75, 0.0, 1.0, 0.75, 0.0, 0.5, 0.5, 0.75, 0.5, 1.0, 0.75, 0.75, 0.0, 1.0, 1.0, 0.75, 0.75, 0.25, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 0.75, 0.0, 1.0, 0.5, 1.0, 0.75, 0.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.5, 1.0, 1.0, 0.75, 0.75, 1.0, 0.75, 0.75, 0.75, 1.0, 0.75, 0.75, 0.5, 0.0, 0.5, 0.75]


# Dữ liệu cho biểu đồ 2
danhgiasim=[0.5429616570472717, 0.9953362941741943, 0.6700404286384583, 0.3908151686191559, 0.7243478298187256, 0.7323312759399414, 0.27908167243003845, 0.5094850659370422, 0.8517231941223145, 0.8058063983917236, 0.6602434515953064, 0.5565595626831055, 0.9817479252815247, 0.7975282669067383, 0.9531183838844299, 0.6360112428665161, 0.9186694622039795, 0.9687082767486572, 0.520943820476532, 0.9693642854690552, 0.8309423923492432, 0.5976107120513916, 0.9200536608695984, 0.5764280557632446, 0.9136621356010437, 0.609735906124115, 0.5093466639518738, 0.9853360652923584, 0.5698574185371399, 0.9268567562103271, 0.8311325311660767, 0.54606032371521, 0.5632047653198242, 0.9643338918685913, 0.48512035608291626, 0.9018197059631348, 0.5462009906768799, 0.6141352653503418, 0.9497401714324951, 0.8114992380142212, 0.2209600955247879, 0.9889300465583801, 1.0000001192092896, 0.8324752449989319, 0.5329779386520386, 0.9741094708442688, 0.6059072017669678, 0.4948406219482422, 0.5712549090385437, 0.9821354150772095, 0.9836446046829224, 0.9999999403953552, 0.6328884363174438, 0.658786416053772, 0.989516019821167, 0.9856758713722229, 0.7648903727531433, 0.9491925835609436, 0.9768938422203064, 0.5830693244934082, 0.986977219581604, 0.6355005502700806, 0.4236285388469696, 0.6533554196357727, 0.5137230753898621, 0.35058265924453735, 0.38424938917160034, 0.5426133871078491, 0.5528787970542908, 0.24256078898906708, 0.25470370054244995, 0.17063480615615845, 0.2969953715801239, 0.6350815296173096, 0.3198046386241913, 0.3296211063861847, 0.28880682587623596, 0.38418281078338623, 0.6285059452056885, 0.8837304711341858, 0.33062028884887695, 0.7708356380462646, 0.7650249004364014, 0.2731013596057892, 0.6043403148651123, 0.902839183807373, 1.0, 0.7677936553955078, 0.5011777281761169, 0.2933526933193207, 0.5575568079948425, 0.8758882284164429, 0.9904098510742188, 0.9680287837982178, 0.3263823390007019, 0.5699490904808044, 0.945182204246521, 0.9484105110168457, 0.34748029708862305, 0.4221509099006653, 0.8375036120414734, 0.7551459670066833, 0.9796426296234131, 0.41646674275398254, 0.7751250267028809, 0.8660904169082642, 0.7863216400146484, 0.752861499786377, 0.9392604827880859, 0.8687077164649963, 0.81194007396698, 0.9598257541656494, 0.6425960063934326, 0.7261872291564941, 0.784988284111023, 0.656091570854187, 0.9399593472480774, 0.8649119138717651, 0.8221759796142578, 0.8135696649551392, 0.7229009866714478, 0.8605597615242004, 0.6317599415779114, 0.8041698932647705, 0.9243739247322083, 0.9066267013549805, 0.8671891689300537, 0.9432674050331116, 0.8651703596115112, 0.8288248181343079, 0.8406896591186523, 0.5773807764053345, 0.8089675903320312, 0.7376671433448792, 0.5781539082527161, 0.7658013105392456, 0.752861499786377, 0.7934507727622986, 0.7302433848381042, 0.8759734034538269, 0.8311097621917725, 0.7267001271247864, 0.6402918100357056, 0.6016507744789124, 0.8911019563674927, 0.9045867323875427, 0.8438625335693359, 0.8695254325866699, 0.6873937845230103, 0.26293131709098816,
0.9627436995506287, 0.6541458964347839, 0.7810946702957153, 0.4968639016151428, 0.8609004616737366, 0.6147698760032654, 0.6818544864654541, 0.7816640138626099, 0.6580109000205994, 0.5553412437438965, 0.3758769631385803, 0.5906950831413269, 0.6944580674171448, 0.8016882538795471, 0.5412843823432922, 0.47779932618141174, 0.6232244372367859, 0.4618586599826813, 0.6183137893676758, 0.31661075353622437, 0.6080109477043152, 0.41646674275398254, 0.5135371685028076, 0.4968675971031189, 0.7614060044288635, 0.4043010473251343, 0.4152156710624695, 0.8979560136795044, 0.5490361452102661, 0.9772631525993347, 0.67430579662323, 0.8329886198043823, 0.9755370616912842, 0.9419229030609131, 0.8030960559844971, 0.5163364410400391, 0.6787258386611938, 0.818975031375885, 0.7026301026344299, 0.8835476636886597, 0.8910068869590759, 0.8884505033493042, 0.8110058903694153, 0.2655772864818573, 0.49065765738487244, 0.43361490964889526, 0.5484643578529358, 0.25484928488731384, 0.1622398942708969, 0.9357655048370361, 0.9572274684906006, 0.7030915021896362, 0.888124942779541, 0.4183526337146759, 0.4764196574687958, 0.18514187633991241, 0.6485909819602966, 0.18770286440849304, 0.11970274895429611, 0.9644851684570312, 0.8999971151351929, 0.6691467761993408, 0.8353913426399231, 0.6147662997245789, 0.7306902408599854, 0.4853183925151825, 0.7229049801826477, 0.5190980434417725, 0.17726583778858185, 0.837924599647522, 0.7849196791648865, 0.5744782090187073, 0.7929528951644897, 0.537837564945221, 0.5692267417907715, 0.4934334456920624, 0.7402443289756775, 0.19533754885196686, 0.04867228865623474, 0.5600149631500244, 0.30704429745674133, 0.3501029908657074, 0.5153281688690186, 0.6268438100814819, 0.5569474697113037, 0.309489369392395, 0.6607894897460938, 0.30338728427886963, 0.9155597686767578, 0.95768803358078, 0.9316527247428894, 0.9385513067245483, 0.742946207523346, 0.8751161694526672, 0.7189661264419556, 0.7239416837692261, 0.6654787659645081, 0.17293161153793335, 0.27488335967063904, 0.641735315322876, 0.7584089040756226, 0.5161607265472412, 0.7368825078010559, 0.4844539165496826, 0.42684441804885864, 0.24835893511772156, 0.4673784375190735, 0.16821332275867462, 0.10928986966609955, 0.9270395636558533, 0.9566906690597534, 0.803917407989502, 0.7091286182403564, 0.5057172179222107, 0.8019226789474487, 0.3653451204299927, 0.6756033301353455, 0.48584794998168945, 0.2741710841655731, 0.9307043552398682, 0.9624230265617371, 0.8276327252388, 0.6579859256744385, 0.5349090695381165, 0.8019226789474487, 0.36854737997055054, 0.6756033301353455, 0.7013940215110779, 0.6633071899414062, 0.9644851684570312, 0.8999971151351929, 0.5295392870903015, 0.7929528951644897, 0.6268438100814819, 0.5569474697113037, 0.8158040046691895, 0.682723343372345, 0.19533754885196686, 0.0224989652633667, 0.9076530337333679, 0.8973237872123718, 0.8260665535926819, 0.7882287502288818, 0.4795151650905609, 0.2659645080566406, 0.8158040046691895, 0.682723343372345, 0.3061855137348175, 0.39936456084251404,
0.9999999403953552, 0.8456663489341736, 0.958252489566803, 0.73627108335495, 0.7088617086410522, 0.6147662997245789, 0.34005290269851685, 0.4853183925151825, 0.7229049801826477, 0.5190980434417725, 0.17726583778858185, 0.9801592230796814, 0.9604437947273254, 0.8725298047065735, 0.7929528951644897, 0.6268438100814819, 0.5569474697113037, 0.585040807723999, 0.4673460125923157, 0.25535258650779724, 0.0224989652633667, 0.6902135014533997, 0.803460419178009, 0.7152044773101807, 0.7171214818954468, 0.6332003474235535, 0.6158567667007446, 0.6307268738746643, 0.6788744926452637, 0.2634138762950897, 0.23651719093322754, 0.8284960985183716, 0.9622199535369873, 0.9273672103881836, 0.7446742653846741, 0.4839257597923279, 0.6008667945861816, 0.5402266383171082, 0.47775566577911377, 0.22366954386234283, 0.15373776853084564, 0.9607774019241333, 0.9572958946228027, 0.7246301770210266, 0.5451095700263977, 0.3517501652240753, 0.3020990192890167, 0.4422440528869629, 0.43876296281814575, 0.41107940673828125, 0.0760178491473198, 0.9189966917037964, 0.8804786205291748, 0.6177466511726379, 0.824655294418335, 0.6354432702064514, 0.6615971326828003, 0.36478888988494873, 0.3840869665145874, 0.055762842297554016, 0.14911338686943054, 0.8835476636886597, 0.8910068869590759, 0.8884505033493042, 0.8110058903694153, 0.2655772864818573, 0.6994692087173462, 0.43361490964889526, 0.5484643578529358, 0.25484928488731384, 0.1622398942708969, 0.9576441049575806, 0.6677656769752502, 0.9147135019302368, 0.6180647611618042, 0.7397974729537964, 0.6137492656707764, 0.41290247440338135, 0.6276785731315613, 0.4260282516479492, 0.4733688235282898, 0.5600149631500244, 0.592552125453949, 0.6109381914138794, 0.5153281688690186, 0.6268438100814819, 0.5569474697113037, 0.309489369392395, 0.6607894897460938, 0.30338728427886963, 0.9155597686767578, 0.8242933750152588, 0.803460419178009, 0.5989614129066467, 0.7929528951644897, 0.537837564945221, 0.5692267417907715, 0.4934334456920624, 0.7402443289756775, 0.19533754885196686, 0.04867228865623474, 0.5288630127906799, 0.8925492763519287, 0.677551805973053, 0.5650577545166016, 0.5326154232025146, 0.5667486786842346, 0.3634049892425537, 0.36908382177352905, 0.3552567958831787, 0.2816632390022278, 0.9629499316215515, 0.6768210530281067, 0.7435101866722107, 0.8613482117652893, 0.9196131229400635, 0.83359694480896, 0.775431215763092, 0.47008419036865234, 0.9381031394004822, 0.6173450350761414, 0.23597735166549683, 0.6964573264122009, 0.72913658618927, 0.39625781774520874, 0.6641589403152466, 0.9170259237289429, 0.8180588483810425, 0.786077618598938, 0.7266582250595093, 0.8158444762229919, 0.8293910622596741, 0.6073809862136841, 0.5590755343437195, 0.4575571119785309, 0.8794588446617126, 0.858302116394043, 0.5973531007766724, 0.10068883746862411, 0.44642651081085205, 0.8430241346359253, 0.4250125288963318, 0.719892680644989, 0.7762227058410645, 0.8941292762756348, 0.7460338473320007, 0.9722510576248169, 0.8682547807693481, 0.819913923740387,
0.5630830526351929, 0.8881428241729736, 0.7900322675704956, 0.8019553422927856, 0.4022449553012848, 0.7916897535324097, 0.8758767247200012, 0.22174081206321716, 0.6516241431236267, 0.8004001379013062, 0.866820752620697, 0.9832074642181396, 0.6387187242507935, 0.37649425864219666, 0.9536895751953125, 0.6839473843574524, 0.9128456115722656, 0.4956415593624115, 0.8562315702438354, 0.9498987793922424, 0.8521065711975098, 0.5324811339378357, 0.919529914855957, 0.7616704702377319, 0.8539469838142395, 0.9447559714317322, 0.9479445219039917, 0.981249213218689, 0.6947429180145264, 0.9265919327735901, 0.8538681268692017, 0.4508201777935028, 0.4918808341026306, 0.28303951025009155, 0.8482593894004822, 0.9272509813308716, 0.7504935264587402, 0.7460960745811462, 0.9003021717071533, 0.8034543991088867, 0.9717219471931458, 0.907719612121582, 0.9027287364006042, 0.7464603781700134, 0.60211181640625, 0.6831530332565308, 0.9151668548583984, 0.8042448163032532, 0.3374599814414978, 0.6667670607566833, 0.7505009174346924, 0.8511730432510376, 0.5898795127868652, 0.9047149419784546, 0.9643459916114807, 0.9109177589416504, 0.9644877314567566, 0.4892217218875885, 0.5318461656570435, 0.9341965913772583, 0.44272857904434204, 0.8138620257377625, 0.9412482976913452, 0.5582469701766968, 0.7363163828849792, 0.9902997016906738, 0.9718828797340393, 0.8463886380195618, 1.0, 0.6950797438621521, 0.6022480726242065, 0.2246933877468109, 0.6849620938301086, 0.42705899477005005, 0.5674585103988647, 0.6653510928153992, 0.4285559058189392, 0.9999998807907104, 0.8040411472320557, 0.8032840490341187, 0.8700778484344482, 0.7550801038742065, 0.8750600814819336, 0.7711895704269409, 0.9999998807907104, 0.21995815634727478, 0.808760941028595, 0.8575201034545898, 0.7078499794006348, 0.6429733037948608, 0.565780520439148, 0.9999999403953552, 0.8526520133018494, 0.7707705497741699, 0.9001230597496033, 0.7494359016418457, 0.7076992392539978, 0.30630066990852356, 0.38562050461769104, 1.0, 0.8790847659111023, 0.3118550777435303, 0.7048709988594055, 0.603187084197998, 0.8251155614852905, 0.7831240892410278, 0.9999998807907104, 0.8174461126327515, 0.7960693836212158, 0.5905847549438477, 0.7849406599998474, 0.9053467512130737, 0.7687071561813354, 0.8332633376121521, 0.6073092222213745, 1.0, 0.8549634218215942, 0.7148398160934448, 0.9379879236221313, 0.7721846103668213, 0.7510965466499329, 0.9560456871986389, 0.7336298227310181, 0.909456729888916, 0.7602866291999817, 0.2675286829471588, 0.9102038145065308, 0.6794922351837158, 0.8984657526016235, 0.7782893776893616, 0.3716704845428467, 0.9341800808906555, 0.7520345449447632, 0.8836876749992371, 0.7556913495063782, 0.5098124742507935, 0.7742125391960144, 1.0000001192092896, 0.9706451892852783, 0.773784875869751, 0.7815784215927124, 1.0, 0.8070284724235535, 0.9167638421058655, 0.7405744791030884, 1.0000001192092896, 0.6720896363258362, 0.855359673500061, 0.5194534063339233, 0.3749842047691345, 0.7249078750610352, 0.7956905364990234]


# Tạo và cấu hình subplot
plt.figure(figsize=(10, 5))  # Kích thước của subplot
plt.subplot(1, 2, 1)  # Chia màn hình thành 1 hàng và 2 cột, và chọn subplot thứ 1
plt.plot(causo,danhgiasim,color='purple') # Vẽ biểu đồ cột cho dữ liệu 1
plt.title('Biểu đồ 1')  # Đặt tiêu đề cho subplot đầu tiên

plt.subplot(1, 2, 2)  # Chọn subplot thứ 2
plt.plot(causo,danhgiachuyengia,color='red') # Vẽ biểu đồ cột cho dữ liệu 2
plt.title('Biểu đồ 2')  # Đặt tiêu đề cho subplot thứ hai

plt.tight_layout()  # Tùy chỉnh khoảng cách giữa các subplot
plt.show()  # Hiển thị biểu đồ