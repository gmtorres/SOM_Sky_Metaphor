# Read data from Java SOMToolbox
from SOMToolBox_Parse import SOMToolBox_Parse
idata = SOMToolBox_Parse("datasets\\iris\\iris.vec").read_weight_file()
weights = SOMToolBox_Parse("datasets\\iris\\iris.wgt.gz").read_weight_file()
classes = SOMToolBox_Parse("datasets\\iris\\iris.cls").read_weight_file()

from somtoolbox import SOMToolbox

sm = SOMToolbox(weights=weights['arr'],m=weights['ydim'],n=weights['xdim'],
                dimension=weights['vec_dim'], input_data=idata['arr'],
               classes=classes['arr'], component_names=classes['classes_names'])
sm._mainview


from somtoolbox import SOMToolbox
from minisom import MiniSom    


som = MiniSom(10, 10, 4, sigma=0.3, learning_rate=0.5)
som.train(idata['arr'], 1000)

sm = SOMToolbox(weights=som._weights.reshape(-1,4), m=10, n=10, dimension=4, input_data=idata['arr'])
sm._mainview