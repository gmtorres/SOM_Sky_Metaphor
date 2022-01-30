from operator import ne
import numpy as np
from visualizations.iVisualization import VisualizationInterface
from controls.controllers import MetroMapController
from controls.controllers import SkyMetaphorController
import panel as pn
import holoviews as hv
from holoviews.streams import Pipe
import bokeh.palettes as colors
import random
from skimage.transform import resize

class SkyMetaphor(VisualizationInterface):

	def __init__(self, main):
		self._main = main
		self._controls = SkyMetaphorController(self._calculate, self._main._dim, self._main._component_names, name='Sky Metaphor Visualization')
		self._raw_solutions = []
		self._snapped_lines = []
		self.background = None


	def _activate_controllers(self, ):
		reference = pn.pane.Str("<ul><li><b>Maximilian Holzmueller, 11770953 | Bijan Nikkhah, 11930506 | Gustavo Torres, 12122457</b> \"The Sky Metaphor, Stardinates, for self-organising maps.\"</li></ul>")
		self._main._controls.append(pn.Column(self._controls, reference))
		self._calculate(calculating=True)

	def _deactivate_controllers(self,):
		self._main._pipe_paths.send([])
		self._main._pipe.send([])
		self._main._pdmap[0] = self._main._Image.apply.opts(cmap=self._main._maincontrol.param.colormap, color_levels=None) * self._main._Paths

	def _calculate(self, calculating, calc_back = False, calc_points = True):

		overlay = [] 
		data = None
		
		if (calc_back or self.background == None) and self._controls.back:
			background = None
			if self._controls.background == 'Hist':
				background = self.HitHist(self._main._m,self._main._n,self._main._weights,self._main._idata)
			elif self._controls.background == 'SHD':
				background = self.SDH(self._main._m,self._main._n,self._main._weights,self._main._idata, self._controls.sdh_factor, 0)
			elif self._controls.background == 'UMatrix':
				background = self.UMatrix(self._main._m,self._main._n,self._main._weights, self._main._dim)
			
			if self._controls.high_def:
				background = resize(background, (self._controls.definition_scale, self._controls.definition_scale))

			data = hv.Image(background).opts(xaxis=None, yaxis=None,cmap=self._controls.palette)
			self.background = data
		
		if self.background != None and self._controls.back:
			overlay.append(self.background)

		points = self.calculatePointsPosition()

		overlay.append(hv.Points(np.array(points)).opts(color=self._controls.points_color if self._controls.back else 'black', size=1))

		self._main._pdmap[0] = hv.Overlay(overlay).collate()
		
	
	def calculateExactPosition(self, vector, x_dim, x_scale, y_scale, offset_x, offset_y, y_length ): #in the unit
		lmb = self._controls.lbd
		k = self._controls.neighbors

		dists_x = np.sqrt(np.sum(np.power(self._main._weights - vector, 2), axis=1)) #dist to input vector x
		dists_args_sorted = np.argsort(dists_x)
		bmu = dists_args_sorted[0]
		next_bmus = dists_args_sorted[1:k+1]

		bmu_pos = self.getMapPosition(bmu, x_dim, x_scale, y_scale, offset_x, offset_y, y_length)

		px, py = 0,0
		
		for p in next_bmus:
			f_i = dists_x[bmu] / dists_x[p]
			pos_i = self.getMapPosition(p, x_dim, x_scale, y_scale, offset_x, offset_y, y_length)

			if (pos_i[0] - bmu_pos[0]) != 0:
				dx = f_i * (pos_i[0] - bmu_pos[0])
				px += dx
				
			if (pos_i[1] - bmu_pos[1]) != 0:
				dy = f_i * (pos_i[1] - bmu_pos[1])
				py += dy

		return lmb * px, lmb * py

	
	def getMapPosition(self, index, x_dim, x_scale, y_scale, offset_x, offset_y, y_length):
		x_pos = (index % x_dim) * x_scale + x_scale / 2
		y_pos = (index // x_dim) * y_scale + y_scale / 2
		return x_pos + offset_x, y_pos + y_length + offset_y

	def calculatePointsPosition(self): # in the output space
		x_dim = self._main._n
		y_dim = self._main._m

		x_length = self._main._xlim[1] - self._main._xlim[0]
		y_length = self._main._ylim[1] - self._main._ylim[0]

		offset_x =  (x_length - y_length)/2 if x_length > y_length else 0
		offset_y =  (y_length - x_length)/2 if y_length > x_length else 0

		x_length = 1
		y_length = 1

		x_scale = x_length / x_dim
		y_scale = y_length / y_dim
		
		y_scale = -y_scale
		points = []

	
		for vector in self._main._idata:
			index = np.argmin(np.sqrt(np.sum(np.power(self._main._weights - vector, 2), axis=1))) #best matching unit
			x_pos, y_pos = self.getMapPosition(index, x_dim, x_scale, y_scale, offset_x, offset_y, y_length)
			exact_post = (0,0)
			
			exact_post = self.calculateExactPosition(vector, x_dim, x_scale, y_scale, offset_x, offset_y, y_length)
			x_pos = x_pos + exact_post[0] + self._main._xlim[0] 
			y_pos = y_pos + exact_post[1] + self._main._ylim[0]

			points.append((x_pos, y_pos))

		return points








	def SDH(self, _m, _n, _weights, _idata, factor, approach):
		import heapq

		sdh_m = np.zeros( _m * _n)

		cs=0
		for i in range(factor): cs += factor-i

		for vector in _idata:
			dist = np.sqrt(np.sum(np.power(_weights - vector, 2), axis=1))
			c = heapq.nsmallest(factor, range(len(dist)), key=dist.__getitem__)
			if (approach==0): # normalized
				for j in range(factor):  sdh_m[c[j]] += (factor-j)/cs 
			if (approach==1):# based on distance
				for j in range(factor): sdh_m[c[j]] += 1.0/dist[c[j]] 
			if (approach==2): 
				dmin, dmax = min(dist[c]), max(dist[c])
				for j in range(factor): sdh_m[c[j]] += 1.0 - (dist[c[j]]-dmin)/(dmax-dmin)

		return sdh_m.reshape(_m, _n)

	def UMatrix(self, _m, _n, _weights, _dim):
		U = _weights.reshape(_m, _n, _dim)
		U = np.insert(U, np.arange(1, _n), values=0, axis=1)
		U = np.insert(U, np.arange(1, _m), values=0, axis=0)
		#calculate interpolation
		for i in range(U.shape[0]): 
			if i%2==0:
				for j in range(1,U.shape[1],2):
					U[i,j][0] = np.linalg.norm(U[i,j-1] - U[i,j+1], axis=-1)
			else:
				for j in range(U.shape[1]):
					if j%2==0: 
						U[i,j][0] = np.linalg.norm(U[i-1,j] - U[i+1,j], axis=-1)
					else:      
						U[i,j][0] = (np.linalg.norm(U[i-1,j-1] - U[i+1,j+1], axis=-1) + np.linalg.norm(U[i+1,j-1] - U[i-1,j+1], axis=-1))/(2*np.sqrt(2))

		U = np.sum(U, axis=2) #move from Vector to Scalar

		for i in range(0, U.shape[0], 2): #count new values
			for j in range(0, U.shape[1], 2):
				region = []
				if j>0: region.append(U[i][j-1]) #check left border
				if i>0: region.append(U[i-1][j]) #check bottom
				if j<U.shape[1]-1: region.append(U[i][j+1]) #check right border
				if i<U.shape[0]-1: region.append(U[i+1][j]) #check upper border

				U[i,j] = np.median(region)
		return U

	def HitHist(self,_m, _n, _weights, _idata):
		hist = np.zeros(_m * _n)
		for vector in _idata: 
			position = np.argmin(np.sqrt(np.sum(np.power(_weights - vector, 2), axis=1)))
			hist[position] += 1

		return hist.reshape(_m, _n)

		