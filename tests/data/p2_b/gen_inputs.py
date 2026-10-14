import numpy as np, scipy.io as sio, json
from pyhctsa.utils import z_score
X=np.load('/Users/benfulcher/Sydney Uni Dropbox/Ben Fulcher/hctsa/Empirical1000Dataset/Empirical_Synthetic1000/data/stationary1000/figshare_1000x1000/series.npz',allow_pickle=True)['X']
S={}
for k,i in enumerate(range(0,1000,25)): S[f's{k}']=z_score(X[i])
rng=np.random.default_rng(11)
from scipy.ndimage import uniform_filter1d
S['rw537']=z_score(np.cumsum(rng.standard_normal(537)))
S['wn1213']=z_score(rng.standard_normal(1213))
S['short60']=z_score(rng.standard_normal(60))
S['short100']=z_score(np.cumsum(rng.standard_normal(100)))
S['smooth800']=z_score(uniform_filter1d(rng.standard_normal(800),25))
S['sin500']=z_score(np.sin(np.arange(500)*0.07)+0.1*rng.standard_normal(500))
S['quant600']=z_score(np.round(2*rng.standard_normal(600))+0.01*np.arange(600)/600)
S['plateau400']=z_score(np.repeat(rng.standard_normal(100),4))
S['const300']=np.ones(300)
S['const_half']=np.r_[np.zeros(150),rng.standard_normal(150)]
S['expn700']=z_score(rng.exponential(size=700))
S['lognorm900']=z_score(np.exp(rng.standard_normal(900)))
S['cauchy500']=z_score(rng.standard_cauchy(500))
S['ar2_1000']=z_score(np.convolve(rng.standard_normal(1100),[1,.9,.5,.2],'valid')[:1000])
S['sin3_800']=z_score(np.sin(0.05*2*np.pi*np.arange(800))+0.5*np.sin(0.13*2*np.pi*np.arange(800))+0.3*np.sin(0.21*2*np.pi*np.arange(800))+0.3*rng.standard_normal(800))
# raw (unnormalised) positive-mean series
S['posmean500']=5+rng.standard_normal(500)
S['rawexp600']=rng.exponential(2,600)+1
names=list(S)
sio.savemat('inputs.mat',{'names':np.array(names,dtype=object),**{n:S[n] for n in names}})
json.dump(names,open('names.json','w')); print(len(names))
