
import numpy as np
import tables
import glob

from scipy import ndimage,stats
from scipy.optimize import curve_fit

import matplotlib.cm as pltcm
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import parula #matlab colormap used in previous foil plots
pltcm.register_cmap(name="parula",cmap=parula.parula_map);

hg = [{},{}]; #histograms
ad = [{},{}]; #data arrays

gnames = ["inner","outer","blocked","defect","etching"];

convf = 4.34; #pixel to um

opt = OptionParser(usage="%prog [options]",version="%prog 0.1",description="Neo-TPCQA GEM foil plotter / database map generator.");
opt.option_list[0].help = "Print this help screen and exit";
opt.add_option("-S","--S-dir",default=None,metavar="DIR",action="store",type="string",dest="sdir",help="Source directory for the segmented side input files.");
opt.add_option("-U","--U-dir",default=None,metavar="DIR",action="store",type="string",dest="udir",help="Source directory for the unsegmented side input files.");
opt.add_option("-t","--type",default=None,metavar="TYPE",action="store",type="string",dest="type",help="Force foil type to TYPE=IROC/O1/O2/O3. By default, type is recognized from the source directory name.");
opt.add_option("-o","--output-dir",default="./",metavar="DIR",action="store",type="string",dest="outdir",help="Destination directory for the output files (report and maps) [default: %default]");
opt.add_option("-n","--name",default=None,metavar="NAME",action="store",type="string",dest="name",help="Name for the generated files. If unspecified, an attempt will be made to acquire the name from segmented side source path.");
opt.add_option("-c","--color-map",default="parula",metavar="CMAP",type="string",dest="colormap",help="Name of the colormap to use from the list of matplotlib colormaps. [default: %default]");
opt.add_option("-q","--quiet",action="store_true",dest="quiet",help="Quiet mode. Do not show plots.");
(options,args) = opt.parse_args();

if options.sdir is None:
	opt.error("S-side source unspecified.");
if options.udir is None:
	opt.error("U-side source unspecified.");

class ProgressBar():
	def __init__(self, l):
		self.l = l;
	def __enter__(self):
		return self;
	def update(self, r):
		a = np.ceil(r*self.l);
		b = np.floor(self.l-r*self.l);
		print("{:.2f}%\t|{}>{}|".format(100*r,int(a)*'=',int(b)*'-'),end='\r');
	def __exit__(self, type, val, tb):
		print("{:.2f}%\t|{}=|".format(100.0,self.l*'='));

srcdir = options.sdir.replace('\\','/'); #take the foil name from S-source
while srcdir[-1] == '/':
	srcdir = srcdir[:-1];
pathc = srcdir.split('/');

foiln = options.name;
if foiln is None:
	for p in reversed(pathc):
		if any([s in p for s in
			("I_","I-","O1_","O1-","O2_","O2-","O3_","O3-")]):
			foiln = p;
			break;
	else:
		opt.error("Unable to extract foil name. Use -n to manually specify the name.");

	for s in ["_S","_U","-s","-u","-"]:
		try:
			u = foiln.index(s);
			foiln = foiln[:u];
			break;
		except ValueError:
			pass;
if len(foiln) == 0:
	foiln = "output";

t = options.type;
if t is None:
	if any([s in foiln for s in ("I_","I-")]):
		t = "IROC";
	else:
		o = foiln.index('O');
		t = foiln[o:o+2];

try:
	nl = {
		"IROC":np.array([467.0,496.5]),
		"O1":np.array([595.8,353.0]),
		"O2":np.array([726.2,350.0]),
		"O3":np.array([867.0,379.0])
	}[t];
	nb = [int(np.ceil(n)) for n in nl];
	print("Foil/{}: assuming type {}.".format(foiln,t));
except KeyError:
	opt.error("Unrecognized foil type / invalid option -t.");

def groupCounter(hf):
	for g in hf.walk_groups():
		yield g;

print("Loading sources...");
for u,srcdir in enumerate([options.sdir,options.udir]):
	print("[{}]".format(srcdir));
	with ProgressBar(70) as pb:
		hfls = glob.glob(srcdir+"/*.h5");
		if len(hfls) == 0:
			opt.error("No source files (*.h5) found.");

		for i,hfn in enumerate(hfls):
			hf = tables.open_file(hfn,"r");

			for N,_ in enumerate(groupCounter(hf)):
				pass;

			for j,g in enumerate(hf.walk_groups()):
				try:
					data = g.data.read();

					a = ad[u].get(g._v_name);
					if a is None:
						a = {"x":np.array([]),"y":np.array([]),"d":np.array([]),"fl":np.array([])};
						ad[u][g._v_name] = a;
					a["x"] = np.concatenate((a["x"],data[:,0]));
					a["y"] = np.concatenate((a["y"],data[:,1]));
					a["d"] = np.concatenate((a["d"],data[:,7]));
					a["fl"] = np.concatenate((a["fl"],data[:,24]));

				except tables.exceptions.NoSuchNodeError:
					pass;

				pb.update((i+j/N)/len(hfls));
				
			hf.close();

print("Booking...");
with ProgressBar(70) as pb:
	for u in range(0,2):
		for i,name in enumerate(gnames):
			a = ad[u][name];
			x = a["x"];
			y = a["y"];

			pb.update(0.5*i/len(gnames)+0.5*u);

			he = {};
			hg[u][name] = he;
			try:
				he["count"],_,_ = np.histogram2d(x,y,bins=[edgesx,edgesy]);
			except NameError:
				he["count"],edgesx,edgesy = np.histogram2d(x,y,bins=nb);
			
			he["diam"],_,_,_ = stats.binned_statistic_2d(x,y,a["d"]*convf,"mean",bins=[edgesx,edgesy]);
			he["diam"][np.isnan(he["diam"])] = 0;
			he["dstd"],_,_,_ = stats.binned_statistic_2d(x,y,a["d"]*convf,"std",bins=[edgesx,edgesy]);
			he["dstd"][np.isnan(he["dstd"])] = 0;

			he["fl"],_,_,_ = stats.binned_statistic_2d(x,y,a["fl"],"mean",bins=[edgesx,edgesy]);
			he["fl"][np.isnan(he["fl"])] = 0;
			he["flstd"],_,_,_ = stats.binned_statistic_2d(x,y,a["fl"],"std",bins=[edgesx,edgesy]);
			he["flstd"][np.isnan(he["flstd"])] = 0;

		he = {};
		hg[u]["rim"] = he;
		
		he["diam"] = 0.5*(hg[u]["outer"]["diam"]-hg[u]["inner"]["diam"]);
		he["diam"][np.isnan(he["diam"])] = 0.0;
		he["dstd"] = np.sqrt(0.25*hg[u]["outer"]["dstd"]**2+0.25*hg[u]["inner"]["dstd"]**2);
		he["dstd"][np.isnan(he["dstd"])] = 0.0;

def renderFoilPlot(fig, ax, h, title, cdfrange = True, zrnan = False):
	if cdfrange:
		hf = h.flatten();
		#estimate optimal plotting range by removing peaks
		ph,bins = np.histogram(hf[hf > 0],1000,density=True);
		cdf = np.cumsum(ph)*(bins[1]-bins[0]);
		a = np.argmax(cdf > 0.05);
		b = np.argmax(cdf > 0.95);
		umin,umax = bins[a],bins[b];
	else:
		umin,umax = None,None;
	
	if zrnan:
		h[h < 1e-6] = np.nan;

	ax.set_title(title,fontsize=8);
	#ax.set_xlabel("x (mm)",fontsize=6);
	#ax.set_ylabel("y (mm)",fontsize=6);
	cmap = ax.imshow(h,interpolation="nearest",cmap=options.colormap,extent=(0,nl[0],0,nl[1]),
		vmin=umin,vmax=umax);
	ax.plot([0,63],[0,nl[1]],color="black");
	ax.plot([nl[0],nl[0]-63],[0,nl[1]],color="black");
	[t.label.set_fontsize(6) for t in ax.xaxis.get_major_ticks()];
	[t.label.set_fontsize(6) for t in ax.yaxis.get_major_ticks()];
	cbar = fig.colorbar(cmap,ax=ax);
	cbar.ax.tick_params(labelsize=6);

	ax.autoscale(False);

def renderPointOverlay(ax, x, y):
	x = nl[1]*(x-edgesx[0])/(edgesx[-1]-edgesx[0]);
	y = nl[0]*(y-edgesy[0])/(edgesy[-1]-edgesy[0]);
	ax.plot(y,nl[1]-x,marker="o",linestyle="none",color="red",mfc="none");

def gauss(x, *p):
	A,mu,sigma = p;
	return A*np.exp(-(x-mu)**2/(2.0*sigma*sigma));

options.outdir += "/";
print("Generating output... (dst: {})".format(options.outdir));
with PdfPages(options.outdir+foiln+".pdf") as pdf:

	fig,ax = plt.subplots(4,2,figsize=(0.707*8,8));
	for i,r in enumerate([
		("inner","diam","Inner Diameter (S)"),("inner","diam","Inner Diameter (U)"),
		("inner","dstd","Inner Diameter Std. Dev (S)"),("inner","dstd","Inner Diameter Std. Dev (U)"),
		("outer","diam","Outer Diameter (S)"),("outer","diam","Outer Diameter (U)"),
		("outer","dstd","Outer Diameter Std. Dev (S)"),("outer","dstd","Outer Diameter Std. Dev (U)")]):
		h = hg[i%2][r[0]][r[1]];
		q = np.unravel_index(i,(4,2));
		renderFoilPlot(fig,ax[q],h,r[2]);
	plt.savefig(pdf,format="pdf");

	fig,ax = plt.subplots(4,2,figsize=(0.707*8,8));
	for i,r in enumerate([
		("inner","count","Inner N (S)",True),("inner","count","Inner N (U)",True),#("outer","count","Outer N"),
		("defect","count","Defect N (S)",False),("defect","count","Defect N (U)",False),
		("etching","count","Etching N (S)",False),("etching","count","Etching N (U)",False),
		("blocked","count","Blocked N (S)",False),("blocked","count","Blocked N (U)",False)]):
		h = hg[i%2][r[0]][r[1]];
		q = np.unravel_index(i,(4,2));
		renderFoilPlot(fig,ax[q],h,r[2],r[3],not r[3]);
		if not r[3]:
			renderPointOverlay(ax[q],ad[i%2][r[0]]["x"],ad[i%2][r[0]]["y"]);
	plt.savefig(pdf,format="pdf");

	fig,ax = plt.subplots(4,2,figsize=(0.707*8,8));
	for i,r in enumerate([
		("rim","diam","Rim Diameter (S)"),("rim","diam","Rim Diameter (U)"),
		("rim","dstd","Rim Diameter Std. Dev (S)"),("rim","dstd","Rim Diameter Std. Dev (U)"),
		("inner","fl","Foreground Light (S)"),("inner","fl","Foreground Light (U)"),
		("inner","flstd","Foreground Light Std. Dev (S)"),("inner","flstd","Foreground Light Std. Dev (U)")]):
		h = hg[i%2][r[0]][r[1]];
		q = np.unravel_index(i,(4,2));
		renderFoilPlot(fig,ax[q],h,r[2]);
	plt.savefig(pdf,format="pdf");

	#m = np.fromfunction(lambda i,j: np.vectorize(Get)(i,j),(x,y),dtype=float);
	#h = hg[name]["diam"];
	#labels,nlabels = ndimage.label(h > 1e-6);
	#labelr = ndimage.sum(h,labels,range(nlabels+1));

	#h[(labelr < 1000)[labels]] = 0.0;
	#h = h[~np.all(h == 0,axis=1)][:,~np.all(h == 0,axis=0)];

	#remove disconnected artifacts
	#h[(labelr < 100)[labels]] = np.nan;
	#q1 = np.bitwise_or(np.isnan(h),h == 0.0);
	#q2 = np.bitwise_and(~np.all(q1,axis=1),np.any(np.isnan(h),axis=1));
	#q3 = np.bitwise_and(~np.all(q1,axis=0),np.any(np.isnan(h),axis=0));
	#h = h[q2][:,q3];
	#h[np.isnan(h)] = 0.0;

	#remove connected artifacts
	#h = h[~np.bitwise_and(np.any(h > 15,axis=1),np.count_nonzero(h == 0.0,axis=1)/y > 0.9)];
	#h = h[:,~np.bitwise_or(
	#	np.bitwise_and(np.any(h > 15,axis=0),np.count_nonzero(h == 0.0,axis=0)/x > 0.9),
	#	np.all(h == 0.0,axis=0))];
	#while True:
	#	n = len(h);
	#	h = h[~np.bitwise_and(h[:,0] > 1e-6,np.count_nonzero(h == 0.0,axis=1)/y > 0.9)];
	#	if len(h) == n:
	#		break;
	#	h = h[:,~np.all(h == 0.0,axis=0)];

	yi,bins_i = [None]*2,[None]*2;
	yo,bins_o = [None]*2,[None]*2;
	yr,bins_r = [None]*2,[None]*2;
	fig,ax = plt.subplots(2,1,sharey=True,figsize=(0.707*8,8));
	#fig.subplots_adjust(wspace=0.0,hspace=0.0);
	for u in range(0,2):
		yi[u],bins_i[u],_ = ax[u].hist(ad[u]["inner"]["d"]*convf,300,range=(0,110),alpha=0.5);
		yo[u],bins_o[u],_ = ax[u].hist(ad[u]["outer"]["d"]*convf,300,range=(0,110),alpha=0.5);

		rh = 0.5*(hg[u]["outer"]["diam"]-hg[u]["inner"]["diam"]); #rim histogram
		rh[np.isnan(rh)] = 0.0;
		rh = rh.flatten();

		yr[u],bins_r[u] = np.histogram(rh,300,range=(1,110));
		w = min(np.max(yi[u]),np.max(yo[u]))/np.max(yr[u]);
		#ax.bar(0.5*(bins[:-1]+bins[1:]),y*w,align="center",color="black",alpha=0.3);
		ax[u].hist(rh,300,range=(1,110),weights=np.full(len(rh),w),alpha=0.5);

		ax[u].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
		ax[u].set_title("Diameter Profile ("+["S","U"][u]+")");
	
	ax[1].set_xlabel("Diameter ($\\mathrm{\\mu m}$)");
	#ax.set_ylabel("Occurrence");
	plt.savefig(pdf,format="pdf");

	#binc = 0.5*(bins[:-1]+bins[1:]);
	#binc = binc[y > 10];
	#y = y[y > 10];
	#coeff,cov = curve_fit(gauss,binc,y,p0=[1,0,1000]);
	#ax.plot(binc,t);
	#ax.plot(binc,gauss(binc,*coeff));

binc = 0.5*(bins_i[0][:-1]+bins_i[0][1:]);
with open(options.outdir+foiln+"_profile.txt","w") as pf:
	for i,_ in enumerate(binc):
		pf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
			.format(binc[i],yr[0][i],yr[1][i],yi[0][i],yi[1][i],yo[0][i],yo[1][i]));

with open(options.outdir+foiln+"_map.txt","w") as mf:
	for (i,j),_ in np.ndenumerate(hg[0]["inner"]["count"].T):
		mf.write("{}\t{}\t\
			{}\t{}\t{}\t{}\t{}\
			{}\t{}\t{}\t{}\t{}\
			{}\t{}\t{}\t{}\t{}\
			{}\t{}\t{}\t{}\t{}\
			{}\t{}\t{}\t{}\t\
			{}\t{}\n"
			.format(float(i)*nl[0]/nb[0]+0.5/nl[0],float(j)*nl[0]/nb[0]+0.5/nl[0],
			hg[0]["inner"]["diam"][j][i],hg[0]["outer"]["diam"][j][i],
			hg[0]["blocked"]["diam"][j][i],hg[0]["defect"]["diam"][j][i],hg[0]["etching"]["diam"][j][i],
			hg[1]["inner"]["diam"][j][i],hg[1]["outer"]["diam"][j][i],
			hg[1]["blocked"]["diam"][j][i],hg[1]["defect"]["diam"][j][i],hg[1]["etching"]["diam"][j][i],
			hg[0]["inner"]["count"][j][i],hg[0]["outer"]["count"][j][i],
			hg[0]["blocked"]["count"][j][i],hg[0]["defect"]["count"][j][i],hg[0]["etching"]["count"][j][i],
			hg[1]["inner"]["count"][j][i],hg[1]["outer"]["count"][j][i],
			hg[1]["blocked"]["count"][j][i],hg[1]["defect"]["count"][j][i],hg[1]["etching"]["count"][j][i],
			hg[0]["inner"]["dstd"][j][i],hg[0]["outer"]["dstd"][j][i],hg[1]["inner"]["dstd"][j][i],hg[1]["outer"]["dstd"][j][i],
			hg[0]["inner"]["fl"][j][i],hg[1]["inner"]["fl"][j][i]));

if not options.quiet:
	plt.show();

