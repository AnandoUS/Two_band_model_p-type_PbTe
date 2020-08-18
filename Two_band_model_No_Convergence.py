import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from fdint import * 

##################### PbTe data from James's paper###################################
data=np.loadtxt('p_type_Piserenko.dat')

########################CONSTANTS####################################################
kB = 1.38064852 * 10**-23		#Boltzman Constant SI unit
e = 1.6*10**-19				#Charge on electron SI unit
h = 6.626*10**-34			#Planks Constant SI unit
pi = np.pi				#pi
m_e = 9.10938356 * 10**-31		#mass of an electron in SI unit
T = 300					#Temperature for hall-measurement in K
Efs = -1				#Starting Fermi-level
Efe = 1					#Ending Fermi-level
npts = 800				#Number of points between starting and ending Ef 

l1=0					#Acoustic Scattering for L band
l2=0					#Acoustic Scattering for Sigma band

mp1 = 0.255				#Seebeck mass of L band
mp2=2.25				#Seebeck mass of Sigma band

Eo = 0.2				#Band-offset in eV
Eg = 0.3				#Band-gap in eV

wm1 =0.002				#Weighted mobility of L band
wm2 = 0.0012				#Weighted mobility of Sigma band

###################################   MODEL     ################################################
a = (((kB*T)/e)/Eg)			#Non parabolicity parameter
c = ((kB*T)/e)				#normalizing parameter for reduced chemical potential

eta = np.array([ef/c for ef in np.linspace(Efs,Efe,npts)])


################################    Defining Kane band integrals       ################################


def integrand(x, l,m,n, Ef):
	return ((((1+2*a*x)**2+2)**(l/2))*((x+a*x**2)**m)*((x)**n))*(np.exp(x-(Ef/c))/(1+np.exp(x-(Ef/c)))**2)

F_0_15_0 = np.array([quad(integrand, 0, 350, args=(0,1.5,0,ef))[0] for ef in np.linspace(Efs,Efe,npts)])
F__2_1_1 = np.array([quad(integrand, 0, 350, args=(-2,1,1,ef))[0] for ef in np.linspace(Efs,Efe,npts)])
F__2_1_0 = np.array([quad(integrand, 0, 350, args=(-2,1,0,ef))[0] for ef in np.linspace(Efs,Efe,npts)])
F__4_05_0 = np.array([quad(integrand, 0, 350, args=(-4,0.5,0,Ef))[0] for Ef in np.linspace(Efs,Efe,npts)])


################################  Transport properties of L band   ######################################
S1 = 10**6*(kB/e)*((F__2_1_1/F__2_1_0) - eta) 
p1 = np.array((((4)*pi*(((2*mp1*m_e*kB*T)/(h**2))**1.5))*((F__2_1_0)**2/(F__4_05_0)))/(10**6))
sig1 = (8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*F__2_1_0*(wm1)

################################  Transport properties of Sigma band   ###################################
S2 = np.array([10**6*(kB/e)*( (((l2+2)*fdk(l2+1,(Ef-Eo)/((kB*T)/e)))/((l2+1)*fdk(l2,(Ef-Eo)/((kB*T)/e))))-((Ef-Eo)/((kB*T)/e)) ) for Ef in np.linspace(Efs,Efe,npts)])
p2 =np.array([((16/3)*pi*(((2*mp2*m_e*kB*T)/(h**2))**1.5)*((fdk(0,(Ef-Eo)/(c)))**2/(fdk(-0.5,(Ef-Eo)/(c)))))/(10**6) for Ef in np.linspace(Efs,Efe,npts)])	#electron carrier concentration in cm-3
sig2= np.array([(8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*(l2+1)*fdk(l2,(Ef-Eo)/((kB*T)/e))*(wm2) for Ef in np.linspace(Efs,Efe,npts)])


S = (sig1*S1+sig2*S2)/(sig1+sig2)


########################################### PLOT DETAILS #################################################
plt.rcParams['figure.figsize'] = 8, 6
plt.plot((p1+p2)/10**18,S,linewidth=1.5)
plt.plot(p1/10**18,S1,linewidth=0.4,c='indianred')
plt.plot(p2/10**18,S2,linewidth=0.4)
plt.scatter(data[:,0], data[:,1],s=50)

plt.xlim( 1,300 )
plt.ylim(0,300)

plt.xlabel('Carrier Concentration (X $10^{18} cm^{-3}$)', fontsize= 18)
plt.ylabel('Thermopower |S| ($\mu V/ K$)', fontsize =18)

plt.xscale('log')

plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)

plt.show()
