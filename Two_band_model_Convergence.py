import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from fdint import * 

#####################             PbTe data from Pei's paper            ###################################

data=np.loadtxt('p_type_Piserenko.dat')

########################               CONSTANTS                        ###################################

kB = 1.38064852 * 10**-23		#Boltzman Constant SI unit
e = 1.6*10**-19				#Charge on electron SI unit
h = 6.626*10**-34			#Planks Constant SI unit
pi = np.pi				#pi
m_e = 9.10938356 * 10**-31		#mass of an electron in SI unit
T = 300					#Temperature for hall-measurement in K
v = 67/10**24				#Volume of a primitive unit-cell in cm-3
Efs = -0.5				#Starting Fermi-level
Efe = 0.5				#Ending Fermi-level
npts = 1000				#Number of points between starting and ending Ef 

#######################		Electronic Structure parameters		###################################
wm1 = 0.002 				#weighted mobility of L band
wm2 = 0.0012				#weighted mobility of Sigma band

mp_1 = 0.26				#seebeck mass of L band
mp_2 = 2.25				#seebeck mass of Sigma band

Ego = 0.3				#band-gap of undoped PbTe
Eo = 0.2				#band-offset of undoped PbTe

###################################        MODEL           ################################################

Efermi = np.array([x for x in np.linspace(Efs,Efe,npts)])

c = ((kB*T)/e)
eta = np.array([ef/c for ef in np.linspace(Efs,Efe,npts)])
x = np.array([x for x in np.linspace(0,350,npts)])

def integrand(x, l,m,n, Ef,a):
	return ((((1+2*a*x)**2+2)**(l/2))*((x+a*x**2)**m)*((x)**n))*(np.exp(x-(Ef/c))/(1+np.exp(x-(Ef/c)))**2)

npts2=20
p=np.array([p for p in np.linspace(1,300,npts2)])

###########################       Uncompensated Na-doping model         ####################################

S=[]

for i in p:
	j=Eo-0.000110095*i

	Eg = Ego + 2*(Eo-j)								#Band-gap for this Band-offset 
	a = (((kB*T)/e)/Eg)								#Non-parabolicity factor for this Band-offset

	mp1 = mp_1 - 1.7333 * (j-Eo)							#Kane band-mass for this Band-offset
	mp2 = mp_2 - 7.052 * (j-Eo)							#SPB mass for this Band-offset

	F__2_1_0 = np.array([quad(integrand, 0, 350, args=(-2,1,0,Ef,a))[0] for Ef in np.linspace(Efs,Efe,npts)])
	F__4_05_0 = np.array([quad(integrand, 0, 350, args=(-4,0.5,0,Ef,a))[0] for Ef in np.linspace(Efs,Efe,npts)])

	p1 = np.array((((4)*pi*(((2*mp1*m_e*kB*T)/(h**2))**1.5))*((F__2_1_0)**2/(F__4_05_0)))/(10**6))
	p2 = np.array([((16/3)*pi*(((2*mp2*m_e*kB*T)/(h**2))**1.5)*((fdk(0,(Ef-j)/(c)))**2/(fdk(-0.5,(Ef-j)/(c)))))/(10**6) for Ef in np.linspace(Efs,Efe,npts)])	#electron carrier concentration in cm-3

	k=np.argmin(np.abs(((p1+p2)/10**18)-i))

	F__2_1_1 = quad(integrand, 0, 350, args=(-2,1,1,Efermi[k],a))[0]
	F__2_1_0 = quad(integrand, 0, 350, args=(-2,1,0,Efermi[k],a))[0]
	S1 = 10**6*(kB/e)*((F__2_1_1/F__2_1_0) - eta[k]) 
	sig1 = (8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*F__2_1_0*(wm1)

	sig2= (8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*fdk(0,(Efermi[k]-j)/(c))*(wm2)
	S2 = 10**6*(kB/e)*( (((2)*fdk(1,(Efermi[k]-j)/(c)))/(fdk(0,(Efermi[k]-j)/(c))))-((Efermi[k]-j)/(c)) )

	S.append((sig1*S1+sig2*S2)/(sig1+sig2))

S=np.array(S)

###########################       Compensated Na-doping model         #######################################

S_1=[]

for i in p:
	j=Eo-0.000277*i	

	Eg = Ego+2*(Eo-j)								#Band-gap for this Band-offset 
	a = (((kB*T)/e)/Eg)								#Non-parabolicity factor for this Band-offset

	mp1 = mp_1 - 1.7333 * (j-Eo)							#Kane band-mass for this Band-offset
	mp2 = mp_2 - 7.052 * (j-Eo)							#SPB mass for this Band-offset

	F__2_1_0 = np.array([quad(integrand, 0, 350, args=(-2,1,0,Ef,a))[0] for Ef in np.linspace(Efs,Efe,npts)])
	F__4_05_0 = np.array([quad(integrand, 0, 350, args=(-4,0.5,0,Ef,a))[0] for Ef in np.linspace(Efs,Efe,npts)])

	p1 = np.array((((4)*pi*(((2*mp1*m_e*kB*T)/(h**2))**1.5))*((F__2_1_0)**2/(F__4_05_0)))/(10**6))
	p2 = np.array([((16/3)*pi*(((2*mp2*m_e*kB*T)/(h**2))**1.5)*((fdk(0,(Ef-j)/(c)))**2/(fdk(-0.5,(Ef-j)/(c)))))/(10**6) for Ef in np.linspace(Efs,Efe,npts)])	#electron carrier concentration in cm-3

	k=np.argmin(np.abs(((p1+p2)/10**18)-i))

	F__2_1_1 = quad(integrand, 0, 350, args=(-2,1,1,Efermi[k],a))[0]
	F__2_1_0 = quad(integrand, 0, 350, args=(-2,1,0,Efermi[k],a))[0]
	S1 = 10**6*(kB/e)*((F__2_1_1/F__2_1_0) - eta[k]) 
	sig1 = (8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*F__2_1_0*(wm1)

	sig2= (8/3)*np.pi*e*((2*m_e*kB*T)/(h**2))**1.5*fdk(0,(Efermi[k]-j)/(c))*(wm2)
	S2 = 10**6*(kB/e)*( (((2)*fdk(1,(Efermi[k]-j)/(c)))/(fdk(0,(Efermi[k]-j)/(c))))-((Efermi[k]-j)/(c)) )

	S_1.append((sig1*S1+sig2*S2)/(sig1+sig2))

S_1=np.array(S_1)

########################################## HARD CODING DATA ############################################

# Tan just 2% Na doped, in case you want to include with plain Na-doped samples
Tan002 = np.array([[0.02, 1.36796E+20, 62.11031175]])

# Tan SrTe alloy. All 2% Na doped
TanSrTe = np.array([[0.02, 1.4455E+2, 72.1822542],
                    [0.02, 1.59437E+2, 91.36690647],
                    [0.02, 1.66423E+2, 80.33573141],
                    [0.02, 1.69509E+2, 93.2853717]])

# Our samples, columns are: nominal Na, nH, S
PJ = np.array([[0.01, 5.98264E+1, 63.30935252],
        [0.02, 1.29458E+2, 63.30935252],
        [0.03, 1.29458E+2, 74.34052758],
        [0.035, 1.63893E+2, 73.14148681],
        [0.04, 1.85256E+2, 81.53477218]])

########################################### PLOT DETAILS #################################################
# Marker size
ms=50

plt.scatter(PJ[0][1], PJ[0][2], marker='o', color='#EE3080',s=ms)
plt.scatter(PJ[1][1], PJ[1][2], marker='^', color='#8193CA',s=ms+15)
plt.scatter(PJ[2][1], PJ[2][2], marker='D', color='#70C6A5',s=ms)
plt.scatter(PJ[3][1], PJ[3][2], marker='s', color='#0F9347',s=ms)
plt.scatter(PJ[4][1], PJ[4][2], marker='*', color='#21459C',s=ms+30)

plt.scatter(TanSrTe[:,1], TanSrTe[:,2], marker='o',facecolor='white',edgecolor='#F47D30', s=ms)

plt.rcParams['figure.figsize'] = 8, 6

plt.plot(p,S,linewidth=0.5)
plt.plot(p,S_1,linewidth=1)
plt.scatter(data[:,0], data[:,1],s=7)
#plt.scatter(data1[:,1], data1[:,2],s=7)

plt.xlim( 1,300 )
plt.ylim(0,300)

plt.xlabel('Carrier Concentration (X $10^{18} cm^{-3}$)', fontsize= 18)
plt.ylabel('Thermopower |S| ($\mu V/ K$)', fontsize =18)

plt.xscale('log')

plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)

plt.show()
