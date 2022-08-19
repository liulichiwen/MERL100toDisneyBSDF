#统计行数-均匀间隔选15个
import codecs
import numpy as np
import math

def HSV2RGB(H,S,V):
    
    global R,G,B
    
    C = V * S
    X = C * (1- abs(((H/60)%2) -1 ))
    m = V - C

    if H < 60:
        R_ = C
        G_ = X
        B_ = 0
    elif (H>=60) and (H<120):
        R_ = X
        G_ = C
        B_ = 0
    elif (H>=120) and (H<180):
        R_ = 0
        G_ = C
        B_ = X
    elif (H>=180) and (H<240):
        R_ = 0
        G_ = X
        B_ = C
    elif (H>=240) and (H<300):
        R_ = X
        G_ = 0
        B_ = C
    else:
        R_ = C
        G_ = 0
        B_ = X

    R = (R_ + m )
    G = (G_ + m )
    B = (B_ + m )

    return R,G,B

result = []
file  =  open  ("fruitwood-241-all.txt",  "r")
lines = len(file.readlines())
#print(lines)
n0 = 125
if int((lines - n0) / 30)%2 == 0:
    gap = int((lines - n0) / 30)
else:
    gap = int((lines - n0) / 30 - 1)
file.close()
#print(gap)

lines_result_ = []
for x in range(15):
    with codecs.open('./fruitwood-241-all.txt', 'r', 'gb18030') as infile:
        m = infile.readlines()[int(n0) + int(x*gap)]
        #print((m))
        lines_result_.append(m)
        #lines_cal.append(m[0])
#print(lines_result_)
lines_result = []
for r in lines_result_:
    r_ = r.replace('\r\n','').replace(' ',',')
    r__ = r_.split(',')
    r__ = np.float_(r__)
    rNew = np.array(r__)
    lines_result.append((rNew))
#print(lines_result)


lines_cal_ = []
for x in range(15):
    with codecs.open('./fruitwood-241-all.txt', 'r', 'gb18030') as infile:
        n = infile.readlines()[int(n0) + int(x*gap) -1]
        #print((n))
        lines_cal_.append(n)
#print(lines_cal_)
lines_cal = []
for f in lines_cal_:
    f__ = f.split(',')
    #print(len(f__))
    for z in f__[0:3]:
        #print(type(z))
        z_ = z.replace('[','').replace(' ', ',').replace(']','').replace('\r\n','')
        z__ = z_.split(',')
        if(len(z__) == 4):
           z__ = z__[1::]
    #    print(z__)
        z__ = np.float_(z__)
        f__.append(z__)
        fnew = f__[3::]
    #print(fnew)
    lines_cal.append((fnew))
#print(lines_cal)

from sympy import *
import sympy

def SchlickFresnel(u):
    yy = [1-u, 0, 1]
    yy.sort()
    m = yy[1]
#    m = clamp(1-u, 0, 1);
    m2 = m*m
    return m2*m2*m # pow(m,5)

#算法改写
#math.sqrt→sqrt
#log10()→log()/log10
#baseColor[0]→baseColor0

#需求解的参数
roughness = symbols('roughness')
anistrophic = symbols('anistrophic')
specular = symbols('specular')
specularTint = symbols('specularTint')
baseColor0 = symbols('baseColor0')
baseColor1 = symbols('baseColor1')
baseColor2 = symbols('baseColor2')
metalic = symbols('metalic')
subsurface = symbols('subsurface')
sheen = symbols('sheen')
sheenTint = symbols('sheenTint')
clearCoat = symbols('clearCoat')
clearCoatGloss = symbols('clearCoatGloss')

ax = symbols('ax')#max(.001, (roughness)**2/math.sqrt(1-anistrophic*.9))
ay = symbols('ay')#max(.001, (roughness)**2*math.sqrt(1-anistrophic*.9))
Cdlum = symbols('Cdlum')#(0.3*Cdlin[0] + .6*Cdlin[1] + .1*Cdlin[2])>0?Cdlin/Cdlum:np.array([1,1,1])

#需替换 .subs()
PI = symbols('PI')
NdotL = symbols('NdotL')
LdotX = symbols('LdotX')
LdotY = symbols('LdotY')
NdotV = symbols('NdotV')
VdotX = symbols('VdotX')
VdotY = symbols('VdotY')
HdotX = symbols('HdotX')
HdotY = symbols('HdotY')
NdotH = symbols('NdotH')
LdotH = symbols('LdotH')
FL = symbols('FL')
FV = symbols('FV')
FH = symbols('FH')


BSDF0 = (ax**3*ay**3/(PI*(NdotL + sqrt(LdotX**2*ax**2 + LdotY**2*ay**2 + NdotL**2))*(NdotV + sqrt(NdotV**2 + VdotX**2*ax**2 + VdotY**2*ay**2))*(HdotX**2*ay**2 + HdotY**2*ax**2 + NdotH**2*ax**2*ay**2)**2))*((specular*.08*((1-specularTint)+baseColor0**2.2/Cdlum*specularTint)*(1-metalic) + baseColor0**2.2*metalic)*(1-FH)+FH) + .25*clearCoat*(((.1*(1-clearCoatGloss)+.001*clearCoatGloss)**2-1) / (PI*sympy.log((.1*(1-clearCoatGloss)+.001*clearCoatGloss)**2)/sympy.log(10)*(1 + ((.1*(1-clearCoatGloss)+.001*clearCoatGloss)**2-1)*NdotH*NdotH)))*( .04*(1-FH) + 1*FH)*(1 / (NdotL + sqrt(0.25*0.25 + NdotL*NdotL - 0.25*0.25*NdotL*NdotL))* 1 / (NdotV + sqrt(0.25*0.25 + NdotV*NdotV - 0.25*0.25*NdotV*NdotV)))+ (1-metalic) * (( (1/PI) * (((1-FL)+(0.5 + 2 * LdotH*LdotH * roughness) * FL) * ((1-FV)+(0.5 + 2 * LdotH*LdotH * roughness) * FV) * (1-subsurface)+ (1.25 * ((1-FL+LdotH*LdotH*roughness*FL) * (1-FV+LdotH*LdotH*roughness*FV) * (1 / (NdotL + NdotV) - .5) + .5)) * subsurface)*baseColor0**2.2)+(FH * sheen * ((1-sheenTint)+baseColor0**2.2/Cdlum * sheenTint)))

equation = []
solution = []
merl_fix = 2000
light_fix = 600
for x in range(14):
    cals = lines_cal[x]
    N = cals[2]
    H = cals[2]
    L = cals[0]
    V = cals[1]
    X = np.array([1,0,0])
    Y = np.array([0,1,0])
    Z = np.array([0,0,1])

    PI_ = 3.14159265358979323846
    NdotL_ = np.dot(N,L)
    LdotX_ = np.dot(L,X)
    LdotY_ = np.dot(L,Y)
    NdotV_ = np.dot(N,V)
    VdotX_ = np.dot(V,X)
    VdotY_ = np.dot(V,Y)
    HdotX_ = np.dot(H,X)
    HdotY_ = np.dot(H,Y)
    NdotH_ = np.dot(N,H)
    LdotH_ = np.dot(L,H)
    FL_ = SchlickFresnel(NdotL_)
    FV_ = SchlickFresnel(NdotV_)
    FH_ = SchlickFresnel(LdotH_)
    
    BSDF0 = BSDF0.subs(PI,PI_).subs(NdotL,NdotL_).subs(LdotX,LdotX_).subs(LdotY,LdotY_).subs(NdotV,NdotV_).subs(VdotX,VdotX_).subs(VdotY,VdotY_).subs(HdotX,HdotX_).subs(HdotY,HdotY_).subs(NdotH,NdotH_).subs(LdotH,LdotH_).subs(FL,FL_).subs(FV,FV_).subs(FH,FH_)
    BSDF0 = BSDF0 - lines_result[x][0]*merl_fix/light_fix/255
    equation.append(BSDF0)
    #print((FV_))
#print((equation[1]))

f0 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[0],'numpy')
f1 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[1],'numpy')
f2 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[2],'numpy')
f3 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[3],'numpy')
f4 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[4],'numpy')
f5 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[5],'numpy')
f6 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[6],'numpy')
f7 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[7],'numpy')
f8 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[8],'numpy')
f9 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[9],'numpy')
f10 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[10],'numpy')
f11 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[11],'numpy')
f13 = lambdify([baseColor0,baseColor1,baseColor2,metalic,subsurface,specular,specularTint, roughness,anistrophic,sheen,sheenTint,clearCoat,clearCoatGloss,ax,ay,Cdlum],equation[13],'numpy')
#yyy = f(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
#print(yyy)

#yyy = o
import numpy as np
import math

def normalize(x):
    return x/np.linalg.norm(x)

def SchlickFresnel(u):
    yy = [1-u, 0, 1]
    yy.sort()
    m = yy[1]
#    m = clamp(1-u, 0, 1);
    m2 = m*m
    return m2*m2*m # pow(m,5)

def sqrt(x):
    return math.sqrt(x)

def log(x):
    return math.log(x)

H = 29
S = np.random.uniform(0,1,1000000)
V = np.random.uniform(0,1,1000000)
metalic = np.random.uniform(0,1,1000000)
subsurface = np.random.uniform(0,1,1000000)
specular = np.random.uniform(0,1,1000000)
specularTint = np.random.uniform(0,1,1000000)
roughness = np.random.uniform(0,1,1000000)
anistrophic = np.random.uniform(0,1,1000000)
sheen = np.random.uniform(0,1,1000000)
sheenTint = np.random.uniform(0,1,1000000)
clearCoat = np.random.uniform(0,1,1000000)
clearCoatGloss = np.random.uniform(0,1,1000000)



result_all = []
ran = 0.00140

for i in range(1000000):
    HSV2RGB(H,S[i],V[i])
    baseColor0 = R
    baseColor1 = G
    baseColor2 = B
    
    ax_ = max(.001, (roughness[i])**2/math.sqrt(1-anistrophic[i]*.9))
    ay_ = max(.001, (roughness[i])**2*math.sqrt(1-anistrophic[i]*.9))
    Cdlum_ = 0.3*baseColor0**2.2 + 0.6*baseColor1**2.2 + 0.1*baseColor2**2.2

    BSDF0 = f0(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF1 = f1(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF2 = f2(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF3 = f3(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF4 = f4(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF5 = f5(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF6 = f6(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF7 = f7(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF8 = f8(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF9 = f9(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF10 = f10(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)
    BSDF11 = f11(baseColor0,baseColor1,baseColor2,metalic[i],subsurface[i],specular[i],specularTint[i], roughness[i],anistrophic[i],sheen[i],sheenTint[i],clearCoat[i],clearCoatGloss[i],ax_,ay_,Cdlum_)


    if (BSDF0 > -ran) and (BSDF0 < ran) and (BSDF1 < ran) and (BSDF1 > -ran) and (BSDF2 < ran) and (BSDF2 > -ran) and (BSDF3 < ran) and (BSDF3 > -ran) and (BSDF4 < ran) and (BSDF4 < ran) and (BSDF5 > -ran) and (BSDF5 < ran) and (BSDF6 > -ran) and (BSDF6 < ran) and (BSDF7 > -ran) and (BSDF7 < ran) and (BSDF8 < ran) and (BSDF8 > -ran) and (BSDF9 < ran) and (BSDF10 < ran) and (BSDF10 > -ran) and (BSDF11 < ran):
        result = []
        result.append(baseColor0)
        result.append(baseColor1)
        result.append(baseColor2)
        result.append(metalic[i])
        result.append(subsurface[i])
        result.append(specular[i])
        result.append(specularTint[i])
        result.append(roughness[i])
        result.append(anistrophic[i])
        result.append(sheen[i])
        result.append(sheenTint[i])
        result.append(clearCoat[i])
        result.append(clearCoatGloss[i])
        result_all.append(result)
    else:
        pass
print(len(result_all))
print(result_all)

