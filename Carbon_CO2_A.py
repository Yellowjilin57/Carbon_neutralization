# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 15:36:18 2021

Who would be so cruel to someone like you?
No one but you
Who would make the rules for the things that you do?
No one but you
I woke up this morning, looked at the sky
I thought of all of the time passing by
It didn't matter how hard we tried
'Cause in the end

@author: KING
"""
#%%  主函数
from sympy import symbols, solve
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as logitm
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import statsmodels.regression.linear_model as lm_
import statsmodels.robust.robust_linear_model as roblm_
import statsmodels.regression.quantile_regression as QREG
from statsmodels.stats.outliers_influence import summary_table
import itertools
def log_A(x):## log10 以10为底数
    xs = math.log(x,10)
    return xs
chla = symbols('chla')
SA = symbols('SA')
depth = symbols('depth')
WT = symbols('WT')


def log_series(x):## log10 以10为底数
    ou = []
    for ir in range(0,len(x)):
        try:
            xs = math.log(x[ir],10)
            ou.append(xs)
        except:
            xs = np.nan
            ou.append(xs)
    oui = pd.Series(ou)
    return oui 

def FuntOUT(olspp,ols_r,TYPE,XZXsp,zxqj,UTR):
    if TYPE == 'OLS':
        result_XianZHU = ols_r.pvalues
        result_params = ols_r.params
        result_conf = ols_r.conf_int(zxqj)
        resultCB = pd.concat([result_XianZHU, result_params, result_conf  ],axis=1)
        resultCB.columns = ['p','params','conf_left','conf_right']
        XZxiang = resultCB[resultCB['p']<=XZXsp]
        functio_XZ = XZxiang.index.tolist()
        model_r2 = np.round(ols_r.rsquared,3)
        model_r2_rejusted = np.round(ols_r.rsquared_adj,3)
        model_aic = np.round(ols_r.aic,3)
        model_bic = np.round(ols_r.bic,3)
        st, rssew, ss2 = summary_table(ols_r, alpha= 0.05)
        predict_mean_ci_low, predict_mean_ci_upp = rssew[:, 4:6].T
        conf_left_p = sm.OLS(predict_mean_ci_low, logbl).fit() 
        conf_right_p = sm.OLS(predict_mean_ci_upp, logbl).fit() 
        baseline = TRT(resultCB,functio_XZ,1,UTR)
        
        
        result_XianZHU_CONF = conf_left_p.pvalues
        result_params_CONF = conf_left_p.params
        result_conf_CONF = conf_left_p.conf_int(zxqj)
        resultCB_CONF = pd.concat([result_XianZHU_CONF, result_params_CONF, result_conf_CONF  ],axis=1)
        resultCB_CONF.columns = ['p','params','conf_left','conf_right']
        XZxiang_CONF = resultCB_CONF[resultCB_CONF['p']<=XZXsp]
        functio_XZ_CONF = XZxiang_CONF.index.tolist()
        conf_left = TRT(resultCB_CONF,functio_XZ_CONF,1,UTR)
        result_XianZHU_CONF2 = conf_right_p.pvalues
        result_params_CONF2 = conf_right_p.params
        result_conf_CONF2 = conf_right_p.conf_int(zxqj)
        resultCB_CONF2 = pd.concat([result_XianZHU_CONF2, result_params_CONF2, result_conf_CONF2 ],axis=1)
        resultCB_CONF2.columns = ['p','params','conf_left','conf_right']
        XZxiang_CONF2 = resultCB_CONF2[resultCB_CONF2['p']<=XZXsp]
        functio_XZ_CONF2 = XZxiang_CONF2.index.tolist()
        conf_right = TRT(resultCB_CONF2,functio_XZ_CONF2,1,UTR)
        ALL_func = resultCB
        functio_ALL = ALL_func.index.tolist()
        quanbuFUNCTION = TRT( ALL_func,functio_ALL ,1,UTR)
        WE_res = pd.DataFrame(list(['('+str(quanbuFUNCTION)+')','('+str(baseline)+')','('+str(conf_left)+')','('+str(conf_right)+')',model_r2,model_r2_rejusted,model_aic,model_bic])).T
        WE_res.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic']

    if TYPE == 'MID':
        result_XianZHU = ols_r.pvalues
        result_params = ols_r.params
        result_conf = ols_r.conf_int(zxqj)
        resultCB = pd.concat([result_XianZHU, result_params, result_conf  ],axis=1)
        resultCB.columns = ['p','params','conf_left','conf_right']
        XZxiang = resultCB[resultCB['p']<=XZXsp]
        functio_XZ = XZxiang.index.tolist()
        model_r2 = np.round(ols_r.prsquared,3)
        model_r2_rejusted = np.nan
        model_aic = np.nan
        model_bic = np.nan
        baseline = TRT(resultCB,functio_XZ,1,UTR)
        conf_left = TRT(resultCB,functio_XZ,2,UTR)
        conf_right = TRT(resultCB,functio_XZ,3,UTR)
        ALL_func = resultCB
        functio_ALL = ALL_func.index.tolist()
        quanbuFUNCTION = TRT( ALL_func,functio_ALL ,1,UTR)
        WE_res = pd.DataFrame(list(['('+str(quanbuFUNCTION)+')','('+str(baseline)+')','('+str(conf_left)+')','('+str(conf_right)+')',model_r2,model_r2_rejusted,model_aic,model_bic])).T
        WE_res.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic']
    if TYPE == 'RLM':   
            result_XianZHU = ols_r.pvalues
            result_params = ols_r.params
            result_conf = ols_r.conf_int(zxqj)
            resultCB = pd.concat([result_XianZHU, result_params, result_conf  ],axis=1)
            resultCB.columns = ['p','params','conf_left','conf_right']
            XZxiang = resultCB[resultCB['p']<=XZXsp]
            functio_XZ = XZxiang.index.tolist()
            model_r2 = np.round(olspp.rsquared,3)
            model_r2_rejusted = np.nan
            model_aic = np.nan
            model_bic = np.nan
            baseline = TRT(resultCB,functio_XZ,1,UTR)
            conf_left = TRT(resultCB,functio_XZ,2,UTR)
            conf_right = TRT(resultCB,functio_XZ,3,UTR)
            ALL_func = resultCB
            functio_ALL = ALL_func.index.tolist()
            quanbuFUNCTION = TRT( ALL_func,functio_ALL ,1,UTR)
            WE_res = pd.DataFrame(list(['('+str(quanbuFUNCTION)+')','('+str(baseline)+')','('+str(conf_left)+')','('+str(conf_right)+')',model_r2,model_r2_rejusted,model_aic,model_bic])).T
            WE_res.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic']
    return baseline,conf_left,conf_right,WE_res


def Trans(eq):
    if eq == 'chla' :
        globals()[eq] = chla
        return chla
    if eq == 'SA' :
        globals()[eq] = SA
        return SA
    if eq == 'depth' :
        globals()[eq] = depth
        return depth
    if eq == 'WT' :
        globals()[eq] = WT
        return WT
def TRT(resultCB,functio_WW,NUM,UTR):
    FC = []
    if UTR[0] == 'chla':
        EP_V1 = 'chla'
    if UTR[0] == 'SA':
        EP_V1 = 'SA'
    if UTR[0] == 'depth':
        EP_V1 = 'depth'
    if UTR[0] == 'restime':
        EP_V1 = 'restime'
    if UTR[0] == 'WT':
        EP_V1 = 'WT'
    if UTR[1] == 'chla':
        EP_V2 = 'chla'
    if UTR[1] == 'SA':
        EP_V2 = 'SA'
    if UTR[1] == 'depth':
        EP_V2 = 'depth'
    if UTR[1] == 'restime':
        EP_V2 = 'restime'
    if UTR[1] == 'WT':
        EP_V2 = 'WT'
    if UTR[2] == 'chla':
        EP_V3 = 'chla'
    if UTR[2] == 'SA':
        EP_V3 = 'SA'
    if UTR[2] == 'depth':
        EP_V3 = 'depth'
    if UTR[2] == 'restime':
        EP_V3 = 'restime'
    if UTR[2] == 'WT':
        EP_V3 = 'WT'
    EP_SBL_JH1 = EP_V1 + '&'+  EP_V2
    EP_SBL_JH2 = EP_V1 + '&'+  EP_V3
    EP_SBL_JH3 = EP_V2 + '&'+  EP_V3
    EP_TBL_JH = EP_V1 + '&'+  EP_V2 + '&'+  EP_V3
    for renj in functio_WW:
        if renj == 'chla':
            dy = (np.round(resultCB.T['chla'][NUM],3) * symbols('chla'))
            FC.append(dy)
        if renj == 'SA':
            dy = (np.round(resultCB.T['SA'][NUM],3) *symbols('SA'))
            FC.append(dy)
        if renj == 'depth':
            dy = (np.round(resultCB.T['depth'][NUM],3) *symbols('depth'))
            FC.append(dy)
        if renj == 'WT':
            dy = (np.round(resultCB.T['WT'][NUM],3) * symbols('WT'))
            FC.append(dy)                
        if renj == EP_SBL_JH1:
            dy = (np.round(resultCB.T[EP_SBL_JH1 ][NUM],3) * symbols(EP_SBL_JH1))
            FC.append(dy)
        if renj == EP_SBL_JH2:
            dy = (np.round(resultCB.T[EP_SBL_JH2 ][NUM],3) * symbols(EP_SBL_JH2))
            FC.append(dy)            
        if renj == EP_SBL_JH3:
            dy = (np.round(resultCB.T[EP_SBL_JH3 ][NUM],3) * symbols(EP_SBL_JH3))
            FC.append(dy)  
        if renj == EP_TBL_JH:
            dy = (np.round(resultCB.T[EP_TBL_JH][NUM],3) * symbols(EP_TBL_JH))
            FC.append(dy)  
    DFC = ((np.sum(np.array(FC))) + np.round(resultCB.T['const'][NUM] ,3))
    return DFC

def Flux_Canculate(module_ADD_CHANGSHU,MODEL,UTR,Model_Standerror):
#module_ADD_CHANGSHU = 0
    all_HY = []
    all_HY_z = []
    all_HY_y = []
    for i in range(0,180):
        for j in range(0,20):
                section_SA = upscaling.iloc[i + 2 ,j + 8]
                if section_SA  != 0:
                    m_SA = log_A( upscaling.iloc[i + 2 , 5]   )
                    m_Chla = log_A( upscaling.iloc[0, j + 8 ]   )
                    m_WT = log_A( upscaling.iloc[i + 2 , 6]   )
                    m_Depth = log_A( upscaling.iloc[i + 2 , 3]   )
                S_Variable1 = symbols(UTR[0])
                S_Variable2 = symbols(UTR[1])
                S_Variable3 = symbols(UTR[2])
                
                S_SBLjh_1 = symbols(UTR[0] + '&' + UTR[1])
                S_SBLjh_2 = symbols(UTR[0] + '&' + UTR[2])
                S_SBLjh_3 = symbols(UTR[1] + '&' + UTR[2])
                S_TBLjh = symbols(UTR[0] + '&' + UTR[1] + '&' + UTR[2])
                
                if UTR[0] == 'chla':
                    EP_V1 = m_Chla
                if UTR[0] == 'SA':
                    EP_V1 =m_SA
                if UTR[0] == 'depth':
                    EP_V1 = m_Depth
                if UTR[0] == 'WT':
                    EP_V2 = m_WT
                if UTR[1] == 'chla':
                    EP_V2 = m_Chla
                if UTR[1] == 'SA':
                    EP_V2 =m_SA
                if UTR[1] == 'depth':
                    EP_V2 = m_Depth
                if UTR[1] == 'WT':
                    EP_V2 = m_WT
                if UTR[2] == 'chla':
                    EP_V3 = m_Chla
                if UTR[2] == 'SA':
                    EP_V3 =m_SA
                if UTR[2] == 'depth':
                    EP_V3 = m_Depth
                if UTR[2] == 'WT':
                    EP_V3 = m_WT
                    
           
                EP_SBL_JH1 = EP_V1 * EP_V2
                EP_SBL_JH2 = EP_V1 * EP_V3
                EP_SBL_JH3 = EP_V2 * EP_V3
                EP_TBL_JH = EP_V1 * EP_V2 * EP_V3

                test_fUNC=  float(MODEL.evalf(subs={
                    S_Variable1:EP_V1,
                    S_Variable2:EP_V2,
                    S_Variable3:EP_V3,
                    S_SBLjh_1:(EP_SBL_JH1),
                    S_SBLjh_2:(EP_SBL_JH2),
                    S_SBLjh_3:(EP_SBL_JH3),
                    S_TBLjh:(EP_TBL_JH),
}))
                
                reTUEN_FuncValues = (10** test_fUNC  )   - module_ADD_CHANGSHU 
                rF_z =  (10** (test_fUNC - 1.96*Model_Standerror)  )   - module_ADD_CHANGSHU 
                rF_y =  (10** (test_fUNC + 1.96*Model_Standerror)  )   - module_ADD_CHANGSHU 

                hy_dA  =  (reTUEN_FuncValues)  * section_SA * 1000000*365
                hy_dA_z = (rF_z)  * section_SA * 1000000*365
                hy_dA_y = (rF_y)  * section_SA * 1000000*365
                
                all_HY.append(hy_dA)
                all_HY_z.append(hy_dA_z)
                all_HY_y.append(hy_dA_y)
                if section_SA == 0:
                    all_HY.append(0)
                    all_HY_z.append(0)
                    all_HY_y.append(0)

                    
    ALL_origin = pd.Series(all_HY)
    sum_allORIG = np.sum(ALL_origin);
    out_TONGLIANGTOTAL  = (sum_allORIG*(10**(-15))) * 0.6
    
    ALL_origin_z = pd.Series(all_HY_z)
    sum_allORIG_z = np.sum(ALL_origin_z);
    out_TONGLIANGTOTAL_z  = (sum_allORIG_z*(10**(-15))) * 0.6
    
    ALL_origin_y = pd.Series(all_HY_y)
    sum_allORIG_y = np.sum(ALL_origin_y);
    out_TONGLIANGTOTAL_y  = (sum_allORIG_y*(10**(-15))) * 0.6
    

###########################
    ALLmit = []
    ALLmit_z = []
    ALLmit_y = []
    rwe = [0,20,40,60,80,100,120,140,160]
    for rw in rwe:
        all_HY2 = []
        all_HY2_z = []
        all_HY2_y = []

        dwmj = []
        for i2 in range(0+rw,20+rw):
            for j2 in range(0,20):
                    section_SA2 = upscaling.iloc[i2 + 2 ,j2 + 8]
                    if section_SA2  != 0:
                        m_SA2 = log_A( upscaling.iloc[i2 + 2 , 5]   )
                        m_Chla2 = log_A( upscaling.iloc[0, j2 + 8 ]   )
                        m_WT2 = log_A( upscaling.iloc[i2 + 2 , 6]   )
                        m_Depth2 = log_A( upscaling.iloc[i2 + 2 , 3]   )
                    S_Variable1 = symbols(UTR[0])
                    S_Variable2 = symbols(UTR[1])
                    S_Variable3 = symbols(UTR[2])
                    S_SBLjh_1 = symbols(UTR[0] + '&' + UTR[1])
                    S_SBLjh_2 = symbols(UTR[0] + '&' + UTR[2])
                    S_SBLjh_3 = symbols(UTR[1] + '&' + UTR[2])
                    S_TBLjh = symbols(UTR[0] + '&' + UTR[1] + '&' + UTR[2])
                    if UTR[0] == 'chla':
                        EP_V1 = m_Chla2
                    if UTR[0] == 'SA':
                        EP_V1 =m_SA2
                    if UTR[0] == 'depth':
                        EP_V1 = m_Depth2
                    if UTR[0] == 'WT':
                        EP_V2 = m_WT2
                    if UTR[1] == 'chla':
                        EP_V2 = m_Chla2
                    if UTR[1] == 'SA':
                        EP_V2 =m_SA2
                    if UTR[1] == 'depth':
                        EP_V2 = m_Depth2
                    if UTR[1] == 'WT':
                        EP_V2 = m_WT2
                    if UTR[2] == 'chla':
                        EP_V3 = m_Chla2
                    if UTR[2] == 'SA':
                        EP_V3 =m_SA2
                    if UTR[2] == 'depth':
                        EP_V3 = m_Depth2
                    if UTR[2] == 'WT':
                        EP_V3 = m_WT2
                        
                    EP_SBL_JH1 = EP_V1 * EP_V2
                    EP_SBL_JH2 = EP_V1 * EP_V3
                    EP_SBL_JH3 = EP_V2 * EP_V3
                    EP_TBL_JH = EP_V1 * EP_V2 * EP_V3

                    test_fUNC2=  float(MODEL.evalf(subs={
                    S_Variable1:EP_V1,
                    S_Variable2:EP_V2,
                    S_Variable3:EP_V3,
                    S_SBLjh_1:(EP_SBL_JH1),
                    S_SBLjh_2:(EP_SBL_JH2),
                    S_SBLjh_3:(EP_SBL_JH3),
                    S_TBLjh:(EP_TBL_JH),
}))
                    reTUEN_FuncValues2 = (10** test_fUNC2 )   - module_ADD_CHANGSHU 
                    
                    reTUEN_FuncValues2_z = (10** (test_fUNC2 - 1.96*Model_Standerror)  )   - module_ADD_CHANGSHU 
                    reTUEN_FuncValues2_y = (10** (test_fUNC2 + 1.96*Model_Standerror)  )   - module_ADD_CHANGSHU 

                    hy_dA2  =  (reTUEN_FuncValues2)  * section_SA2 * 1000000*365
                    hy_dA2_z  =  (reTUEN_FuncValues2_z)  * section_SA2 * 1000000*365
                    hy_dA2_y  =  (reTUEN_FuncValues2_y)  * section_SA2 * 1000000*365

                    all_HY2.append(hy_dA2)
                    all_HY2_z.append(hy_dA2_z)
                    all_HY2_y.append(hy_dA2_y)
                    
                    dwmj.append(reTUEN_FuncValues)
                    if section_SA2 == 0:
                        all_HY2.append(0)
                        dwmj.append(0)
                        all_HY2_z.append(0)
                        all_HY2_y.append(0)

        ALL_origin2 = pd.Series(all_HY2)
        sum_allORIG2 = np.sum(ALL_origin2);
        out_TONGLIANG2  = sum_allORIG2*(10**(-15))
        ALLmit.append(out_TONGLIANG2*0.6)
        
        ALL_origin2_z = pd.Series(all_HY2_z)
        sum_allORIG2_z = np.sum(ALL_origin2_z);
        out_TONGLIANG2_z  = sum_allORIG2_z*(10**(-15))
        ALLmit_z.append(out_TONGLIANG2_z*0.6)        
        
        ALL_origin2_y = pd.Series(all_HY2_y)
        sum_allORIG2_y = np.sum(ALL_origin2_y);
        out_TONGLIANG2_y  = sum_allORIG2_y*(10**(-15))
        ALLmit_y.append(out_TONGLIANG2_y*0.6)              
        

    print ('=================================================')
    print ('输出的  ' + str(ForcastName)+ '  总通量是(Tg)   ',np.round(out_TONGLIANGTOTAL,3))
    print ('输出的  ' + str('左右分别为')+ '  总通量是(Tg)   ',np.round(out_TONGLIANGTOTAL_z,3))
    print ('输出的  ' + str('左右分别为')+ '  总通量是(Tg)   ',np.round(out_TONGLIANGTOTAL_y,3))       
    print ('次级组合分别为',np.round(ALLmit,3),np.round(ALLmit_z,3),np.round(ALLmit_y,3))
    print ('显著的方程是',MODEL)
    print ('=====================运算finish=======================')       
    return np.round(out_TONGLIANGTOTAL,3),np.round(ALLmit,3),np.round(out_TONGLIANGTOTAL_z,3) ,np.round(out_TONGLIANGTOTAL_y,3),np.round(ALLmit_z,3),np.round(ALLmit_y,3)




def UDI(two_interaction,num):
    shuang = []
    for two_IT_1 in itertools.combinations(two_interaction, num):
        if num == 1:
            shuang.append(two_IT_1[0])
        if num == 2:
            shuang.append([two_IT_1[0],two_IT_1[1] ])
        if num == 3:
            shuang.append([two_IT_1[0],two_IT_1[1] ,two_IT_1[2]])
        if num == 4:
            shuang.append([two_IT_1[0],two_IT_1[1] ,two_IT_1[2],two_IT_1[3]])
    return shuang
def UDI_Value(two_interaction_values,num):
    shuang = []
    for two_IT_1 in itertools.combinations(two_interaction_values, num):
        if num == 1:
            shuang.append(two_IT_1[0])
        if num == 2:
            shuang.append([two_IT_1[0],two_IT_1[1] ])
        if num == 3:
            shuang.append([two_IT_1[0],two_IT_1[1] ,two_IT_1[2]])
        if num == 4:
            shuang.append([two_IT_1[0],two_IT_1[1] ,two_IT_1[2],two_IT_1[3]])
    return shuang


#%% 首先预测CH4_T
upscaling = pd.read_excel(r'C:\Users\KING\Desktop\温室气体数据整理_集合0508\UPscaling.xlsx',sheet_name='Lake_99')
eEdata = pd.read_excel(r'C:\Users\KING\Desktop\Triumphant_0515_湖泊.xlsx')
eEdatatype = eEdata[eEdata['Type']!='reservoirs']
zhdata  = eEdatatype
VARILIST = ['chla','SA','depth']

y_BL =  'co2'
Yfor_Variable = zhdata[y_BL]










FC_testA = []
FC_nameA = []
JH_2ji = []
JH_2ji_SYM = []
JH_3ji = []
JH_3ji_SYM = []
JH_1ji = []
JH_1ji_SYM = []
FC_testA = []
FC_nameA = []
JH_2ji = []
JH_2ji_SYM = []
JH_1ji = []
JH_1ji_SYM = []
Y_forlist = []



for UTR in itertools.combinations(VARILIST, 3):
    S_Variable1 = symbols(UTR[0])
    S_Variable2 = symbols(UTR[1])
    S_Variable3 = symbols(UTR[2])
    S_SBLjh_1 = symbols(UTR[0] + '&' + UTR[1])
    S_SBLjh_2 = symbols(UTR[0] + '&' + UTR[2])
    S_SBLjh_3 = symbols(UTR[1] + '&' + UTR[2])
    S_TBLjh = symbols(UTR[0] + '&' + UTR[1] + '&' + UTR[2])
    Q_Data = pd.concat([Yfor_Variable,zhdata['chla_execute'],zhdata['SA'],zhdata['Mean_execute'] ,zhdata['WT_execute']      ],  axis = 1).dropna(axis=0).reset_index(drop=True)
    Y_for = log_series(Q_Data[y_BL]  +  328.1)
    Y_forlist.append(Y_for)
    for pie in range(0,3):
        if UTR[pie] == 'chla':
            chla= log_series(Q_Data['chla_execute'])
        if UTR[pie] == 'SA':
            SA = log_series(Q_Data['SA'])
        if UTR[pie] == 'depth':
            depth = log_series(Q_Data['Mean_execute'])         
        if UTR[pie] == 'WT':
            WT = log_series(Q_Data['WT_execute'])         

    BLmz = UTR
    Var1 = BLmz[0];BL1 = Trans(Var1).rename( UTR[0], inplace = True)
    Var2 = BLmz[1];BL2 = Trans(Var2).rename( UTR[1], inplace = True)
    Var3 = BLmz[2];BL3 = Trans(Var3).rename( UTR[2], inplace = True)
    comB = [BL1,BL2,BL3]
    #comB.columns = [UTR[0],UTR[1],UTR[2]]
    SymB = [Var1,Var2,Var3]
    FC_testA.append(comB)
    FC_nameA.append(SymB)
    selfbl = SymB
    for rew_2ji in itertools.combinations(selfbl, 2):   ##此时添加最次级子交互项：
        JH_G_1 = rew_2ji[0]; BLJH_G_1 = Trans(JH_G_1)
        JH_G_2 = rew_2ji[1]; BLJH_G_2 = Trans(JH_G_2)
        JH_C_bl = BLJH_G_1*BLJH_G_2
        JH_C_symbol = JH_G_1+ '&' + JH_G_2
        JH_2ji.append(JH_C_bl)
        JH_2ji_SYM.append(JH_C_symbol)
                                                         ## 此时添加初级子交互项
    rew_1j =  BL1*BL2*BL3
    rew_1jSYM = SymB[0] + '&' + SymB[1] + '&' + SymB[2] 
    JH_1ji.append(rew_1j)
    JH_1ji_SYM.append(rew_1jSYM)


single_noninteraction_numble =  len(FC_nameA)
Value_FC_Single = []
Value_FC_twointer_dan = []
Value_FC_threeinter_dan = []
Value_FC_twointer_shuang = []
Value_FC_threeinter_shuang = []
Value_FC_threeinter_san = []
Value_FC_twointer_san = []
Value_FC_threeinter_si = []
Value_FC_twointer_si  = []
Value_FC_fourthinter_one  = []
Value_FC_2dan_plus_3dan = []
Value_FC_2dan_plus_3shuang= []
Value_FC_2dan_plus_3san= []
Value_FC_2dan_plus_3si= []
Value_FC_2dan_plus_4= []
Value_FC_2shuang_plus_3dan = []
Value_FC_2shuang_plus_3shuang= []
Value_FC_2shuang_plus_3san= []
Value_FC_2shuang_plus_3si= []
Value_FC_2shuang_plus_4= []
Value_FC_2san_plus_3dan = []
Value_FC_2san_plus_3shuang= []
Value_FC_2san_plus_3san= []
Value_FC_2san_plus_3si= []
Value_FC_2san_plus_4= []
Value_FC_2si_plus_3dan = []
Value_FC_2si_plus_3shuang= []
Value_FC_2si_plus_3san= []
Value_FC_2si_plus_3si= []
Value_FC_2si_plus_4= []
Value_FC_3dan_plus_4 = []
Value_FC_3shuang_plus_4 = []
Value_FC_3san_plus_4 = []
Value_FC_3si_plus_4 = []
FC_Single = []
FC_twointer_dan = []
FC_threeinter_dan = []
FC_twointer_shuang = []
FC_threeinter_shuang = []
FC_threeinter_san = []
FC_twointer_san = []
FC_threeinter_si = []
FC_twointer_si = []
FC_fourthinter_one = []
FC_2dan_plus_3dan = []
FC_2dan_plus_3shuang = []
FC_2dan_plus_3san = []
FC_2dan_plus_3si = []
FC_2dan_plus_4 = []
FC_2shuang_plus_3dan = []
FC_2shuang_plus_3shuang = []
FC_2shuang_plus_3san = []
FC_2shuang_plus_3si = []
FC_2shuang_plus_4 = []
FC_2san_plus_3dan = []
FC_2san_plus_3shuang = []
FC_2san_plus_3san = []
FC_2san_plus_3si = []
FC_2san_plus_4 = []
FC_2si_plus_3dan = []
FC_2si_plus_3shuang = []
FC_2si_plus_3san = []
FC_2si_plus_3si = []
FC_2si_plus_4 = []
FC_3dan_plus_4 = []
FC_3shuang_plus_4 = []
FC_3san_plus_4 = []
FC_3si_plus_4 = []
Y_FC_Single = []
Y_FC_twointer_dan = []
Y_FC_threeinter_dan = []
Y_FC_twointer_shuang = []
Y_FC_threeinter_shuang = []
Y_FC_threeinter_san = []
Y_FC_twointer_san = []
Y_FC_threeinter_si = []
Y_FC_twointer_si  = []
Y_FC_fourthinter_one  = []
Y_FC_2dan_plus_3dan = []
Y_FC_2dan_plus_3shuang= []
Y_FC_2dan_plus_3san= []
Y_FC_2dan_plus_3si= []
Y_FC_2dan_plus_4= []
Y_FC_2shuang_plus_3dan = []
Y_FC_2shuang_plus_3shuang= []
Y_FC_2shuang_plus_3san= []
Y_FC_2shuang_plus_3si= []
Y_FC_2shuang_plus_4= []
Y_FC_2san_plus_3dan = []
Y_FC_2san_plus_3shuang= []
Y_FC_2san_plus_3san= []
Y_FC_2san_plus_3si= []
Y_FC_2san_plus_4= []
Y_FC_2si_plus_3dan = []
Y_FC_2si_plus_3shuang= []
Y_FC_2si_plus_3san= []
Y_FC_2si_plus_3si= []
Y_FC_2si_plus_4= []
Y_FC_3dan_plus_4 = []
Y_FC_3shuang_plus_4 = []
Y_FC_3san_plus_4 = []
Y_FC_3si_plus_4 = []

for single_order in range( 0 , single_noninteraction_numble):
    two_interaction = JH_2ji_SYM[3*single_order:3 + 3*single_order]                 ## 双变量符号
    two_interaction_values = JH_2ji[3*single_order:3 + 3*single_order]              ## 双变量数值
    three_interaction = JH_1ji_SYM[1*single_order:1 + 1*single_order]                 ## 三变量符号
    three_interaction_values = JH_1ji[1*single_order:1 + 1*single_order]              ## 三变量数值
#######################   华丽分割线   #######################   
    Value_single_order = single_order
    Sym_single   = FC_nameA[single_order]       
    Value_single = FC_testA[single_order]
    FC_Single.append(Sym_single)                          ####  A先单独变量组一个方程组    
    Value_FC_Single.append(Value_single)
    Y_FC_Single.append(Y_forlist[single_order])                                                      ####  B 单独变量 +   双变量交互单   
    for Order_twointer_dan in range(0,3):
        twointer_dan_Sym = [two_interaction[Order_twointer_dan]]
        twointer_dan_Value = two_interaction_values[Order_twointer_dan]
        FC_twointer_dan.append( Sym_single + twointer_dan_Sym      )
        Value_FC_twointer_dan.append( Value_single + [twointer_dan_Value]      )
        Y_FC_twointer_dan.append(Y_forlist[single_order])
        for Order_2dan_plus_3dan in range(0,1):                     ####  单独变量 + 两遍量单 +   三变量交互单   
            Sym_2dan_plus_3dan = [three_interaction[Order_2dan_plus_3dan]]
            Value_2dan_plus_3dan = three_interaction_values[Order_2dan_plus_3dan]
            FC_2dan_plus_3dan.append( Sym_single +twointer_dan_Sym +  Sym_2dan_plus_3dan      )
            Value_FC_2dan_plus_3dan.append( Value_single + [twointer_dan_Value]   + [Value_2dan_plus_3dan]      )
            Y_FC_2dan_plus_3dan.append(Y_forlist[single_order])
        for Order_2dan_plus_3shuang in range(0,1):              ####  单独变量 +  两遍量单 +  三变量交互双   
            Sym_2dan_plus_3shuang = UDI(three_interaction,1)
            Value_2dan_plus_3shuang = UDI_Value(three_interaction_values,1)
            Sym_2dan_plus_3shuang_A = Sym_2dan_plus_3shuang[Order_2dan_plus_3shuang]
            Value_2dan_plus_3shuang_A = Value_2dan_plus_3shuang[Order_2dan_plus_3shuang]
            FC_2dan_plus_3shuang.append( Sym_single +twointer_dan_Sym  +[Sym_2dan_plus_3shuang_A ]    )
            Value_FC_2dan_plus_3shuang.append( Value_single + [twointer_dan_Value] +[ Value_2dan_plus_3shuang_A])    
            Y_FC_2dan_plus_3shuang.append(Y_forlist[single_order])
        for Order_2dan_plus_3san in range(0,1):              ####  单独变量 +  两遍量单 +  三变量交互三
            Sym_2dan_plus_3san = UDI(three_interaction,1)
            Value_2dan_plus_3san = UDI_Value(three_interaction_values,1)
            Sym_2dan_plus_3san_A = Sym_2dan_plus_3san[Order_2dan_plus_3san]
            Value_2dan_plus_3san_A = Value_2dan_plus_3san[Order_2dan_plus_3san]
            FC_2dan_plus_3san.append( Sym_single + twointer_dan_Sym + [Sym_2dan_plus_3san_A ]     )
            Value_FC_2dan_plus_3san.append( Value_single + [twointer_dan_Value] + [Value_2dan_plus_3san_A ]    )
            Y_FC_2dan_plus_3san.append(Y_forlist[single_order])
    for Order_twointer_shuang in range(0,3):              ####  C 单独变量 +   双变量交互双   
        shuangQJ = UDI(two_interaction,2)
        Values_shuangQJ = UDI_Value(two_interaction_values,2)
        twointer_shuang_Sym = shuangQJ[Order_twointer_shuang]
        twointer_shuang_Value = Values_shuangQJ[Order_twointer_shuang]
        FC_twointer_shuang.append( Sym_single + twointer_shuang_Sym      )
        Value_FC_twointer_shuang.append( Value_single + twointer_shuang_Value     )
        Y_FC_twointer_shuang.append(Y_forlist[single_order])
        for Order_2dan_plus_3dan in range(0,1):                     ####  单独变量 + 两变量双 +   三变量交互单   
            Sym_2dan_plus_3dan = [three_interaction[Order_2dan_plus_3dan]]
            Value_2dan_plus_3dan = three_interaction_values[Order_2dan_plus_3dan]
            FC_2shuang_plus_3dan.append( Sym_single +twointer_shuang_Sym +  Sym_2dan_plus_3dan      )
            Value_FC_2shuang_plus_3dan.append( Value_single + [twointer_shuang_Value]   + [Value_2dan_plus_3dan]      )
            Y_FC_2shuang_plus_3dan.append(Y_forlist[single_order])
        for Order_2dan_plus_3shuang in range(0,1):              ####  单独变量 +  两变量双  +  三变量交互双   
            Sym_2dan_plus_3shuang = UDI(three_interaction,1)
            Value_2dan_plus_3shuang = UDI_Value(three_interaction_values,1)
            Sym_2dan_plus_3shuang_A = Sym_2dan_plus_3shuang[Order_2dan_plus_3shuang]
            Value_2dan_plus_3shuang_A = Value_2dan_plus_3shuang[Order_2dan_plus_3shuang]
            FC_2shuang_plus_3shuang.append( Sym_single +twointer_shuang_Sym  +[Sym_2dan_plus_3shuang_A  ]   )
            Value_FC_2shuang_plus_3shuang.append( Value_single + [twointer_shuang_Value] + [Value_2dan_plus_3shuang_A])    
            Y_FC_2shuang_plus_3shuang.append(Y_forlist[single_order])
        for Order_2dan_plus_3san in range(0,1):              ####  单独变量 +  两变量双  +  三变量交互三
            Sym_2dan_plus_3san = UDI(three_interaction,1)
            Value_2dan_plus_3san = UDI_Value(three_interaction_values,1)
            Sym_2dan_plus_3san_A = Sym_2dan_plus_3san[Order_2dan_plus_3san]
            Value_2dan_plus_3san_A = Value_2dan_plus_3san[Order_2dan_plus_3san]
            FC_2shuang_plus_3san.append( Sym_single + twointer_shuang_Sym + [Sym_2dan_plus_3san_A ]     )
            Value_FC_2shuang_plus_3san.append( Value_single + [twointer_shuang_Value] + [Value_2dan_plus_3san_A ]    )
            Y_FC_2shuang_plus_3san.append(Y_forlist[single_order])
    for Order_twointer_san in range(0,1):                  ####  D 单独变量 +   双变量交互三
        sanQJ = UDI(two_interaction,3)
        Values_sanQJ = UDI_Value(two_interaction_values,3)
        twointer_san_Sym = sanQJ[Order_twointer_san]
        twointer_san_Value = Values_sanQJ[Order_twointer_san]
        FC_twointer_san.append( Sym_single + twointer_san_Sym      )
        Value_FC_twointer_san.append( Value_single + twointer_san_Value     )        
        Y_FC_twointer_san.append(Y_forlist[single_order])  
        for Order_2dan_plus_3dan in range(0,1):                     ####  单独变量 + 两变量三 +   三变量交互单   
            Sym_2dan_plus_3dan = [three_interaction[Order_2dan_plus_3dan]]
            Value_2dan_plus_3dan = three_interaction_values[Order_2dan_plus_3dan]
            FC_2san_plus_3dan.append( Sym_single +twointer_san_Sym +  Sym_2dan_plus_3dan      )
            Value_FC_2san_plus_3dan.append( Value_single + [twointer_san_Value]   + [Value_2dan_plus_3dan]      )
            Y_FC_2san_plus_3dan.append(Y_forlist[single_order])  
    for Order_fourth in range(0,1):              ####  J 单独变量 +   四变量交互单   
        QJ_fourth = UDI(three_interaction,1)
        Values_QJ_fourth = UDI_Value(three_interaction_values,1)
        fourth_Sym = QJ_fourth[Order_fourth]
        fourth_Value = Values_QJ_fourth[Order_fourth]
        FC_fourthinter_one.append( Sym_single + [fourth_Sym  ]   )
        Value_FC_fourthinter_one.append( Value_single + [fourth_Value ])    
        Y_FC_fourthinter_one.append(Y_forlist[single_order])                               






FC_all_QUANBU = FC_Single + FC_twointer_dan + FC_threeinter_dan + FC_twointer_shuang + FC_threeinter_shuang + FC_threeinter_san+FC_twointer_san + FC_threeinter_si + FC_twointer_si + FC_fourthinter_one + FC_2dan_plus_3dan + FC_2dan_plus_3shuang + FC_2dan_plus_3san + FC_2dan_plus_3si + FC_2dan_plus_4 + FC_2shuang_plus_3dan + FC_2shuang_plus_3shuang + FC_2shuang_plus_3san +  FC_2shuang_plus_3si + FC_2shuang_plus_4 + FC_2san_plus_3dan + FC_2san_plus_3shuang + FC_2san_plus_3san + FC_2san_plus_3si + FC_2san_plus_4 + FC_2si_plus_3dan + FC_2si_plus_3shuang + FC_2si_plus_3san + FC_2si_plus_3si + FC_2si_plus_4 + FC_3dan_plus_4 + FC_3shuang_plus_4 + FC_3san_plus_4 + FC_3si_plus_4 


Value_all_QUANBU = Value_FC_Single + Value_FC_twointer_dan + Value_FC_threeinter_dan + Value_FC_twointer_shuang + Value_FC_threeinter_shuang + Value_FC_threeinter_san+Value_FC_twointer_san + Value_FC_threeinter_si + Value_FC_twointer_si + Value_FC_fourthinter_one + Value_FC_2dan_plus_3dan + Value_FC_2dan_plus_3shuang + Value_FC_2dan_plus_3san + Value_FC_2dan_plus_3si + Value_FC_2dan_plus_4 + Value_FC_2shuang_plus_3dan + Value_FC_2shuang_plus_3shuang + Value_FC_2shuang_plus_3san +  Value_FC_2shuang_plus_3si + Value_FC_2shuang_plus_4 + Value_FC_2san_plus_3dan + Value_FC_2san_plus_3shuang + Value_FC_2san_plus_3san + Value_FC_2san_plus_3si + Value_FC_2san_plus_4 + Value_FC_2si_plus_3dan + Value_FC_2si_plus_3shuang + Value_FC_2si_plus_3san + Value_FC_2si_plus_3si + Value_FC_2si_plus_4 + Value_FC_3dan_plus_4 + Value_FC_3shuang_plus_4 + Value_FC_3san_plus_4 + Value_FC_3si_plus_4 


Y_all_QUANBU = Y_FC_Single + Y_FC_twointer_dan + Y_FC_threeinter_dan + Y_FC_twointer_shuang + Y_FC_threeinter_shuang + Y_FC_threeinter_san+ Y_FC_twointer_san + Y_FC_threeinter_si + Y_FC_twointer_si + Y_FC_fourthinter_one + Y_FC_2dan_plus_3dan + Y_FC_2dan_plus_3shuang + Y_FC_2dan_plus_3san + Y_FC_2dan_plus_3si + Y_FC_2dan_plus_4 + Y_FC_2shuang_plus_3dan + Y_FC_2shuang_plus_3shuang + Y_FC_2shuang_plus_3san +  Y_FC_2shuang_plus_3si + Y_FC_2shuang_plus_4 + Y_FC_2san_plus_3dan + Y_FC_2san_plus_3shuang + Y_FC_2san_plus_3san + Y_FC_2san_plus_3si + Y_FC_2san_plus_4 + Y_FC_2si_plus_3dan + Y_FC_2si_plus_3shuang + Y_FC_2si_plus_3san + Y_FC_2si_plus_3si + Y_FC_2si_plus_4 + Y_FC_3dan_plus_4 + Y_FC_3shuang_plus_4 + Y_FC_3san_plus_4 + Y_FC_3si_plus_4 




Second_Q_Func_ALL = []
for du_2rd,du_2nd in enumerate(FC_all_QUANBU):
    if FC_all_QUANBU[du_2rd][0] == 'chla1':
        m2 = [FC_all_QUANBU[du_2rd][0]] + ['chla2'] + FC_all_QUANBU[du_2rd][1:] 
        Second_Q_Func_ALL.append(m2)
Third_Q_Func_ALL = []
for du_3rd,du_3nd in enumerate(FC_all_QUANBU):
    if FC_all_QUANBU[du_3rd][0] == 'chla1':
        m3 = [FC_all_QUANBU[du_3rd][0]] + ['chla2'] + ['chla3'] + FC_all_QUANBU[du_3rd][1:] 
        Third_Q_Func_ALL.append(m3)
Fourth_Q_Func_ALL = []
for du_4rd,du_4nd in enumerate(FC_all_QUANBU):
    if FC_all_QUANBU[du_4rd][0] == 'chla1':
        m4 = [FC_all_QUANBU[du_4rd][0]] + ['chla2'] + ['chla3'] + ['chla4']+ FC_all_QUANBU[du_4rd][1:] 
        Fourth_Q_Func_ALL.append(m4)

VL_Second_Q = []
for du_2rd,du_2nd in enumerate(Value_all_QUANBU ):
    if Value_all_QUANBU [du_2rd][0].name == 'chla1':
        m2 = [Value_all_QUANBU [du_2rd][0]] + [Value_all_QUANBU [du_2rd][0]**2] + Value_all_QUANBU [du_2rd][1:] 
        VL_Second_Q.append(m2)
VL_Third_Q = []
for du_3rd,du_3nd in enumerate(Value_all_QUANBU):
    if Value_all_QUANBU[du_3rd][0].name == 'chla1':
        m3 = [Value_all_QUANBU[du_3rd][0]] + [Value_all_QUANBU[du_3rd][0]**2] + [Value_all_QUANBU[du_3rd][0]**3]+ Value_all_QUANBU[du_3rd][1:] 
        VL_Third_Q.append(m3)
VL_Foutrh_Q = []
for du_4rd,du_4nd in enumerate(Value_all_QUANBU):
    if Value_all_QUANBU[du_4rd][0].name == 'chla1':
        m4 = [Value_all_QUANBU[du_4rd][0]] + [Value_all_QUANBU[du_4rd][0]**2] + [Value_all_QUANBU[du_4rd][0]**3]+ [Value_all_QUANBU[du_4rd][0]**4]+ Value_all_QUANBU[du_4rd][1:] 
        VL_Foutrh_Q.append(m4)


Y_Second_Q = []
for du_2rd,du_2nd in enumerate(Value_all_QUANBU):
    if Value_all_QUANBU[du_2rd][0].name == 'chla1':
        m2 = Y_all_QUANBU[du_2rd]
        Y_Second_Q.append(m2)
Y_Third_Q = []
for du_3rd,du_3nd in enumerate(Value_all_QUANBU):
    if Value_all_QUANBU[du_3rd][0].name == 'chla1':
        m3 = Y_all_QUANBU[du_3rd]
        Y_Third_Q.append(m3)
Y_Foutrh_Q = []
for du_4rd,du_4nd in enumerate(Value_all_QUANBU):
    if Value_all_QUANBU[du_4rd][0].name == 'chla1':
        m4 =Y_all_QUANBU[du_4rd]
        Y_Foutrh_Q .append(m4)





Y_all_57 =  Y_all_QUANBU  +  Y_Second_Q + Y_Third_Q  + Y_Foutrh_Q 

FC_all_57 =  FC_all_QUANBU  +  Second_Q_Func_ALL + Third_Q_Func_ALL  + Fourth_Q_Func_ALL

VL_all_57 =  Value_all_QUANBU +  VL_Second_Q + VL_Third_Q  + VL_Foutrh_Q



#%%

CLASS_res = []
module_ADD_CHANGSHU = 328.22
ForcastName = 'CO2'
XZXsp = 0.05
zxqj = 0.6

#Carbon_neutralization = 5


for otwe,DJLmmot in enumerate(VL_all_57):
    for TYPE in ['OLS','RLM','MID']:
        try:
            logbl = pd.concat(DJLmmot,axis=1)
            logbl.columns= FC_all_57[otwe]
            logbl = sm.add_constant(logbl)
            UTR = FC_all_57[otwe][:4]
            Yfor = Y_all_57[otwe]
            olspp = sm.OLS(Yfor, logbl).fit() 
            Model_Standerror = np.round((np.mean((sm.OLS(Yfor, logbl).fit()).get_prediction().se_mean ) ) , 3)     

            if TYPE == 'OLS':
                ols_r = sm.OLS(Yfor, logbl).fit() 
                Model_Standerror = np.round((np.mean((sm.OLS(Yfor, logbl).fit()).get_prediction().se_mean ) ) , 3)     
                baseline,conf_left,conf_right,MODEL_res = FuntOUT( olspp, ols_r,'OLS',XZXsp,zxqj,UTR)
                GAS_TG_Base,allmit,blz,bly,flz,fly = Flux_Canculate(module_ADD_CHANGSHU,baseline,UTR,Model_Standerror)[0:6]               
                if  200 <= GAS_TG_Base <= 800:
                    #GAS_TG_left = Flux_Canculate(module_ADD_CHANGSHU,conf_left,UTR)[0]
                    #GAS_TG_right = Flux_Canculate(module_ADD_CHANGSHU,conf_right,UTR)[0]
                    GAS_pd = pd.concat( [MODEL_res,pd.DataFrame([GAS_TG_Base]),pd.DataFrame([blz]),pd.DataFrame([bly]),
                                         pd.DataFrame(allmit),
                                         pd.DataFrame([flz]).T,pd.DataFrame([fly]).T,
                                         pd.DataFrame([TYPE]),  pd.DataFrame([len(logbl)])     ],axis=1    )
                    GAS_pd.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic','Gas_base','Gas_z','Gas_y','ALLMIT','ALLMITz','ALLMITy','TYPE','n']
                    CLASS_res.append(GAS_pd.reset_index(drop=True))
            if TYPE == 'RLM':        
                ols_r = sm.OLS(Yfor, logbl).fit() 
                rlm_r = sm.RLM(Yfor, logbl).fit() 
                baseline,conf_left,conf_right,MODEL_res = FuntOUT(olspp,rlm_r,'RLM',XZXsp,zxqj,UTR)
                GAS_TG_Base,allmit,blz,bly,flz,fly = Flux_Canculate(module_ADD_CHANGSHU,baseline,UTR,Model_Standerror)[0:6]
                if  200 <= GAS_TG_Base <= 800:
                    #GAS_TG_left = Flux_Canculate(module_ADD_CHANGSHU,conf_left,UTR)[0]
                    #GAS_TG_right = Flux_Canculate(module_ADD_CHANGSHU,conf_right,UTR)[0]
                    GAS_pd = pd.concat( [MODEL_res,pd.DataFrame([GAS_TG_Base]),pd.DataFrame([blz]),pd.DataFrame([bly]),
                                         pd.DataFrame(allmit),
                                         pd.DataFrame([flz]).T,pd.DataFrame([fly]).T,
                                         pd.DataFrame([TYPE]),  pd.DataFrame([len(logbl)])     ],axis=1    )
                    GAS_pd.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic','Gas_base','Gas_z','Gas_y','ALLMIT','ALLMITz','ALLMITy','TYPE','n']
                    CLASS_res.append(GAS_pd.reset_index(drop=True))
            if TYPE == 'MID':        
                mid_r = QREG.QuantReg(Yfor, logbl).fit(q=0.5)
                baseline,conf_left,conf_right,MODEL_res = FuntOUT(olspp,ols_r,mid_r,'MID',XZXsp,zxqj,UTR)
                GAS_TG_Base,allmit,blz,bly,flz,fly = Flux_Canculate(module_ADD_CHANGSHU,baseline,UTR,Model_Standerror)[0:6]


                if  200 <= GAS_TG_Base <= 800:
                    #GAS_TG_left = Flux_Canculate(module_ADD_CHANGSHU,conf_left,UTR)[0]
                    #GAS_TG_right = Flux_Canculate(module_ADD_CHANGSHU,conf_right,UTR)[0]
                    GAS_pd = pd.concat( [MODEL_res,pd.DataFrame([GAS_TG_Base]),pd.DataFrame([blz]),pd.DataFrame([bly]),
                                         pd.DataFrame(allmit),
                                         pd.DataFrame([flz]).T,pd.DataFrame([fly]).T,
                                         pd.DataFrame([TYPE]),  pd.DataFrame([len(logbl)])     ],axis=1    )
                    GAS_pd.columns = ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic','Gas_base','Gas_z','Gas_y','ALLMIT','ALLMITz','ALLMITy','TYPE','n']
                    CLASS_res.append(GAS_pd.reset_index(drop=True))
        except:
            #dye = pd.DataFrame(list(['error'])*12).T.reset_index(drop=True)
            #dye.columns =  ['ALL_Function','SigniFicant_baseline','conf_left','conf_right','r2','r2_adj','aic','bic','Gas_base','GAS_left','Gas_right','TYPE']
            #CLASS_res.append(dye)
            continue
        
        
        
aLL_DATA_CLASS = pd.concat(CLASS_res)



aLL_DATA_CLASS.to_csv('C:\\Users\\KING\\Desktop\\三变量_CO2_碳中和与削减CCCC.csv')




#%% 验证是否准确：



