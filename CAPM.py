######################################################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm

######################################################################################################

def CAPM(dependent_variable, explanatory_variables, alpha=True):
    """
    
    Runs a linear regression to decompose the dependent variable into the explanatory variables:

    1- CAPM stands for Capital Asset Pricing Model.

    2- The main idea of Factor Analysis is to take a set of observed returns and decompose
       it into a set of explanatory returns.
    
    3- The CAPM is in some sense an example of a factor model. In particular, it is
       used to determine a theoretically appropriate return of an asset to make decisions
       about adding this asset to a well-diversified portfolio.

    4- CAPM is a one-factor model since the excess return of the security depends only
       on the excess return of the market.

    5- The function returns an object of type statsmodel's regression results.
    
    6- Call .summary() to print a full summary.
    
    7- Call .params for the coefficients, here Beta and Alpha
    
    8- Call .tvalues and .pvalues for the significance levels.

    9- Call .rsquared_adj and .rsquared for quality of fit.

    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm