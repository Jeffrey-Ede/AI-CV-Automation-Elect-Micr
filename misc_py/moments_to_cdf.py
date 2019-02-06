import numpy as np
from scipy.special import betainc, erfc, gamma, hyp2f1, beta, gammainc
#betainc is regularized incomplete beta
#erfc is complementary error function
#gammainc is regularized upper incomplete gamma
#hyp2f1 is hypergeometric function 2f1
#beta is beta function

"""Approzimation of CDFs for all 7 types of Pearson Distributions from the
first 4 moments of a distribution.

Author: Jeffrey Mark Ede
Email: j.m.ede@warwic.ac.uk
"""

#CDFs for each distribution family

def cdf_family1(x, a0, b2, b1, b0):
    r = np.sqrt((b1/b2)**2 - 4*b0/b2)
    c = 1/2 + (x + b1/(2*b2))/r
    d = (a0/b2 - b1/(b2**2))/r 

    if 0 < c < 1:
        cdf = betainc(1 + d - 1/(2*b2), 1 - d - 1/(2*b2), c)
    elif c >= 1:
        cdf = 1
    else:
        cdf = 0

    return cdf

def cdf_family2(x, a0, b2, b1, b0):
    r = np.sqrt((b1/b2)**2 - 4*b0/b2)
    c = 1/2 + (x + b1/(2*b2))/r

    if 0 < c < 1:
        cdf = betainc(1 - 1/(2*b2), 1-1/(2*b2), c)
    elif c >= 1:
        cdf = 1
    else:
        cdf = 0

    return cdf

def cdf_family3(x, a0, b2, b1, b0):

    if b1 == 0:
        cdf = 0.5*erfc(-a0*x/np.sqrt(2*b0))
    elif v1 + b1*x > 0:
        cdf = gammainc(1 + (b0-a0*b1)/(b1**2), (x+b0/b1)/b1) / gamma((x+b0/b1)/b1)
    else:
        cdf = 0

    return

def cdf_family4(x, a0, b2, b1, b0):
    r = np.sqrt(-(b1/b2)**2 + 4*b0/b2)
    n = b1/(b2**2) - 2*a0/b2
    s = i*n/r + 1/b2
    i = np.complex(0.+1j)

    f1 = 1/(1 - np.exp(-i*np.pi*s))
    f2n = i*2**(2 - 1/b2)*r**(-2 + 1/b2)*(x**2 + b0/b2 + x*b1/b2)**(1 - 1/(2*b2))
    f2n *= np.exp(n*np.arctan2(2*x + b1/b2, r)/r)*np.abs(gamma(s/2)/gamma(1/(2*b2)))**2
    f2n *= hyp2f1(1, 2 - 1/b2, 2  - s/2, 1/2 + i*(2*x + b1/b2)/(2*r))
    f2d = (2 - s)*beta((-1+1/b2)/2, 1/2)

    cdf = f1 + f22/f2d 

    return cdf

def cdf_family5(x, a0, b2, b1, b0):

    if x + b1/(2*b2) > 0:
        cdf = gammainc(-1 + 1/b2, (-a0+b1/(2*b2))/(b2*(x+b1/(2*b2))))
    else:
        cdf = 0

    return cdf

def cdf_family6(x, a0, b2, b1, b0):
    r = np.sqrt((b1/b2)**2 - 4*b0/b2)

    if x > r/2 - b1/(2*b2):
        cdf = betainc((b1 - 2*a0*b2 - r*b2 + 2*r*b2**2)/(2*r*b2**2), (1-b2)/b2, 
                      (x - r/2 +b1/(2*b2)/(x + r/2 + b1/(2*b2)))
    else:
        cdf = 0

    return cdf

def cdf_family7(x, a0, b2, b1, b0):
    f = 1/(1 + (b1 - 2*x*b2)**2/(-b1**2 + 4*b0*b2))

    if 2*x + b1/b2 < 0:
        cdf = betainc((1-b2)/(2*b2), 1/2, f)/2
    else:
        b = betainc((1-b2)/(2*b2), 1/2, 1)
        cdf = (1 + (b - betainc((1-b2)/(2*b2), 1/2, f))/b)/2

    return cdf


def pearson_cdf(x, a0, b2, b1, b0):
    """Identify family of Pearson Distribution and evaluate."""
    
    #a0 = np.complex(a0)
    #b2 = np.complex(b2)
    #b1 = np.complex(b1)
    #b0 = np.complex(b0)

    #Common expressions in conditions
    all_real = np.isreal(a0) and np.isreal(b2) and np.isreal(b1) and np.isreal(b0)
    print(b1, b2, b1**2/(4*b2**2) - b0/b2)
    r = np.sqrt(b1**2/(4*b2**2) - b0/b2)
    rsub = r - b1/(2*b2)
    d = b1**2 - 4*b0*b2

    #Family 4
    cond4 = (all_real and
             b2**2 > 0 and 
             d < 0 and
             1/b2 > 1)
    if cond4:
        return cdf_family4(x, a0, b2, b1, b0)

    #Family 1
    cond1 = (all_real and
             b1**2 - 4*b0*b2 < 0 and
             b2**2 > 0 and
             (a0 - rsub) / (2*b2*r) > -1 and
             (a0 + rsub) / (2*b2*r) > -1)
    if cond1:
        return cdf_family1(x, a0, b2, b1, b0)

    #Family 6
    cond6 = (all_real and
             b2**2 > 0 and
             d > 0 and
             1/b2 - 1 > 0 and
             2*r - (a0 + rsub) / b2 > 0)
    if cond6:
        return cdf_family6(x, a0, b2, b1, b0)

    #Family 3
    cond3 = (np.isreal(a0) and
             b2 == 0 and
             np.isreal(b1) and
             np.isreal(b0) and
             ((b1**2 > 0 and b1 > 0 and b0 - a0*b1 > -b1**2) or
              (b1 == 0 and b0**2 > 0 and b0 > 0)))
    if cond3:
        return cdf_family3(x, a0, b2, b1, b0)


    #Family 5
    cond5 = (all_real and
             b2**2 > 0 and
             b1 == 4*b0*b2 and
             b1 > 2*a0*b2 and
             1/b2 > 1)
    if cond5:
        return cdf_family5(x, a0, b2, b1, b0)

    #Family 2
    cond2 = (all_real and
             d < 0 and
             b2**2 > 0 and
             2*a0*b2 == b1 and 
             (a0 - rsub) / (2*b2*r) > -1)
    if cond2:
        return cdf_family2(x, a0, b2, b1, b0)

    #Family 7
    cond7 = (all_real and
             b2**2 > 0 and
             d < 0 and 
             1/b2 > 1 and 
             b1 == 2*a0*b2)
    if cond7:
        return cdf_family7(x, a0, b2, b1, b0)

    #Raise error if parameters do not match any family
    raise ValueError("""Parameters do not match any Pearson Distribution""")


def moments_to_pearson(mu1, mu2, mu3, mu4):
    """Use raw moments to approximate Pearson distribution coefficients"""

    d = 2*(9*mu2**3 + 4*mu1**3*mu3 - 16*mu1*mu2*mu3 + 6*mu3**2 - 5*mu2*mu4 + mu1**2*(-3*mu2**2 + 5*mu4))

    a0 = 20*mu1**2*mu2*mu3 - 12*mu1**3*mu4 - mu3*(3*mu2**2 + mu4) + mu1*(-9*mu2**3 - 8*mu3**2 + 13*mu2*mu4)
    a0 /= d

    b0 = 6*mu2**3 + 4*mu1**3*mu3 - 10*mu1*mu2*mu3 + 3*mu3**2 - 2*mu2*mu4* + mu1**2*(-3*mu2**2 + 2*mu4)
    b0 /= d

    b1 = 8*mu1**2*mu2*mu3 - 6*mu1**3*mu4 - mu3*(3*mu2**2 + mu4) + mu1*(-3*mu2**3 - 2*mu3**2 + 7*mu2*mu4)
    b1 /= d

    b2 = mu1*mu3*(mu2**2 + mu4) + mu2*(3*mu3**2 - 4*mu2*mu4) + mu1**2*(-4*mu3**2 + 3*mu2*mu4)
    b2 /= d

    return a0, b2, b1, b0

def cdf_from_moments(x, mu1, mu2, mu3, mu4):
    
    #Get Pearson distribution coefficients from raw moments
    a0, b2, b1, b0 = moments_to_pearson(mu1, mu2, mu3, mu4)
    
    cdf = pearson_cdf(x, a0, b2, b1, b0)
    
    return cdf

if __name__ == "__main__":
    
    cdf = cdf_from_moments(2., 1., 1.2, 1.4, 1.6)
    print(cdf)