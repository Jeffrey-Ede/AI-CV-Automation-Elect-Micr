"""
Track moments of loss to calculate its CDF. Use the losses' CDF to redictribute the
loss function.

Author: Jeffrey M. Ede
Email: j.m.ede@warwick.ac.uk
"""

from wolframclient.language import wlexpr
from wolframclient.evaluation import WolframLanguageSession

import tensorflow as tf  

def pearson_cdf(sess, x, a0, b2, b1, b0):
    """Use Wolfram kernel to evaluate Pearson CDF."""
    cmd = wlexpr(f'Abs[CDF[PearsonDistribution[1,{a0},{b2},{b1},{b0}]][{x}]]')
    return sess.evaluate(cmd)

def moments_to_pearson(mu1, mu2, mu3, mu4):
    """Use raw moments to approximate Pearson distribution coefficients."""

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

def _cdf_from_moments(sess, x, mu1, mu2, mu3, mu4):
    
    #Get Pearson distribution coefficients from raw moments
    a0, b2, b1, b0 = moments_to_pearson(mu1, mu2, mu3, mu4)
    
    #Evaluate Pearson CDF at location
    cdf = pearson_cdf(sess, x, a0, b2, b1, b0)
    
    return cdf

#TensorFlow function
def cdf_from_moments(x, mu1, mu2, mu3, mu4):
    """TensorFlow interface."""

    with WolframLanguageSession() as sess:
        cdf_fn = lambda x, mu1, mu2, mu3, mu4: _cdf_from_moments(sess, x, mu1, mu2, mu3, mu4)
        return tf.py_func(cdf_fn, [x, mu1, mu2, mu3, mu4], [tf.float32])

def track_moments(x, beta, num_mom, smart_beta_size=0):
    """Running averages of raw moments."""

    if smart_beta_size:
        beta = beta**smart_beta_size

    moments = [tf.get_variable(f"mom{i}", initializer=tf.constant(0, dtype=tf.float32)) 
               for i in range(1, num_mom+1)]
    update_moments_op = [mom.assign(beta*mom + (1-beta)*tf.reduce_mean(x**i)) for i, mom in enumerate(moments, start=1)]

    return update_moments_op, moments

def redistribute_loss(loss, batch_size, redistribution_fn):
    """Redistribute losses by tracking their moments to estimate their CDF and
    transforming to a new CDF.
    """

    
    update_moments_op, moments = track_moments(loss, prev_moments, num_moments=4, smart_beta_size=batch_size)

    with tf.control_dependencies(update_moments_op):
        #Get position on CDF for each loss
        cdf_fn = lambda x: cdf_from_moments(x, 
                                            mu1=moments[0],
                                            mu2=moments[1],
                                            mu3=moments[2],
                                            mu4=moments[3])
        loss_cdf = cdf_fn(loss)

        #Map to new loss function distribution
        redistributed_loss = redistribution_fn(loss_cdf)

    return redistributed_loss