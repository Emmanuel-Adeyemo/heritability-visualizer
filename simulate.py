import numpy as np
from scipy import stats

def calc_heritability(VA, VE):
    VP = VA + VE

    if VP <= 0:
        raise ValueError('VA + VE must be greater than zero')
    return VA/VP


def simulate_parent_offspring(VA, VE, n_families, offspring_per_family=1, mu=100, rng=None):
    if VA < 0 or VE < 0:
        raise ValueError('VA or VE cannot be negative')

    if n_families < 2:
        raise ValueError('Need at least two families to plot regression')

    if offspring_per_family < 1:
        raise ValueError('Need at least one offspring per family')

    true_h2 = calc_heritability(VA, VE)

    parent1_bv = rng.normal(0, np.sqrt(VA), size=n_families)
    parent2_bv = rng.normal(0, np.sqrt(VA), size=n_families)

    parent1_pheno = mu + parent1_bv + rng.normal(0, np.sqrt(VE), size=n_families)
    parent2_pheno = mu + parent2_bv + rng.normal(0, np.sqrt(VE), size=n_families)

    midparent_pheno = (parent1_pheno + parent2_pheno) / 2.0
    midparent_bv = (parent1_bv + parent2_bv)/ 2.0 # expected bv of children

    sib_sampling_sd = np.sqrt(VA/2.0)
    envt_sd = np.sqrt(VE)

    offspring_pheno_sum = np.zeros(n_families)
    for _ in range(offspring_per_family):
        mendelian_samp = rng.normal(0, sib_sampling_sd, size=n_families)
        offspring_bv = midparent_bv + mendelian_samp
        offspring_envt = rng.normal(0, envt_sd, size=n_families)
        offspring_pheno_sum += mu + offspring_bv + offspring_envt

    offspring_pheno = offspring_pheno_sum / offspring_per_family

    return {
        'midparent_pheno': midparent_pheno,
        'offspring_pheno': offspring_pheno,
        'true_h2': true_h2
    }


def fit_regression(midparent_pheno, offspring_pheno):
    slope, intercept, r_value, p_value, std_err = stats.linregress(midparent_pheno, offspring_pheno)

    x_fit = np.array([midparent_pheno.min(), midparent_pheno.max()])
    y_fit = intercept + slope * x_fit

    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'x_fit': x_fit,
        'y_fit': y_fit
    }
