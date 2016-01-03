import numpy as np


def get_stars(prob_prediction, list_percentiles):
    '''
    given a prediction (FLOAT), and a np.array of percentiles (9 values)
    return the fillings for the html code needed to see the right number of stars
    '''
    percentile = np.digitize(prob_prediction, list_percentiles)+1

    total_stars = 5
    number_of_full_stars = percentile/2
    half_star = percentile%2
    number_of_empty_stars = total_stars - number_of_full_stars - half_star

    full = range(number_of_full_stars)
    half = half_star
    empty = range(number_of_empty_stars)

    return percentile, full, half, empty
