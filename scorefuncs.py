def LinearScoreFunction(weighted_difference, qmean):
    return 20-1.5*weighted_difference

def CubicScoreFunction(weighted_difference, qmean):
    return (200-(weighted_difference-50)**3)/5000.0

def MeanWeightedScoreFunction(weighted_difference, qmean):
    return 2*(10-weighted_difference)/qmean
