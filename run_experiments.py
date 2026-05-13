# uncomment the experiments you want to run
# plots generated for the paper are put in the plots folder 
# if you want to use a plot in the paper you will need to move 
# it into documents/figures and latex the paper again


# experiments for tuning orthogonal collocation encoding of NLP
# from experiments import oc_experiments

# experiments for tuning multiple shooting encoding of NLP
from experiments import ms_experiments

# experiments for tuning Chebyshev pseudo-spectral collocation encoding of NLP
# from experiments import cps_experiments

# experiments that compare the three NLP encodings
# from experiments import nlp_experiment