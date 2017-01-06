"""
A model of thermal comfort with clusters.

Authors:
    Seungjae Lee
    Ilias Bilionis

Date:
    01/05/2017
"""
import cluster_model
import pymc as pm
import pysmc as ps
import mpi4py.MPI as mpi
import argparse
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--db-file', type=str, default='smc.h5',
                    help='the file where the MCMC chain is written')
parser.add_argument('--particles', type=int, default=100,
                    help='the number of particles you want to use')
parser.add_argument('--num-mcmc', type=int, default=1,
                    help='the number of mcmc samples')
parser.add_argument('--model', type=str, default='cluster_pmv',
                    help='specify the model you wish to use')
parser.add_argument('--num-clusters', type=int, default=2,
                    help='specify the number of clusters')
parser.add_argument('--data-train', type=str, default='SynData.npz',
                    help='specify the training data')
args = parser.parse_args()


rank = mpi.COMM_WORLD.Get_rank()

DATA_FILE = os.path.join(os.path.dirname(__file__),'..','data',args.data_train)
data = cluster_model.load_training_data(DATA_FILE)
model = getattr(cluster_model, 'make_model_' + args.model)(*data, num_clusters=args.num_clusters)
mcmc = pm.MCMC(model)
getattr(cluster_model, 'assign_pysmc_step_functions_' + args.model)(mcmc)
smc = ps.SMC(mcmc,
             num_particles=args.particles,
             num_mcmc=args.num_mcmc,
             verbose=1,
             ess_reduction=0.90,
             ess_threshold=0.5,
             db_filename=args.db_file,
             mpi=mpi,
             gamma_is_an_exponent=True)
smc.initialize(0.)
smc.move_to(1.)
