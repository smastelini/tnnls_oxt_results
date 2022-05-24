#PBS -N on-reg
#PBS -l select=ncpus=1
#PBS -l walltime=300:00:00
# #PBS -m abe
# #PBS -M saulomastelini@gmail.com

module load python/3.6.8-pandas

source ~/online-learning/py36/bin/activate
cd ~/online-learning/src

#python run.py
#python baselines.py
#python run_drift.py
#python saturation_study.py
#python parse_saturation.py
#python plot_saturation_curves.py
#python plot_saturation_heatmaps.py
python extract_forest_stats.py
python parse_forest_stats.py

