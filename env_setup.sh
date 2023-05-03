# add minconda to path variable
export PATH="/raid/cs20mds14030/miniconda/bin:$PATH"
# delete if any anconda related files present in the system
anaconda-clean --yes
conda env create -n telugu_asr --file environment.yml