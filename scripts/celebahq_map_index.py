import os
import pandas as pd


mapping = pd.read_table('CelebA-HQ-to-CelebA-mapping.txt', sep='\s+', index_col=0)
mapping_dict = dict()
for i in range(30000):
	mapping_dict.update({f'{i}.jpg': mapping.iloc[i]['orig_file']})

for key, value in mapping_dict.items():
	assert os.path.isfile(os.path.join('CelebA-HQ-img', key))
	os.rename(os.path.join('CelebA-HQ-img', key), os.path.join('CelebA-HQ-img', value))
