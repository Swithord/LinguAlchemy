#Create vectors for languages in the MasakhaNews dataset
from lang2vec import URIEL2 as uriel
from lang2vec import lang2vec as l2v
import torch
import numpy as np

#For old vectors
old_file_path = 'masakhanews_vectors\\old_vectors\\'
#For new vectors
new_file_path = 'masakhanews_vectors\\new_vectors\\'

# #Save
# #For old vectors
# # uriel.vanilla_URIEL()
# #Use feature_averages.npz for old syntax_average.pt (rename to features.npz)
# #Use feature_predictions.npz for old syntax_knn.pt (rename to features.npz)

# #For new vectors
# # uriel.update_URIEL()
# # uriel.imputation_interface()
# #Use average, softImpute for new syntax_average.pt (rename to features.npz)
# #Use union, softImpute for new syntax_knn.pt (rename to features.npz)

# #Run three times: geo.pt, syntax_knn.pt, syntax_average.pt
# iso_code_langs = ['amh', 'eng', 'fra', 'hau', 'ibo', 'lin', 'lug', 'orm', 'pcm', 'run', 'sna', 'som', 'swa', 'tir', 'xho', 'yor']

# #For old vectors only
# # vector = l2v.vector("syntactic", iso_code_langs) #Replace "syntactic" with "geographic" for geo.pt. 
# # vector = {key: [0 if x == -1 else x for x in value] for key, value in vector.items()} #Filling in missing values with 0

# #For new vectors only
# glottocode_langs = ['amha1245', 'stan1293', 'stan1290', 'haus1257', 'nucl1417', 'ling1263', 'gand1255', 'west2721', 'nige1257', 'rund1242', 'shon1251', 'soma1255', 'swah1253', 'tigr1271', 'xhos1239', 'yoru1245']
# vector = l2v.vector("syntactic", glottocode_langs) #Replace "syntactic" with "geographic" for geo.pt, vice-versa for syntax_knn.pt or syntax_average.pt. 
# for index, lang in enumerate(iso_code_langs):
#     vector[lang] = vector[glottocode_langs[index]]
#     del vector[glottocode_langs[index]]

# torch.save(vector, new_file_path + 'syntax_average.pt') #Replace name


#Load
file_path = 'masakhanews_vectors\\new_vectors\\syntax_average_geo.pt'
data = torch.load(file_path, weights_only=False)

for key, value in data.items():
    print(len(key))
    print(len(value))

# Convert data to string based on its type
if isinstance(data, torch.Tensor):
    data_str = str(data)
elif isinstance(data, dict):
    data_str = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_str = str(data_str)
elif isinstance(data, (list, tuple)):
    data_str = [item.tolist() if isinstance(item, torch.Tensor) else item for item in data]
    data_str = str(data_str)
else:
    data_str = str(data)

file_name = 'syntax_average_geo.txt'

# Write the string to a txt file
with open(file_name, 'w') as file:
    file.write(data_str)

print("Data has been written to " + file_name)


# #Concatenate vectors
# new_vectors = ['syntax_average_geo.pt', 'syntax_knn_geo.pt', 'syntax_knn_syntax_average.pt', 'syntax_knn_syntax_average_geo.pt']
# for index, pairs in enumerate([["syntax_average.pt", "geo.pt"], ["syntax_knn.pt", "geo.pt"], ["syntax_knn.pt", "syntax_average.pt"], ["syntax_knn_syntax_average.pt", "geo.pt"]]):
#     vectors = []
#     for p in range(2):
#         file_path = new_file_path +  pairs[p] #Replace with old_file_path for old vectors
#         vector = torch.load(file_path, weights_only=False) 
#         vectors.append(vector)

#     for key in vectors[0].keys():
#         vectors[0][key] = np.append(vectors[0][key], vectors[1][key])

#     torch.save(vectors[0], new_file_path + new_vectors[index]) #Replace with old_file_path for old vectors