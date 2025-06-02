import torch
import itertools
from constants import _HOME_DIR

# read and process the cosine distant matrix for each model
distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-128-10steps-1.7b-matrix/cosine_dist_10_last3_128.pt')
distance_matrix = distance_matrix.detach().numpy()

labels_list = ['am', 'ar', 'av', 'az', 'be', 'bn', 'bo', 'br', 'ca', 'ce', 'ceb', 'ckb', 'cnh', 'co', 'cs', 'da', 'de', 'dv', 'ee', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fo', 'fr', 'fy', 'gd', 'gl', 'grc', 'gsw', 'gu', 'ha', 'haw', 'he', 'hi', 'hil', 'hmn', 'ht', 'hu', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kaa', 'kbd', 'kha', 'kk', 'kl', 'km', 'kn', 'ko', 'ky', 'la', 'lb', 'lg', 'lo', 'lus', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'oc', 'om', 'os', 'pa', 'pap', 'pl', 'ps', 'pt', 'rm', 'ro', 'ru', 'sa', 'sah', 'sd', 'se', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sw', 'ta', 'te', 'tet', 'tg', 'th', 'ti', 'tk', 'to', 'tr', 'ts', 'tt', 'tyv', 'udm', 'ug', 'uk', 'ur', 'uz', 'vec', 'vi', 'xh', 'yi', 'yo', 'yue', 'zh', 'zu']
n = len(labels_list)
numbers = list(range(n))

# The number of class
k = 16
# k = 2

samples_per_cluster = n // k

cluster_list = []

while len(numbers) > 0:
    print(f"Remaining numbers: {numbers}")
    
    # Step 1: find the two samples with the minimal distance
    min_dist = float('inf')
    min_pair = None
    for i, j in itertools.combinations(numbers, 2):
        dist = distance_matrix[i][j]
        if dist < min_dist:
            min_dist = dist
            min_pair = (i, j)

    cluster = list(min_pair)
    print(f"Initial pair with the smallest distance: {cluster} (Languages: {labels_list[cluster[0]]}, {labels_list[cluster[1]]})")
    
    while len(cluster) < samples_per_cluster:
        max_min_dist = float('inf')
        next_sample = None
        for i in numbers:
            if i in cluster:
                continue
            dist_to_cluster = [distance_matrix[i][c] for c in cluster]
            max_dist = max(dist_to_cluster)
            if max_dist < max_min_dist:
                max_min_dist = max_dist
                next_sample = i

        cluster.append(next_sample)
        print(f"Added sample: {next_sample} (Language: {labels_list[next_sample]})")

    cluster_list.append(tuple(cluster))

    for i in cluster:
        numbers.remove(i)

print("Final clusters:", cluster_list)

lang_list = []
for j in cluster_list:
    lang = tuple(labels_list[i] for i in j)
    lang_list.append(lang)

print("Final language clusters:", lang_list)
