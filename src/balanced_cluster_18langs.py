import torch
import itertools
from constants import _HOME_DIR
import sys

model_name = sys.argv[1].lower()
k = int(sys.argv[2])  # number of class

# read and process the cosine distant matrix for each model
# distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-bloom-560m-matrix/cosine_dist_10_last3.pt')
# distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-bloom-560m-matrix/cosine_dist_10_all.pt')
# distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-gemma-2b-matrix/cosine_dist_10_last3.pt')
# distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-bloom-1.7b-matrix/cosine_dist_10_last3.pt')
# distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-qwen2.5-1.5b-matrix/cosine_dist_10_last3.pt')
distance_matrix = torch.load(f'{_HOME_DIR}/data/multi-18-10steps-{model_name}-matrix/cosine_dist_10_last3.pt')
distance_matrix = distance_matrix.detach().numpy()


labels_list = ["ar", "bn", "de", "fr", "hi", "id", "it", "ja", "ko", "nl", "ru", "ta", "te", "th", "uk", "ur", "vi", "zh"]
n = len(labels_list) # number of label
numbers = list(range(n))
cluster_list = []

while len(numbers) > 0:
    combination_size = n // k

    combinations = list(itertools.combinations(numbers, combination_size))
    print(f"Number of combination: {len(combinations)}")

    dist_list = []
    for item in combinations:
        dist_max = 0
        for pair in itertools.combinations(item, 2):  # calculate the distance between each pair
            dist_1, dist_2 = pair
            dist_max = max(dist_max, distance_matrix[dist_1][dist_2])
        dist_list.append(dist_max)

    # get the minimal max distance combination
    min_value = min(dist_list)
    min_index = dist_list.index(min_value)
    cluster = combinations[min_index]
    print(f"The cluster choosen: {cluster}, max distance: {min_value}")
    cluster_list.append(cluster)

    for i in cluster:
        numbers.remove(i)

print(cluster_list)

lang_list = []
for j in cluster_list:
    lang = tuple(labels_list[i] for i in j)
    lang_list.append(lang)

print(lang_list)
