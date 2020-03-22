import pickle

def CCR(test_labels, classifier_labels):
    CCR_Val = 0
    for label_idx in range(len(test_labels)):
        if test_labels[label_idx] == classifier_labels[label_idx]:
            CCR_Val += 1

    return 100.0 * CCR_Val / len(test_labels)

def save_list(my_list, file_name):
    with open(file_name, "wb") as fp:   #Pickling
        pickle.dump(my_list, fp)

def load_list(file_name):
    with open(file_name, "rb") as fp:   # Unpickling
        return pickle.load(fp)

def bayes_rule(obs_prob, state_prob, prior):
    return obs_prob * prior / state_prob