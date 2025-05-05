import faiss
import numpy as np 

def generate_noun_for_adj(word, embeddings_index):
    new_word=""
    if len(word) > 2 and word[-2:] == "er":
        new_word = new_word[:-1]
    elif len(word) > 4 and word[-4:] == "ical":
        new_word2 = word[:-2]
        list_elemnents_embeddings = []
        list_elemnents = []

        if new_word2 in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[new_word2])
            list_elemnents.append(new_word2)

        if new_word2+"e" in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[new_word2+"e"])
            list_elemnents.append(new_word2+"e")
        if word[:-4]+"y" in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[word[:-4]+"y"])
            list_elemnents.append(word[:-4]+"y")
        if word[:-4]+"istry" in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[word[:-4]+"istry"])
            list_elemnents.append(word[:-4]+"istry")

        if list_elemnents!=[]:
            ei = embeddings_index[word].reshape(1,-1)
            faiss.normalize_L2(ei)
            lee = np.array(list_elemnents_embeddings, dtype=np.float32, order='C')
            faiss.normalize_L2(lee)
            our_similarity = list(np.dot(ei.reshape(1,-1), lee.T).flatten())
            if max(our_similarity) > 0.45:
                new_word = list_elemnents[our_similarity.index(max(our_similarity))]

    elif len(word) > 3 and (word[-3:] == "ial" or word[-2:] == "al"):
        if word[-3:]=="ial":
            new_word2 = word[:-3]
        else:
            new_word2 = word[:-2]

        list_elemnents_embeddings = []
        list_elemnents = []

        if new_word2 in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[new_word2])
            list_elemnents.append(new_word2)
        if new_word2+"e" in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[new_word2+"e"])
            list_elemnents.append(new_word2+"e")
        if new_word2+"y" in embeddings_index.keys():
            list_elemnents_embeddings.append(embeddings_index[new_word2+"y"])
            list_elemnents.append(new_word2+"y")
        
        if list_elemnents!=[]:  
            ei = embeddings_index[word].reshape(1,-1)
            faiss.normalize_L2(ei)
            lee = np.array(list_elemnents_embeddings, dtype=np.float32, order='C')
            faiss.normalize_L2(lee)
            our_similarity = list(np.dot(ei.reshape(1,-1), lee.T).flatten())
            if max(our_similarity) > 0.45:
                new_word = list_elemnents[our_similarity.index(max(our_similarity))]
    return new_word

def generate_embedings_index():
    embeddings_index = {}
    english_words = set()


    f = open('../inputs/filtered_numberbatch.txt')

    for line in f:
        values = line.split()
        word = values[0]
        num = word.split(" ")

        coefs = np.asarray(values[1:], dtype='float32')
        list_word = word.split("/")
        lang_origin = list_word[2]
        end_word = list_word[-1]
        if len(list_word) > 1:
            if lang_origin == "en":
                embeddings_index[end_word] = coefs
                english_words.add(end_word)
            elif end_word not in english_words:
                embeddings_index[end_word] = coefs
    f.close()
    return embeddings_index

def is_list_relevant(similar_items, word2, embeddings_index):
    similar_items_embeddings = np.array([embeddings_index.get(wd, np.zeros(300)) for wd in similar_items],  dtype= np.float32, order='C')
    faiss.normalize_L2(similar_items_embeddings)
    values = np.dot(similar_items_embeddings, embeddings_index[word2].T)
    range_items = list(filter(lambda x: values[x]>=0.35, range(len(similar_items))))
    similar_items_temp = similar_items.copy()
    similar_items = [similar_items_temp[i] for i in range_items]

def open_filtered_assertions_file(antonyms_words,isA_relationship, isA_reverse_relationship, occuranceTermForParent, convertPluralToSingular, hasContext_relationship, hasContext_reverse_relationship, embeddings_index):
    with open("../inputs/filtered_assertions.txt") as f:
        rows = f.read().split("\n")
        for row in rows:
            if row=="":
                continue
            words = row.split(" ")
            if (words[0]=="/r/Antonym"):
                word1 = words[1]
                word2 = words[2]
                antonyms_words.add((word1, word2))


            if (words[0]=="/r/IsA"):
                word1 = words[1]
                word2 = words[2]

                if len(word1.split(f"{words[2]}_")) > 1:
                    word1 = word1.split(f"{words[2]}_")[1]
                elif len(word1.split(f"_{words[2]}")) > 1: 
                    word1 = word1.split(f"_{words[2]}")[0]

                if word1 in embeddings_index.keys() and word2 in embeddings_index.keys():
                    list_is = isA_relationship.get(word1, set())
                    list_is.add(word2)
                    isA_relationship[word1] = list_is
                    
                    list_is_rev = isA_reverse_relationship.get(word2, set())
                    list_is_rev.add(word1)
                    isA_reverse_relationship[word2] = list_is_rev
                    occuranceTermForParent[word2] = len(list_is_rev)
                        
                
            
            if (words[0]=="/r/FormOf"):
                word1 = words[1]
                word2 = words[2]
                convertPluralToSingular[word1] = word2


            if (words[0]=="/r/HasContext"):
                word1 = words[1]
                word2 = words[2]
                convertPluralToSingular[word1] = word2
                if len(word1.split(f"{words[2]}_")) > 1:
                    word1 = word1.split(f"{words[2]}_")[1]
                elif len(word1.split(f"_{words[2]}")) > 1: 
                    word1 = word1.split(f"_{words[2]}")[0]

                if word1 in embeddings_index.keys() and word2 in embeddings_index.keys():
                    list_is = isA_relationship.get(word1, set())
                    list_is.add(word2)
                    hasContext_relationship[word1] = list_is
                    
                    list_is_rev = hasContext_reverse_relationship.get(word2, set())
                    list_is_rev.add(word1)
                    hasContext_reverse_relationship[word2] = list_is_rev
    return antonyms_words,isA_relationship, isA_reverse_relationship, occuranceTermForParent, convertPluralToSingular, hasContext_relationship, hasContext_reverse_relationship, embeddings_index
      

def is_list_relevant(similar_items, word2, embeddings_index):
    if similar_items != list():
        similar_items_embeddings = np.array([embeddings_index.get(wd, np.zeros(300)) for wd in similar_items],  dtype= np.float32, order='C')
        faiss.normalize_L2(similar_items_embeddings)
        values = np.dot(similar_items_embeddings, embeddings_index[word2].T)
        range_items = list(filter(lambda x: values[x]>=0.35, range(len(similar_items))))
        similar_items_temp = similar_items.copy()
        similar_items = [similar_items_temp[i] for i in range_items]
    return similar_items

def generate_list_context(word1, word2, hasContext_reverse_relationship, embeddings_index, idx=None):
    similar_items = []
    if word2 in hasContext_reverse_relationship.keys():
        similar_items = list(filter(lambda x: f"_{word1}_" in x or word1 == x or f"{word1}_" == x[0:(len(word1))+1] or f"_{word1}" in x, hasContext_reverse_relationship[word2]))
        similar_items = is_list_relevant(similar_items, word2, embeddings_index)
    noun_for_adj = generate_noun_for_adj(word2, embeddings_index)
    
    if similar_items == list() and noun_for_adj in hasContext_reverse_relationship.keys():
        similar_items = list(filter(lambda x: f"_{word1}_" in x or word1 == x or f"{word1}_" == x[0:(len(word1)+1)] or f"_{word1}" in x, hasContext_reverse_relationship[noun_for_adj]))
        similar_items = is_list_relevant(similar_items, word2, embeddings_index)

    return similar_items


def go_further(word_key_label, word, related_words_to_a_word_similarity):
    if word not in related_words_to_a_word_similarity.keys():
        return []
    return list(filter(lambda x: f"_{word_key_label}_" in x or word_key_label == x or f"{word_key_label}_" == x[0:(len(word_key_label)+1)] or f"_{word_key_label}" in x, related_words_to_a_word_similarity[word]))



def words_related_to_key_words(word_key_label, wd, related_words_to_a_word_similarity, convertPluralToSingular, embeddings_index):
    if (len(wd)==1):
        return list()
    modify_word  = wd
  
    
    filtered_similarity_list = list()
    if modify_word in related_words_to_a_word_similarity:
        filtered_similarity_list = list(filter(lambda x: f"_{word_key_label}_" in x or word_key_label == x or f"{word_key_label}_" == x[0:(len(word_key_label)+1)] or f"_{word_key_label}" in x or go_further(word_key_label, x, related_words_to_a_word_similarity), related_words_to_a_word_similarity[modify_word]))
        filtered_similarity_list = is_list_relevant(filtered_similarity_list, modify_word, embeddings_index)
    if filtered_similarity_list == list() and word_key_label in convertPluralToSingular.keys() and len(convertPluralToSingular[word_key_label].split("_"))>1 and modify_word in related_words_to_a_word_similarity:
        word_key_label_new = convertPluralToSingular[word_key_label].split("_")[0]
        filtered_similarity_list = list(filter(lambda x: f"_{word_key_label_new}_" in x or word_key_label_new == x or f"{word_key_label_new}_" == x[0:(len(word_key_label_new)+1)] or f"_{word_key_label_new}" in x or go_further(word_key_label_new,x, related_words_to_a_word_similarity), related_words_to_a_word_similarity[modify_word]))
        filtered_similarity_list = is_list_relevant(filtered_similarity_list, modify_word, embeddings_index)

    return filtered_similarity_list




            