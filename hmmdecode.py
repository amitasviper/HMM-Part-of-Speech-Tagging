# coding: utf-8

import os
import sys
from math import log
import numpy as np


def getFileContents(filename):
    data = None
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


def getFileFromCommandLine():
    filename = sys.argv[1]
    return getFileContents(filename)



def splitWordTag(word_tag_pair):
    splitted = word_tag_pair.split('/')
    tag = splitted[-1]
    word = '/'.join(splitted[:-1])
    return word, tag



def getUniqueTags(tagged_data):
    tags = {}
    for line in tagged_data:
        word_tag_pairs = line.strip().split(' ')
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            if tag in tags.keys():
                tags[tag] += 1
            else:
                tags[tag] = 1
    return tags


def getUniqueWords(tagged_data):
    words = []
    for line in tagged_data:
        word_tag_pairs = line.strip().split(' ')
        
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            words.append(word)
    return list(set(words))


def readModelFile():
    filename = 'hmmmodel.txt'
    lines = []
    with open(filename, 'r') as model_file:
        lines = model_file.readlines()
    return lines


def parseModel(lines):
    total_tags = int(lines[0].strip().split(':')[-1])
    total_words = int(lines[1].strip().split(':')[-1])
    
    tr_start_line_number = int(lines[2].strip().split(':')[-2])
    tr_end_line_number = int(lines[2].strip().split(':')[-1])
    
    em_start_line_number = int(lines[3].strip().split(':')[-2])
    em_end_line_number = int(lines[3].strip().split(':')[-1])
    
    oc_start_line_number = int(lines[4].strip().split(':')[-2])
    oc_end_line_number = int(lines[4].strip().split(':')[-1])
    
    wi_start_line_number = int(lines[5].strip().split(':')[-2])
    wi_end_line_number = int(lines[5].strip().split(':')[-1])
    
    af_start_line_number = int(lines[6].strip().split(':')[-2])
    af_end_line_number = int(lines[6].strip().split(':')[-1])

    mut_start_line_number = int(lines[7].strip().split(':')[-2])
    mut_end_line_number = int(lines[7].strip().split(':')[-1])
    
#     print total_tags, total_words, tr_start_line_number, tr_end_line_number, em_start_line_number, em_end_line_number, oc_start_line_number,oc_end_line_number, wi_start_line_number, wi_end_line_number
    
    probability_transition_matrix = []
    for line_number in range(tr_start_line_number, tr_end_line_number, 1):
        row_values = map(float, lines[line_number].strip().split('\t'))
        probability_transition_matrix.append(row_values)
    
    probability_emission_matrix = []
    for line_number in range(em_start_line_number, em_end_line_number, 1):
        row_values = map(float, lines[line_number].strip().split('\t'))
        probability_emission_matrix.append(row_values)
        
    
    opening_probabilities = {}
    closing_probabilities = {}
    
    tags_index_dict = {}
    tags_index_dict_reverse = {}
    
    for line_number in range(oc_start_line_number, oc_end_line_number, 1):
        row_values = lines[line_number].strip().split('\t')
        tag_name = row_values[0]
        open_p = float(row_values[1])
        close_p = float(row_values[2])
        index = int(row_values[3])
        
        opening_probabilities[tag_name] = open_p
        closing_probabilities[tag_name] = close_p
        tags_index_dict[tag_name] = index
        tags_index_dict_reverse[index] = tag_name
    
    words_index_dict = {}
    words_index_dict_reverse = {}
    
    for line_number in range(wi_start_line_number, wi_end_line_number, 1):
        row_values = lines[line_number].strip().split('\t')
        word = row_values[0]
        index = None
        try:
            index = int(row_values[1])
        except:
            index = int(row_values[0])
            word = " "
            print "Empty space as a word present in model"
        words_index_dict[word] = index
        words_index_dict_reverse[index] = word
        
    additional_features = {}
    for line_number in range(af_start_line_number, af_end_line_number, 1):
        row_values = lines[line_number].strip().split('\t')
        feature_name = row_values[0]
        feature_tag = row_values[1]
        additional_features[feature_name] = feature_tag
    
    most_frequent_tag = lines[mut_start_line_number].strip()
    # print "The most frequent tag is : ", most_frequent_tag
        
    return opening_probabilities, closing_probabilities, probability_transition_matrix, probability_emission_matrix, tags_index_dict, tags_index_dict_reverse, words_index_dict, words_index_dict_reverse, additional_features, most_frequent_tag



def getMostProbableTags(sentence):
    global opening_probabilities, closing_probabilities, probability_transition_matrix, probability_emission_matrix, tags_index_dict, tags_index_dict_reverse, words_index_dict, words_index_dict_reverse 
    global tag_count, unseen_words, additional_features, most_frequent_tag
    
    sentence_words = sentence.strip().split(' ')
    
    sentence_len = len(sentence_words)
    
    viterbi_matrix = np.zeros(shape=(tag_count, sentence_len))
    
    tracing_matrix = [[most_frequent_tag for x in range(sentence_len)] for y in range(tag_count)]
    
    for col in range(sentence_len):
        word = sentence_words[col]
        for model_tag in tags_index_dict:
            model_tag_index = tags_index_dict[model_tag]
            try:
                word_emission_probability = probability_emission_matrix[model_tag_index][words_index_dict[word]]
                if word_emission_probability == 0.0:
                    continue
            except KeyError as e:
                if any(word.lower().endswith(last) for last in ('.com', '.net', '.org', '.edu')) or word.startswith('http') or word.startswith('www.'):
                    if model_tag == additional_features['URLS']:
                        word_emission_probability = 1.0
                    else:
                        word_emission_probability = 1.1754943508222875e-30

                elif word.count('=') > 10 or word.count('_') > 10 or word.count('*') > 10 or word.count('-') > 10 or word.count('+') > 10:
                    if model_tag == additional_features['PAGE_SEP']:
                        word_emission_probability = 1.0
                    else:
                        word_emission_probability = 1.1754943508222875e-30

                elif [char.isdigit() for char in word].count(True) * 1.0 > len(word) * 0.4:
                    if model_tag == additional_features['NUMERICS']:
                        word_emission_probability = 1.0
                    else:
                        word_emission_probability = 1.1754943508222875e-30
                else:
                    try:
                        word_emission_probability = probability_emission_matrix[model_tag_index][words_index_dict[word.lower()]]
                    except KeyError as e:
                        word_emission_probability = 1.0 #probability_emission_matrix[model_tag_index][-1]
            
            if col == 0:
                try:
                    tag_opening_probability = opening_probabilities[model_tag]
                except KeyError as e:
                    print "tag_opening_probability : Keyerror encountered"
                    tag_opening_probability = 1.1754943508222875e-100
                viterbi_matrix[model_tag_index][col] = tag_opening_probability * word_emission_probability
            else:
                max_probability = 0.0 #np.finfo(float).min
                max_tag = most_frequent_tag
                
                for prev_model_tag in tags_index_dict:
                    prev_model_tag_index = tags_index_dict[prev_model_tag]
                    tag_transition_probability = probability_transition_matrix[prev_model_tag_index][model_tag_index]
                    if tag_transition_probability == 0.0:
                        print "Transition probability still zero"
                        tag_transition_probability = 1.1754943508222875e-100
                    temp_probability = viterbi_matrix[prev_model_tag_index][col-1] * tag_transition_probability * word_emission_probability  
                    if temp_probability >= max_probability:
                        max_probability = temp_probability
                        max_tag = prev_model_tag
                        
                viterbi_matrix[model_tag_index][col] = max_probability
                tracing_matrix[model_tag_index][col] = max_tag
    
    max_probability = 0.0 #np.finfo(float).min
    max_probability_tag = most_frequent_tag
    for model_tag in tags_index_dict:
        model_tag_index = tags_index_dict[model_tag]
        temp_probability = 0.0
        try:
            tag_closing_probabilities = closing_probabilities[model_tag]
        except KeyError as e:
            print "tag_closing_probabilities : Keyerror encountered", 
            tag_closing_probabilities = 1.1754943508222875e-100
        temp_probability =  tag_closing_probabilities * viterbi_matrix[model_tag_index][sentence_len-1]
        if temp_probability >= max_probability:
            max_probability = temp_probability
            max_probability_tag = model_tag

    assigned_tags = [max_probability_tag]
    current_best_tag = max_probability_tag
    
    for col in range(sentence_len-1, 0, -1):
        current_best_tag = tracing_matrix[tags_index_dict[current_best_tag]][col]
        assigned_tags.append(current_best_tag)
    assigned_tags = assigned_tags[::-1]
    
    anotated_sentence = ''
    for index, assigned_tag in enumerate(assigned_tags):
        anotated_sentence += str(sentence_words[index]) + '/' + str(assigned_tag) + ' '
    
    return anotated_sentence.strip()

def startPredicting():
    test_data = getFileFromCommandLine()
#     test_data = getFileContents('data/en_dev_raw.txt')
    output = ''
    for test_line in test_data:
        predicted_tagged_line = getMostProbableTags(test_line)
        output += predicted_tagged_line + '\n'
    
    with open('hmmoutput.txt', 'w') as output_file:
        output_file.write(output)



if __name__ == '__main__':
    lines = readModelFile()
    opening_probabilities, closing_probabilities, probability_transition_matrix, probability_emission_matrix, tags_index_dict, tags_index_dict_reverse, words_index_dict, words_index_dict_reverse, additional_features, most_frequent_tag  = parseModel(lines)
    # print additional_features
    # MIN_VAL = min([min(r) for r in a])
    tag_count = len(tags_index_dict.keys())
    startPredicting()
    # print "Done"

