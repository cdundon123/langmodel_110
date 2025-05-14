"""
15-110 Hw6 - Language Modeling Project
Name: Catherine Dundon
AndrewID: cdundon
"""

import hw6_language_tests as test
import string
import numpy
import matplotlib

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):

    corpus = []
    
    f = open(filename, "r")
    text = f.read()
    
    sentences = text.split("\n") # [hello there, how are you]
    
    for sentence in sentences:
        words = sentence.split()# [hello, there]
        if words != []:
            corpus.append(words)
    
    return corpus
    


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    
    length = 0
    
    for sentence in corpus:
        for word in sentence:
            length += 1
    
    return length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    
    vocab = []
    
    for sentence in corpus:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    
    return vocab


'''
makeStartCorpus(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: 2D list of strs
'''
def makeStartCorpus(corpus):
    
    starters = []
    
    for sentence in corpus:
        starters.append([sentence[0]])
    
    return starters


'''
countUnigrams(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    
    count = {}
    
    for sentence in corpus:
        for word in sentence:
            if word not in count:
                count[word] = 1
            else:
                count[word] = count[word] + 1
    
    return count


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    
    count = {}
    
    for sentence in corpus:
        for i in range(len(sentence)-1):
            word = sentence[i]
            if word not in count:
                count[word] = {}
            if sentence[i + 1] not in count[word]:
                count[word][sentence[i + 1]] = 1
            else: 
                count[word][sentence[i + 1]] = count[word][sentence[i + 1]] + 1
                
    return count


'''
separateWords(line)
#7 [Check6-1]
Parameters: str
Returns: list of strs
'''
def separateWords(line):
    
    tokens = line.split()
    result = []
    
    for token in tokens:
        start = 0
        i = 0
        while i < len(token):
            char = token[i]
            if char in string.punctuation and char != "'":
                if start != i:
                    result.append(token[start:i])
                result.append(char) 
                start = i + 1 
            i += 1
        if start < len(token):
            result.append(token[start:])
        
    return result



'''
cleanBookData(text)
#8 [Check6-1]
Parameters: str
Returns: str
'''
def cleanBookData(text):
    
    
    separated = separateWords(text)
    
    cleaned = ""
    
    
    for word in separated:
        
        if word in ".!?":
            cleaned = cleaned.strip() + " " + word + "\n"
        else:
            cleaned = cleaned + word.lower() + " "
            
    cleaned = cleaned.strip()
    
    return cleaned


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    
    probabilities = []
    
    for unigram in unigrams:
        probabilities.append(1/len(unigrams))
    
    return probabilities


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    
    probs = []
    
    for unigram in unigrams:
        count = unigramCounts[unigram]
        chance = count/totalCount
        probs.append(chance)
    
    return probs


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    
    probs = {}

    for prevWord in bigramCounts:
        words = []
        wordProbs = []

        for nextWord in bigramCounts[prevWord]:
            words.append(nextWord)
            prob = bigramCounts[prevWord][nextWord] / unigramCounts[prevWord]
            wordProbs.append(prob)

        temp = {"words": words, "probs": wordProbs}
        
        probs[prevWord] = temp

    return probs

'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    
    highest = {}
    

    while len(highest) < count:
        
        topWord = None
        highestProb = -1
        
        for i in range(len(words)):
            if words[i] not in ignoreList and words[i] not in highest:
                if probs[i] > highestProb:
                    topWord = words[i]
                    highestProb = probs[i]
        
        highest[topWord] = highestProb
        
                
    return highest


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    
    sentence = []
    
    for i in range(count):
        word = choices(words, weights=probs)
        sentence.append(word[0])
        
    separated = " ".join(sentence)
    
    return separated


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    
    sent = []
    
    
    for i in range(count):
        if sent == [] or sent[-1] in ".!?":
            word = choices(startWords, weights=startWordProbs)
            sent.append(word[0])
        else:
            lastWord = sent[-1] #latest word added to list
            
            word = choices(bigramProbs[lastWord]["words"], weights=bigramProbs[lastWord]["probs"])
            sent.append(word[0])
            
    separated = " ".join(sent)
    
    return separated


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    
    unigramCount = countUnigrams(corpus)
    totalCount = getCorpusLength(corpus)
    vocab = buildVocabulary(corpus)
    
    probs = buildUnigramProbs(vocab, unigramCount, totalCount)
    
    topWords = getTopWords(50, vocab, probs, ignore)
    
    
    barPlot(topWords, "Top 50 Most Frequent Words in Corpus")
        
    
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    
    starters = makeStartCorpus(corpus)
    vocab = buildVocabulary(starters)
    totalCount = len(starters)
    unigramCount = countUnigrams(starters)

    
    probs = buildUnigramProbs(vocab, unigramCount, totalCount)
    
    
    topWords = getTopWords(50, vocab, probs, ignore)
    
    
    barPlot(topWords, "Top 50 Most Frequent Start Words in Corpus")
    
    
    
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    
    
    unigramCounts = countUnigrams(corpus)
    bigramCounts = countBigrams(corpus)
    
    vocabNext = buildBigramProbs(unigramCounts, bigramCounts)[word]["words"]
    
    probs = buildBigramProbs(unigramCounts, bigramCounts)[word]["probs"]
        
    topNextWords = getTopWords(10, vocabNext, probs, ignore)
    
    wordTitle = "'" + word.capitalize() + "'"
    
    barPlot(topNextWords, "Top 10 Words After " + wordTitle)
    
    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    vocab1 = buildVocabulary(corpus1)
    unigramCount1 = countUnigrams(corpus1)
    totalCount1 = getCorpusLength(corpus1)
    probs1 = buildUnigramProbs(vocab1, unigramCount1, totalCount1)
    
    vocab2 = buildVocabulary(corpus2)
    unigramCount2 = countUnigrams(corpus2)
    totalCount2 = getCorpusLength(corpus2)
    probs2 = buildUnigramProbs(vocab2, unigramCount2, totalCount2)
    
    corp1Top = getTopWords(topWordCount, vocab1, probs1, ignore)
    corp2Top = getTopWords(topWordCount, vocab2, probs2, ignore)
    
    topWords = []
    
    for key in corp1Top:
        topWords.append(key)
    for key in corp2Top:
        if key not in topWords:
            topWords.append(key)
            
    
    probs1_dict = {}
    for i in range(len(vocab1)):
        probs1_dict[vocab1[i]] = probs1[i]
        
    probs2_dict = {}
    for i in range(len(vocab2)):
        probs2_dict[vocab2[i]] = probs2[i]
        
    corpus1Probs = []
    corpus2Probs = []
    
    for word in topWords:
        if word in probs1_dict:
            corpus1Probs.append(probs1_dict[word])
        else:
            corpus1Probs.append(0)
            
        if word in probs2_dict:
            corpus2Probs.append(probs2_dict[word])
        else:
            corpus2Probs.append(0)
        
    combined = {"topWords" : topWords, "corpus1Probs" : corpus1Probs, "corpus2Probs" : corpus2Probs}
    
    return combined


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    
    data = setupChartData(corpus1, corpus2, numWords)
    xVals = data["topWords"]
    val1 = data["corpus1Probs"]
    val2 = data["corpus2Probs"]
    
    sideBySideBarPlots(xVals, val1, val2, name1, name2, title)
    
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    
    data = setupChartData(corpus1, corpus2, numWords)
    
    
    label = data["topWords"]
    xVal = data["corpus1Probs"]
    yVal = data["corpus2Probs"]
    
    
    scatterPlot(xVal, yVal, label, title)
    
    return







### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work


if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
