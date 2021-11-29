

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 将单词序列转化为数据元组列表，
# 其中的每个元组格式为([ word_i-2, word_i-1 ], target word)
trigrams = [ ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence) - 2) ]

# 打印出前3条数据，注意观察数据的结构
#print(trigrams[:3])

# set 即去除重复的词
vocab = set(test_sentence)
# 建立词典，它比单词表多了每个词的索引
word_to_ix = { word: i for i, word in enumerate(vocab) }
ix_to_word = { i:word for i, word in enumerate(vocab)}


