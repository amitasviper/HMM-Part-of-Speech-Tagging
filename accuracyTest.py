def getFileContents(filename):
	data = None
	with open(filename, 'r') as f:
		data = f.readlines()
	return data

def computeAccuracy():
	dev_tagged_data = getFileContents('data/zh_dev_tagged.txt')
	predicted_data = getFileContents('hmmoutput.txt')
	correct = 0
	total = 0
	for index, line in enumerate(dev_tagged_data):
		predicted_tagged_line = predicted_data[index]
		expected_tagged_line = dev_tagged_data[index]
		
		predicted_word_tag_pairs = predicted_tagged_line.strip().split(' ')
		expected_word_tag_pairs = expected_tagged_line.strip().split(' ')
		for index, predicted_word in enumerate(predicted_word_tag_pairs):
			if predicted_word == expected_word_tag_pairs[index]:
				correct += 1
			total += 1
			if total % 100 == 0:
				print correct, total, " => ", (correct*100.0)/total
	accuracy = (correct*100.0)/total
	print accuracy

if __name__ == '__main__':
	computeAccuracy()