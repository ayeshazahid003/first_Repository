# 30-9-23
# CSC461 – Assignment 2 – – Regular Expressions
# Ayesha Zahid
# Fa21-BSE-003
# Read the file (example-text.txt) and write regular expressions to accomplish the following.
import re

# Read the content from the file
with open('example-text.txt', 'r') as file:
    text = file.read()

# 1. Extract list of all words.
all_words = re.findall(r'\b\w+\b', text)

# 2. Extract list of all words starting with a capital letter.
capital_words = re.findall(r'\b[A-Z][a-z]*\b', text)

# 3. Extract list of all words of length 5.
five_letter_words = re.findall(r'\b\w{5}\b', text)

# 4. Extract list of all words inside double quotes.
quoted_words = re.findall(r'"([^"]+)"', text)

# 5. Extract list of all vowels.
vowels = re.findall(r'[aeiouAEIOU]', text)

# 6. Extract list of 3 letter words ending with letter ‘e’.
three_letter_e_words = re.findall(r'\b\w{3}e\b', text)

# 7. Extract list of all words starting and ending with letter ‘b’.
b_words = re.findall(r'\b[bB]\w*[bB]\b', text)

# 8. Remove all the punctuation marks from the text.
text_without_punctuation = re.sub(r'[^\w\s]', '', text)

# 9. Replace all words ending ‘n't’ to their full form ‘not’.
text_replaced = re.sub(r'\b\w+n\'t\b', 'not', text)

# 10. Replace all the new lines with a single space.
text_single_space = re.sub(r'\n', ' ', text)

# Print the results
print("1. All Words:", all_words)
print("\n2. Capital Words:", capital_words)
print("\n3. Five-Letter Words:", five_letter_words)
print("\n4. Quoted Words:", quoted_words)
print("\n5. Vowels:", vowels)
print("\n6. Three-Letter Words Ending with 'e':", three_letter_e_words)
print("\n7. Words Starting and Ending with 'b':", b_words)
print("\n8. Text without Punctuation:\n", text_without_punctuation)
print("\n9. Text with 'n't' replaced with 'not':\n", text_replaced)
print("\n10. Text with New Lines Replaced by Spaces:\n", text_single_space)
