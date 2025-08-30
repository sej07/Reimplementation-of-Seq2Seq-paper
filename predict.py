import sys
from inference import translate

input_sentence = sys.argv[1]
output = translate(input_sentence)
print(f"🇫🇷 Translation: {output}")