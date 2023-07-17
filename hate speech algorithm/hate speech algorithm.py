import re

# Define the hate speech detection algorithm
def detect_hate_speech(input_text):
    # List of hate speech keywords or patterns
    hate_keywords = ['hate', 'racist', 'mbwa', 'discriminate', 'dog', 'offensive']
    
    # Check if any of the hate keywords are present in the input text
    for keyword in hate_keywords:
        if re.search(r'\b{}\b'.format(keyword), input_text, re.IGNORECASE):
            return True
    
    return False

# Example usage
def filter_input(input_text):
    # Apply hate speech detection algorithm
    if detect_hate_speech(input_text):
        print("Input contains hate speech. Please provide a different input.")
    else:
        # Proceed with database interaction
        # ...
        print("Input passed the hate speech filter and can be stored in the database.")

# Test the filter_input function
input1 = "I hate that person!"
filter_input(input1)

input2 = "I like dogs."
filter_input(input2)
