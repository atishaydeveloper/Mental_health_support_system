import json

with open('intents1.json') as file:
    intents = json.load(file)

for intent in intents['intents']:
    for response in intent['responses']:
        if response =="I'm very sorry to hear that but you have so much to look forward to.you only live once so and there is a lot more you can do, you dont realise it now but I will help you to figure things out Please seek help by contacting: 9152987821.":
            print(intent['tag'])    