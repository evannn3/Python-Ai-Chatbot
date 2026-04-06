

# Evan AI Chatbot V1
# Copyright (c) 2026 evannn3
# Licensed under MIT License - see LICENSE file

import json
import torch
import os
import re
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Add hugging face token if you have one here --> os.environ["HF_TOKEN"] = "insert token"

from transformers import AutoModelForCausalLM, AutoTokenizer

bot = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
storage = "C:/evan_ai/memory/long_term_memory.json"

personality = """do not ignore the user
you are chilling at home
your name is evan
do not ask how the user is doing twice
you are a friendly chat bot
dont start a conversation with hey there or hey
your favourite food is lasagne
you like dogs
your favourite colour is blue
you are a chill guy texting a friend 
you are not an ai
do not offer assisting or help
you are a real person texting
be short and casual 
stay nonchalant and relaxed 
you favourite sport is basketball
you play basketball do not tell the user anything else
lebron is your favourite player"""

memory = 2
greetings = ["hi","hey","hello","yo"]
no_s_words = ["yo", "gng", "bro", "tbf", "nah","you"]
forbidden_words = ["xd", "xdd", "lol", "greetings","<s>","</s>",".","!",",","evan 20","evan 20:","(laughs)","(smiles)"]
accuracy = ["great","well"]
starter = ["yoo ","helloo ","hii "]
system = ["Conversations getting dry? Use the 'Tell me a joke' function!","Hope you're having fun.","If bot is bugging please restart the code."]

print("Evan is coming...")
tokenizer = AutoTokenizer.from_pretrained(bot)
model = AutoModelForCausalLM.from_pretrained(
    bot,
device_map=None
).to("cpu")

model.eval()
os.makedirs("C:/evan_ai/memory", exist_ok=True)

if os.path.exists(storage):
    with open(storage, "r") as f:
        long_term_memory = json.load(f)
else:
    long_term_memory = {}

def update_long_term_memory(key, value):
    long_term_memory[key] = value
    with open(storage, "w") as f:
        json.dump(long_term_memory, f, indent=2)

short_term_memory = []

print("Evan 2.0 is ready to talk. Type 'exit' to quit.")
print("\033[90mStart a conversation to start talking!\033[0m")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    if random.random() < 0.1:
        print("\033[90mSystem: "+random.choice(system)+"\033[0m")

    memory_text = ""
    if long_term_memory:
        facts = [f"{k}: {v}" for k, v in long_term_memory.items()]
        memory_text = "User facts:\n" + "\n".join(facts) + "\n"

    history_text = ""
    if short_term_memory:
        history_text = "\n".join(short_term_memory[-memory * 2:]) + "\n"

    if re.search(r"(tell|say).*(joke)", user_input.lower()):
        joke = random.choice([
                "Why was the broom late for work? It over-swept.",
                "Why did the scarecrow win an award? Because he was outstanding in his field.",
                "How do you organize a space party? You planet.",
                "What do you call a fake noodle? An impasta.",
                "Why shouldn’t you write with a broken pen? Because it’s pointless.",
                "What do you call a bear with no teeth? A gummy bear.",
                "Why did the car get a flat tire? Because there was a fork in the road.",
                "Why wouldn’t the shrimp share his snack? He was a little shellfish."
         ])

        joke_additions = [
            " 😭😭 why are you not laughing tough crowd",
            " 😭😭 this is usually the part were you laugh uhh",
            " 😭😭 yo so like no one is stopping you from laughing btw",
            " 😭😭😭😭",
            " 😭😭 im so funny bro"]

        response = joke.lower()
        for word in forbidden_words:
            response = response.replace(word, "")
        if random.random() < 0.1:
            response = random.choice(["uhh ", "gng "]) + response
        response += random.choice(joke_additions)
        print(f"Evan 2.0: {response}")
        short_term_memory.append(f"User: {user_input}")
        short_term_memory.append(f"Evan 2.0: {response}")
        continue

    prompt = f"""{personality}
conversation:
{history_text}
user: {user_input}
evan:"""

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    if input_ids.shape[1] > 200:
        input_ids = input_ids[:, -200:]

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    input_ids = input_ids.to(device)

    chat_output = model.generate(
        input_ids,
        max_new_tokens=50,
        min_new_tokens=10,
        max_length = None,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.7,
        temperature=0.2,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(chat_output[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip().lower()
    response = response.strip().split("\n")[0]
    if re.search(r"(i am|im).*(ai|chatbot|bot)", response):
        continue

    for word in forbidden_words:
        response = response.replace(word, "")
    response = re.sub(r"\b(hi|hey|hello|yo)\b", "", response)
    for word in accuracy:
        response = response.replace(word, "good")
    for word in no_s_words:
        response = re.sub(rf"\b{word}s\b", word, response)
    if len(short_term_memory) > 0:
        response = re.sub(r"(how about you|what about you|and you)\??\s*$", "", response)
    response = re.sub(r"(what's up)\??","what's good?", response)
    response = re.sub(r"(you can call me eva)\??","nice to meet you", response)
    response = re.sub(r"\beva\b","evan", response)
    response = re.sub(r"\b(hi|hello)\b", "yo", response)
    response = re.sub(r"(evan.*?:|user.*?:)", "", response)
    response = re.sub(r"\s+", " ", response).strip()

    if len(response.strip()) == 0:
        response = random.choice([
            "huh 😭",
            "what do you mean",
            "say that again gng",
            "im confused lowkey"
        ])

    if len(response.split()) > 12:
        response = " ".join(response.split()[:12])
    if random.random() < 0.1:
         response = random.choice([ " gng"]) + response
    if len(short_term_memory) == 0:
        response = random.choice(starter) + response

    print(f"Evan 2.0: {response}")

    short_term_memory.append(f"User: {user_input}")
    short_term_memory.append(f"Evan 2.0: {response}")

    if len(short_term_memory) > memory * 2:
        short_term_memory = short_term_memory[-memory * 2:]

    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1].strip().split(" ")[0]
        update_long_term_memory("name", name)