import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_birthday_message(name):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Create a prompt for the model
    prompt = f"Let's all wish {name} a happy birthday! Here's a creative message:"

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    message = tokenizer.decode(output[0], skip_special_tokens=True)

    return message

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a name as a command-line argument.")
        sys.exit(1)
    
    name = " ".join(sys.argv[1:])
    birthday_message = generate_birthday_message(name)
    print(birthday_message)