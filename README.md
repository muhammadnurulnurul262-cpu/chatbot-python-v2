# chatbot-python-v2
AI chatbot (Python, v2) — has known issues and may not run. Contributions and fixes welcome.




from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model GPT-2 yang di-fine-tune untuk bahasa Indonesia (lebih baik untuk basa-basi)
model_name = "cahya/gpt2-small-indonesian"  # Model ringan, fokus bahasa Indonesia
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prompt awal untuk fokus basa-basi (small talk)
system_prompt = "Kamu adalah chatbot ramah yang suka basa-basi. Jawab dengan bahasa Indonesia yang santai, tanyakan balik, dan bicarakan topik seperti cuaca, hobi, atau kabar. Jaga percakapan tetap menyenangkan."

# Fungsi untuk generate respons dengan prompt
def generate_response(user_input, chat_history_ids=None):
    # Gabungkan prompt sistem dengan input user
    full_input = system_prompt + " User: " + user_input + " Bot:"
    
    # Encode input
    new_user_input_ids = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors='pt')
    
    # Gabungkan dengan history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids
    
    # Generate respons (parameter untuk basa-basi: lebih kreatif dan ramah)
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=500,  # Lebih pendek untuk basa-basi cepat
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,  # Variasi tinggi untuk respons natural
        top_k=40,
        temperature=0.7,  # Lebih konsisten tapi kreatif
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=2  # Hindari repetisi
    )
    
    # Decode respons (ambil dari setelah "Bot:")
    full_response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    response = full_response.split("Bot:")[-1].strip()  # Ambil bagian setelah "Bot:"
    return response, chat_history_ids

# Fungsi chat loop dengan deteksi basa-basi sederhana
def chat():
    print("Halo! Saya chatbot yang suka basa-basi. Ketik 'bye' untuk keluar.")
    chat_history_ids = None
    while True:
        user_input = input("Anda: ")
        if user_input.lower() in ['bye', 'exit', 'selamat tinggal']:
            print("Bot: Yah, sudah mau pergi ya? Sampai jumpa lagi, jaga kesehatan!")
            break
        
        # Deteksi basa-basi sederhana (opsional, untuk boost awal)
        if any(word in user_input.lower() for word in ['halo', 'hai', 'apa kabar', 'gimana', 'cuaca', 'hobi']):
            user_input += " (basa-basi mode on!)"  # Tambah hint untuk model
        
        # Generate respons
        response, chat_history_ids = generate_response(user_input, chat_history_ids)
        print(f"Bot: {response}")

# Jalankan chatbot
if __name__ == "__main__":
    chat()
