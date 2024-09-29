import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
rutube_tags = []
tag_path = 'tag_list.txt'
with open(tag_path, 'r', encoding='utf-8') as file:
    for line in file:
        rutube_tags.append(line.strip())
video_path = '/home/ubuntu/rutube-hack/test_videos'
DEFAULT_SYSTEM_PROMPT = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, не додумывай, желательно 1-3 тега. Теги должны быть разделены запятой. Ты можешь использовать только теги из этого списка: {rutube_tags}. Если нет подходящих то выведи тег nan. От качества тегов зависит моя работа!"

class LLM():
    def __init__(self, 
                 model_name=MODEL_NAME,
                system_prompt=DEFAULT_SYSTEM_PROMPT
                ):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            #load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.system_prompt = system_prompt
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    def ask_a_question(self, query: str):
        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": self.system_prompt
        }, {
            "role": "user",
            "content": query
        }], tokenize=False, add_generation_prompt=True)
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, generation_config=self.generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(output)