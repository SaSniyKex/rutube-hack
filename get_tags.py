from faster_whisper import WhisperModel
from models import llm
from transformers import pipeline, AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import pandas as pd
import os

video_path = '/home/ubuntu/rutube-hack/test_videos'
whisper_model = WhisperModel('Systran/faster-whisper-large-v3', device='cuda')
print('WHISPER_UPLOADED')

summarizator = pipeline(task="summarization", model='cointegrated/rut5-base-absum', device = 'cuda')
print('SUMMARIZATOR_UPLOADED')
api_key = 'sk-or-vv-ed9b0360080c7f2ebfea7bb6d596cc6f67d2d2d8356b6cd950e66379b63c2d87'
client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
)
def generate_tags(video_description):
    system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, не додумывай, желательно 1-3 тега. Теги должны быть разделены запятой. Ты можешь использовать только теги из этого списка: {rutube_tags}. Если нет подходящих то выведи тег nan. От качества тегов зависит моя работа!"
    prompt = video_description
    messages = [{"role":"system", "content":system_prompt}]
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        n=1,            # Number of responses
        stop=None,      # Stopping condition
    )
    tags = response.choices[0].message.content
    return [tag.strip() for tag in tags.split(',')]

<<<<<<< Updated upstream
results = []

def process_video(file_path):
    file_tags = set()
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    text = ''.join(segment.text for segment in segments)
    summary = summarizator(text, min_length = 100, max_length = 2000)
    print(summary)
    '''
    for chunk in chunked_text:
        tags = generate_tags(chunk)
        for t in tags:
            file_tags.add(t)'''
    return None
tag_path = '/home/ubuntu/rutube-hack/tag_list.txt'
rutube_tags = []
with open(tag_path, 'r', encoding='utf-8') as file:
    for line in file:
        rutube_tags.append(line.strip())

for filename in os.listdir(video_path):
    file_path = os.path.join(video_path, filename)
    result = process_video(file_path)
    results.append({'filename': filename, 'tags': result})
=======
class AudioProcesser():
    def __init__(self):
        self.whisper_model = WhisperModel('Systran/faster-whisper-large-v3', device='cuda')
        print('Whisper downloaded')
        self.embedder = HuggingFaceEmbeddings(model_name='sergeyzh/rubert-tiny-turbo')
        self.splitter = SemanticChunker(self.embedder, breakpoint_threshold_type = 'percentile', breakpoint_threshold_amount=87)
        self.embedder_for_tags = HuggingFaceEmbeddings(model_name='deepvk/USER-bge-m3')
        api_key = 'sk-or-vv-ed9b0360080c7f2ebfea7bb6d596cc6f67d2d2d8356b6cd950e66379b63c2d87'
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.vsegpt.ru/v1",
        )
        print('Openai connected')
    def generate_tags(self, chunk_description, video_description, llm):

        system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, обобщенные теги, сильно не конкретизируй, не додумывай, желательно 1-3 тега. В случае если тебе не нравится текст по любым причинам, ответь пустой строкой. Теги должны быть разделены запятой. Вот название и описание видео, чтобы ты знал контекст: {video_description}"
        prompt = chunk_description
        #messages = [{"role": "system", "content": system_prompt}]
        #messages.append({"role": "user", "content": prompt})

        try:
            response = llm.ask_a_question(prompt, system_prompt=system_prompt)
            tags = response
            if not tags:
                return []
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        except:
            print(f"Модерация отклонила запрос для чанка: {video_description}")
            return []
    def process_video(self, file, video_description, llm):
        file_tags = set()
        
        # Транскрипция видео
        segments, info = self.whisper_model.transcribe(file, beam_size=5)
        text = ' '.join(segment.text for segment in segments)
        print('Video transcribed')
        # Разделение текста на чанки
        chunked_text = self.splitter.split_text(text)
        print(len(chunked_text))
        
        # Получаем эмбеддинги для чанков
        chunk_embeddings = self.embedder.embed_documents(chunked_text)
        video_description_embedding = self.embedder.embed_query(video_description)
        
        # Считаем сходства между эмбеддингами и описанием видео
        similarities = cosine_similarity(chunk_embeddings, [video_description_embedding])
        similarity_threshold = 0.8
        
        for i, sim in enumerate(similarities):
            if sim > similarity_threshold:
                chunk = chunked_text[i]
                
                tags = self.generate_tags(chunk, video_description, llm)
                
                # Если теги есть, добавляем их в множество
                if tags:
                    for t in tags:
                        file_tags.add(t)
                
        
        return file_tags
    
    def get_nearest_tags(self, file_tags, rutube_tags, video_description, similarity_threshold=0.5, top_n=3):
        if len(file_tags)>0:
            combined_tags = ', '.join(file_tags)
        combined_tags += f', {video_description}'
        generated_embedding = self.embedder_for_tags.embed_query(combined_tags)
        available_tag_embeddings = self.embedder_for_tags.embed_documents(rutube_tags)
        similarities = cosine_similarity([generated_embedding], available_tag_embeddings)[0]
        nearest_tags = [(rutube_tags[i], similarities[i]) for i in range(len(rutube_tags)) if similarities[i] > similarity_threshold]
        nearest_tags = sorted(nearest_tags, key=lambda x: x[1], reverse=True)
        nearest_tags = nearest_tags[:top_n]
        return [tag for tag, similarity in nearest_tags]
>>>>>>> Stashed changes

# Создаем DataFrame из результатов
df = pd.DataFrame(results)

# Сохраняем DataFrame в CSV файл (по желанию)
df.to_csv('video_processing_results.csv', index=False)

