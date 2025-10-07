import csv                            
import torch                             
from transformers import AutoModelForCausalLM, AutoTokenizer 
import logging                          
from typing import Optional             

# Логирование
logging.basicConfig(level=logging.INFO) # Настройка базовой конфигурации логирования
logger = logging.getLogger(__name__) # Получаем логгер для текущего модуля

# Выбор устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Класс генератора 
class Generator:
    """
    Класс генератора текста/диалогов с использованием модели AutoModelForCausalLM.
    
    Атрибуты:
        model_name (str): Имя модели HuggingFace
        model (AutoModelForCausalLM): Загруженная модель
        tokenizer (AutoTokenizer): Токенизатор модели
        device (str): Устройство для генерации (cuda или cpu)
    """
    def __init__(self, model_name: str = "sambanovasystems/SambaLingo-Russian-Chat"):
        """
        Инициализация генератора.
        
        Аргументы:
            model_name (str): Имя модели HuggingFace.
        """
        self.model_name = model_name # Сохраняем имя модели в атрибут экземпляра
        self.model = None # Будет хранить загруженный объект модели
        self.tokenizer = None # Будет хранить токенизатор
        self.device = device # Сохраняем выбранное устройство (cuda/cpu) в атрибут для дальнейшего использования

    def load_model(self) -> bool:
        """
        Загружает модель и токенизатор.
        
        Возвращает:
            bool: True, если модель успешно загружена, иначе False.
        """
        try:
            print("Загрузка модели...") # Уведомление о начале загрузки
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if device=="cuda" else None, # Для CPU отключаем auto
                torch_dtype="auto"
            )
            print("Модель загружена.") # Подтверждение успешной загрузки
            return True # Возвращаем True при успехе
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}") # Исключение с подробностями
            return False # При ошибке возвращаем False

    def generate_dialog(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.1) -> Optional[str]:
        """
        Генерация диалога с повторными попытками и сохранением результата.
    
        Аргументы:
            prompt (str): Начальный промпт пользователя.
            generator (Generator): Экземпляр класса Generator.
            max_words (int): Максимальное количество слов для короткой реплики.
            n_attempts (int): Число попыток генерации.
            **kwargs: Дополнительные параметры генерации.
        
        Возвращает:
            None
        """
        if not self.model or not self.tokenizer: # Проверяем, загружены ли модель и токенизатор.
            logger.error("Модель не загружена")
            return None # Выходим, чтобы избежать ошибки при генерации

        try:
            # Форматирование промпта
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n" # Специальные токены user/assistant

            # Токенизация
            tokens = self.tokenizer.encode(
                formatted_prompt,
                truncation=True, # Обрезать если длиннее допустимого
                max_length=1024 # Максимум 1024 токена
            )

            # Преобразование в тензор и перенос на устройство
            inputs = torch.tensor([tokens]).to(self.device)
            attention_mask = torch.ones_like(inputs) # Маска внимания

            # Генерация текста
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask, # Добавлено для стабильности
                    max_new_tokens=max_new_tokens, # Ограничиваем длину генерации
                    do_sample=True, # Включаем вероятностное сэмплирование
                    temperature=temperature, # Контролируем "творчество"
                    top_p=top_p,                          
                    repetition_penalty=repetition_penalty, # Штрафуем за повторы
                    pad_token_id=self.tokenizer.eos_token_id # Задаём токен паддинга
                )

            # Декодирование
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True # Убираем служебные токены
            )

            # Извлечение ответа ассистента
            response = generated_text.split("<|assistant|>")[-1].strip()
            return response if response else None

        except Exception as e:
            logger.error("Ошибка при генерации текста: %s", str(e))
            return None # Ловим ошибки и возвращаем None

# Вспомогательные функции
def split_dialog(text: str):
    """
    Разделение текста на реплики.
    
    Аргументы:
        text (str): Сгенерированный текст диалога.
    
    Возвращает:
        List[str]: Список непустых строк-реплик.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines

def check_replicas_length(lines, max_words=15):
    """
    Проверка длины реплик и вычисление метрики качества.
    
    Аргументы:
        lines (List[str]): Список реплик диалога.
        max_words (int): Максимальное число слов, чтобы реплика считалась короткой.
    
    Возвращает:
        Tuple[List[str], float]: Список длинных реплик и доля реплик, удовлетворяющих условию, в процентах.
    """
    too_long = [line for line in lines if len(line.split()) > max_words] # Список длинных реплик
    valid_count = len(lines) - len(too_long) # Число "коротких" реплик
    metric = valid_count / len(lines) * 100 if lines else 0
    return too_long, metric

def save_result_csv(prompt: str, dialog: list, filename: str = "result.csv"):
    """
    Сохраняет диалог в CSV-файл.
    
    Аргументы:
        prompt (str): Исходный промпт пользователя.
        dialog (List[str]): Список реплик диалога.
        filename (str): Имя CSV-файла для сохранения.
    
    Возвращает:
        None
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Role", "Replica"]) # Шапка CSV
        for i, line in enumerate(dialog):
            role = "User" if i % 2 == 0 else "Bot" # Предполагаем попеременный формат реплик
            writer.writerow([prompt, role, line])
    print(f"Результат сохранён в {filename}") # Уведомление о сохранении

# Главная функция генерации 
def run_dialog_generation(prompt: str, generator: Generator, max_words=15, n_attempts=3, **kwargs):
    """
    Генерация диалога с повторными попытками и сохранением результата.
    
    Аргументы:
        prompt (str): Начальный промпт пользователя.
        generator (Generator): Экземпляр класса Generator.
        max_words (int): Максимальное количество слов для короткой реплики.
        n_attempts (int): Число попыток генерации.
        **kwargs: Дополнительные параметры генерации (max_new_tokens, temperature и т.д.).
    
    Возвращает:
        None
    """
    best_dialog = None
    best_metric = 0

    for attempt in range(1, n_attempts + 1):
        print(f"\nПопытка {attempt}...")
        raw_dialog = generator.generate_dialog(prompt, **kwargs)
        if not raw_dialog:
            print("Не удалось сгенерировать диалог")
            continue

        dialog_lines = split_dialog(raw_dialog)
        too_long, metric = check_replicas_length(dialog_lines, max_words=max_words)
        print(f"Метрика: {metric:.2f}% реплик ≤ {max_words} слов")

        if metric > best_metric:
            best_metric = metric
            best_dialog = dialog_lines

        if metric >= 90:
            print("Условие метрики выполнено, выходим из цикла.")
            break

    if best_dialog:
        print("\n" + "=" * 50)
        print(f"Диалог на тему: {prompt.upper()}")
        print("=" * 50)
        for line in best_dialog:
            print(line)
        save_result_csv(prompt, best_dialog)
    else:
        print("Не удалось получить качественный диалог")

def main():
    """
    Главная функция скрипта.
    
    Выполняет:
        - Создание экземпляра Generator
        - Загрузку модели
        - Генерацию диалога и сохранение результата
    """
    prompt_text = "Разговор между собакой и кошкой" # Начальный промпт

    generation_params = {
        "max_new_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }

    # Загружаем модель один раз
    generator = Generator()
    if generator.load_model():
        run_dialog_generation(prompt_text, generator, max_words=15, n_attempts=3, **generation_params)


if __name__ == "__main__":
    main()
