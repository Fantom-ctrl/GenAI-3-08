import csv                            
import torch                             
from transformers import AutoModelForCausalLM, AutoTokenizer 
import logging                          
from typing import Optional             

# Логирование
logging.basicConfig(level=logging.INFO) # Настройка базовой конфигурации логирования
logger = logging.getLogger(__name__)    # Получаем логгер для текущего модуля

# Выбор устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# ===== Класс генератора =====
class Generator:
    def __init__(self, model_name: str = "sambanovasystems/SambaLingo-Russian-Chat"):
        # конструктор класса: принимает имя модели
        self.model_name = model_name # Сохраняем имя модели в атрибут экземпляра
        self.model = None            # Будет хранить загруженный объект модели
        self.tokenizer = None        # Будет хранить токенизатор
        self.device = device         # Сохраняем выбранное устройство (cuda/cpu) в атрибут для дальнейшего использования

    def load_model(self) -> bool:
        """Загрузка модели и токенизатора"""
        try:
            print("Загрузка модели...")  # уведомление о начале загрузки
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if device=="cuda" else None, # для CPU отключаем auto
                torch_dtype="auto"
            )
            print("Модель загружена.")  # подтверждение успешной загрузки
            return True                 # Возвращаем True при успехе
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}") # Исключение с подробностями
            return False                # При ошибке возвращаем False

    def generate_dialog(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Optional[str]:
        """
        Генерация диалога на основе промпта.
        Параметры:
        - prompt: текстовый запрос/инструкция
        - max_new_tokens: макс. число новых токенов, которые модель может сгенерировать
        - temperature: параметр "температуры" для сэмплинга (чем выше — тем разнообразнее)
        - top_p: nucleus-сэмплинг (оставляет минимальную совокупность токенов с суммарной prob >= top_p)
        - repetition_penalty: штраф за повторения (увеличивает разнообразие по повторениям)
        Возвращает: сгенерированный текст или None при ошибке.
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
                max_length=1024  # Максимум 1024 токена
            )

            # Преобразование в тензор и перенос на устройство
            inputs = torch.tensor([tokens]).to(self.device)
            attention_mask = torch.ones_like(inputs) # Маска внимания

            # Генерация текста
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,       # добавлено для стабильности
                    max_new_tokens=max_new_tokens,       # ограничиваем длину генерации
                    do_sample=True,                      # включаем вероятностное сэмплирование
                    temperature=temperature,             # контролируем "творчество"
                    top_p=top_p,                          # nucleus sampling
                    repetition_penalty=repetition_penalty, # штрафуем за повторы
                    pad_token_id=self.tokenizer.eos_token_id # задаём токен паддинга
                )

            # Декодирование
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True # убираем служебные токены
            )

            # Извлечение ответа ассистента
            response = generated_text.split("<|assistant|>")[-1].strip()
            return response if response else None

        except Exception as e:
            logger.error("Ошибка при генерации текста: %s", str(e))
            return None # Ловим ошибки и возвращаем None

# Вспомогательные функции
def split_dialog(text: str):
    """Разделение текста на реплики — убираем пустые строки и обрезаем пробелы"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines

def check_replicas_length(lines, max_words=15):
    """Проверка длины реплик и вычисление метрики — доля реплик, не превышающих max_words"""
    too_long = [line for line in lines if len(line.split()) > max_words] # Список длинных реплик
    valid_count = len(lines) - len(too_long)  # Число "коротких" реплик
    metric = valid_count / len(lines) * 100 if lines else 0
    return too_long, metric

def save_result_csv(prompt: str, dialog: list, filename: str = "result.csv"):
    """Сохранение результата в CSV"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Role", "Replica"])  # шапка CSV
        for i, line in enumerate(dialog):
            role = "User" if i % 2 == 0 else "Bot" # Предполагаем попеременный формат реплик
            writer.writerow([prompt, role, line])
    print(f"Результат сохранён в {filename}")  # Уведомление о сохранении

# Главная функция генерации 
def run_dialog_generation(prompt: str, generator: Generator, max_words=15, n_attempts=3, **kwargs):
    """Генерация с автоперегенерацией и сохранением в CSV"""
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
    prompt_text = "Поддержка" # Начальный промпт

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
