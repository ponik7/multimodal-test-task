# Test Task

### Установка зависимостей
```bash
pip install -r requirements.txt
```


### Подготовка данных
Свои изображения можно добавить в `data/images/`. Также нужно будет обновить `data/mapping.json`, чтобы он содержал промпт для каждого изображения.

Пример `mapping.json`:
```json
{
    "1.jpg": "What are these people doing? Answer with one word.",
    "2.jpg": "What's the number on left's man t-shirt? Answer with one word."
}
```

### Запуск скрипта
```bash
python src/run_exp.py --model_id <MODEL_ID> --im_path data/images/1.jpg
```

- `<MODEL_ID>`: Идентификатор модели Hugging Face (например, `your-org/your-model`).
- `--im_path`: Путь до изображения.


### Результат
Результат в виде интерактивного HTML находится в папке `vis/`.