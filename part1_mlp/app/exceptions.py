class NeuralNetworkError(Exception):
    """Базовое исключение для нейросети."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Некорректный размер слоя."""


class MismatchedDataError(NeuralNetworkError):
    """Несоответствие размерности данных и сети."""


class DataFileNotFoundError(NeuralNetworkError):
    """Файл не найден."""


class TrainingError(NeuralNetworkError):
    """Ошибка в процессе обучения."""
