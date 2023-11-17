def processing(func):
    def wrapper(*args, **kwargs):
        print('-' * 20)
        print(f"Processing {func.__name__} of {func.__class__}.")
        print('-' * 20)
        return

    return wrapper