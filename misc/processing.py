def processing(func):
    def wrapper(*args, **kwargs):
        print('-' * 20)
        f = func(*args, **kwargs)
        print(f"Processing {func.__name__} of {func.__class__}.")
        print('-' * 20)
        return f

    return wrapper