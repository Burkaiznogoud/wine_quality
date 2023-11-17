def evaluation(func):
    def wrapper(*args, **kwargs):
        print('-' * 20)
        d = func(*args, **kwargs)
        for k, v in d.items():
            print(f"{k} {v}")
        print('-' * 20)
        return d
    return wrapper
