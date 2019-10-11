import threading


started_threads = []


def do_work(fn, *args, in_background=True):
    if in_background:
        t = threading.Thread(target=fn, args=args)
        t.start()
        started_threads.append(t)
    else:
        fn(*args)


def join_all():
    for t in started_threads:
        t.join()
