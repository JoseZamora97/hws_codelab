from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor

import ipyparallel as ipp

import ejercicio7 as ex
from utils import timeit, split_array


class Tester:
    def __init__(self, detector: str, matcher: str, image_path: str,
                 dataset_path: str, dataset_size: int, ):
        self.detector, self.matcher = detector, matcher
        self.image_path, self.dataset_path = image_path, dataset_path
        self.dataset_size = dataset_size

        self.to_analyse = ex.load_dataset(self.dataset_path)

    def run(self):
        raise RuntimeError("Not implemented yet")


class SequentialTester(Tester):
    def __init__(self, detector: str, matcher: str, image_path: str,
                 dataset_path: str, dataset_size: int):
        super().__init__(detector, matcher, image_path, dataset_path, dataset_size)

    @timeit
    def run(self):
        return ex.search_candidates(self.image_path,
                                    self.to_analyse[:self.dataset_size],
                                    self.detector, self.matcher, ex.MATCHES_THRESHOLD_TO_SHOW)


class ParallelTester(Tester):
    def __init__(self, detector: str, matcher: str, image_path: str,
                 dataset_path: str, dataset_size: int, workers: int):
        super().__init__(detector, matcher, image_path, dataset_path, dataset_size)

        self.workers = workers
        self.to_analyse = [dict(search=self.image_path, covers=segment, detector=self.detector,
                                matcher=self.matcher, min_matches=ex.MATCHES_THRESHOLD_TO_SHOW)
                           for segment in split_array(self.to_analyse[:self.dataset_size],
                                                      self.dataset_size // self.workers)]

    @staticmethod
    def helper(args):
        import ejercicio7 as ex
        return ex.search_candidates(**args)

    def run(self):
        raise RuntimeError("Not implemented yet")


class ThreadingTester(ParallelTester):
    @timeit
    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(ParallelTester.helper, args)
                       for args in self.to_analyse]
            return [candidate for future in as_completed(futures) for candidate in future.result()]


class MultiprocessingTester(ParallelTester):
    @timeit
    def run(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(ParallelTester.helper, self.to_analyse)
            return [candidate for result in results for candidate in result]


class IPyParallelTester(ParallelTester):
    @timeit
    def run(self):
        candidates = []

        try:

            direct_view = ipp.Client()[:self.workers]  # use all engines
            result = direct_view.map(ParallelTester.helper, self.to_analyse)
            candidates = [candidate for result in result.get() for candidate in result]

        except Exception as e:
            print(e)

        return candidates


if __name__ == '__main__':
    _im_path = "book-covers/Art-Photography/0000001.jpg"
    _dataset_path = "book-covers"

    _th_workers = 10
    _pr_workers = 10
    _ip_workers = 10
    _nb_workers = 10

    _amount = 100
    _test = 1

    if _test == 0:
        tester = SequentialTester(ex.DEFAULT_DETECTOR, ex.DEFAULT_MATCHER,
                                  _im_path, _dataset_path, _amount)
    elif _test == 1:
        tester = ThreadingTester(ex.DEFAULT_DETECTOR, ex.DEFAULT_MATCHER,
                                 _im_path, _dataset_path, _amount, _th_workers)
    elif _test == 2:
        tester = MultiprocessingTester(ex.DEFAULT_DETECTOR, ex.DEFAULT_MATCHER,
                                       _im_path, _dataset_path, _amount, _pr_workers)
    elif _test == 3:
        tester = IPyParallelTester(ex.DEFAULT_DETECTOR, ex.DEFAULT_MATCHER,
                                   _im_path, _dataset_path, _amount, _ip_workers)
    else:
        tester = Tester(ex.DEFAULT_DETECTOR, ex.DEFAULT_MATCHER,
                        _im_path, _dataset_path, _amount)

    _candidates, time = tester.run()
    print(len(_candidates), 'candidates in', round(time, 4), 'secs')
