"""Batch prefetch support for generation."""

import random
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from utils.logger import print_log
from utils.random_seed import setup_seed


_PREFETCH_DATASET = None


def _init_batch_prefetch_worker(dataset, torch_num_threads):
    global _PREFETCH_DATASET
    _PREFETCH_DATASET = dataset
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))


def _setup_prefetch_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))


def _collate_batch_worker(task):
    batch_list, task_seed = task
    if task_seed is not None:
        _setup_prefetch_seed(task_seed)
    return batch_list, _PREFETCH_DATASET.collate_fn([
        _PREFETCH_DATASET[i]
        for i, _ in batch_list
    ])


class BatchPrefetcher:

    def __init__(
            self,
            dataset,
            recorder,
            batch_size,
            num_workers=0,
            prefetch_batches=2,
            backend='process',
            start_method='spawn',
            seed=None,
            torch_num_threads=1,
        ):
        self.dataset = dataset
        self.recorder = recorder
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.backend = str(backend).lower()
        self.start_method = str(start_method)
        self.seed = seed
        self.torch_num_threads = torch_num_threads
        self.executor = None
        self.pending = deque()
        self.task_index = 0

    def __enter__(self):
        if self.num_workers > 0:
            if self.backend == 'thread':
                self.executor = ThreadPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_batch_prefetch_worker,
                    initargs=(self.dataset, None),
                )
            elif self.backend == 'process':
                mp_context = mp.get_context(self.start_method)
                self.executor = ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_batch_prefetch_worker,
                    initargs=(self.dataset, self.torch_num_threads),
                    mp_context=mp_context,
                )
            else:
                raise ValueError('generation prefetch backend must be "process" or "thread"')
            self._fill()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._shutdown_executor()

    def _shutdown_executor(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def _next_seed(self):
        if self.seed is None:
            return None
        return int(self.seed) + self.task_index * 1000003

    def _prepare_serial(self, batch_list, task_seed=None):
        if task_seed is not None:
            setup_seed(int(task_seed))
        return batch_list, self.dataset.collate_fn([
            self.dataset[i]
            for i, _ in batch_list
        ])

    def _submit_next(self):
        if self.recorder.is_finished():
            return False
        batch_list = self.recorder.get_next_batch_list(self.batch_size)
        if not batch_list:
            return False
        task_seed = self._next_seed() if self.backend == 'process' and self.executor is not None else None
        self.task_index += 1
        if self.executor is None:
            self.pending.append(('ready', batch_list, self._prepare_serial(batch_list, task_seed)))
        else:
            try:
                self.pending.append((
                    'future',
                    batch_list,
                    self.executor.submit(
                        _collate_batch_worker,
                        (batch_list, task_seed),
                    ),
                ))
            except Exception as exc:
                print_log(
                    f'Generation prefetch worker submission failed; '
                    f'falling back to serial batch preparation: {exc}',
                    level='WARN',
                )
                self._shutdown_executor()
                self.pending.append(('ready', batch_list, self._prepare_serial(batch_list)))
        return True

    def _fill(self):
        while len(self.pending) < self.prefetch_batches:
            if not self._submit_next():
                break

    def next(self):
        if self.executor is None:
            if not self.pending:
                self._submit_next()
            if not self.pending:
                raise StopIteration
            kind, batch_list, payload = self.pending.popleft()
            if kind == 'ready':
                return payload
            return self._prepare_serial(batch_list)

        self._fill()
        if not self.pending:
            raise StopIteration
        kind, batch_list, payload = self.pending.popleft()
        if kind == 'ready':
            return payload
        future = payload
        try:
            result = future.result()
        except Exception as exc:
            print_log(
                f'Generation batch prefetch failed; falling back to serial batch '
                f'preparation. Original error: {exc}',
                level='WARN',
            )
            self._shutdown_executor()
            result = self._prepare_serial(batch_list)
        if self.executor is not None:
            self._fill()
        return result
