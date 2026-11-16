# Copyright © 2023 Apple Inc.

import contextlib
import os
import platform
import sys
import unittest
from typing import Any, Callable, List, Tuple, Union

import mlx.core as mx
import numpy as np


@contextlib.contextmanager
def scoped_env(**environ):
    """
    Temporarily set the process environment variables.

    Passing a value of None removes the variable for the duration of the context.
    """
    old_environ = dict(os.environ)
    for key, value in environ.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class MLXTestRunner(unittest.TestProgram):
    def __init__(self, *args, **kwargs):
        # Do not exit in runTests
        kwargs["exit"] = False
        super().__init__(*args, **kwargs)

    def createTests(self, *args, **kwargs):
        super().createTests(*args, **kwargs)

        # Check if we're running on a non-Metal GPU backend (CUDA or ROCm)
        device_name = os.getenv("DEVICE", None)
        if device_name is not None:
            device = getattr(mx, device_name)
        else:
            device = mx.default_device()

        skip_tests, _ = _get_backend_skip_tests(device)

        if not skip_tests:
            return

        filtered_suite = unittest.TestSuite()

        def filter_and_add(t):
            if isinstance(t, unittest.TestSuite):
                for sub_t in t:
                    filter_and_add(sub_t)
            else:
                t_id = ".".join(t.id().split(".")[-2:])
                if t_id in skip_tests:
                    print(f"Skipping {t_id}")
                else:
                    filtered_suite.addTest(t)

        filter_and_add(self.test)
        self.test = filtered_suite

    def runTests(self):
        super().runTests()
        mx.clear_streams()
        sys.exit(0 if self.result.wasSuccessful() else 1)


class MLXTestCase(unittest.TestCase):
    @property
    def is_apple_silicon(self):
        return platform.machine() == "arm64" and platform.system() == "Darwin"

    def setUp(self):
        self.default = mx.default_device()

        device_name = os.getenv("DEVICE", None)
        if device_name is not None:
            device = getattr(mx, device_name)
        else:
            device = self.default

        skip_tests, backend = _get_backend_skip_tests(device)
        test_id = f"{self.__class__.__name__}.{self._testMethodName}"
        if test_id in skip_tests:
            self.skipTest(f"Skipped on {backend} backend")

        if device_name is not None:
            mx.set_default_device(device)

    def tearDown(self):
        mx.set_default_device(self.default)

    # Note if a tuple is passed into args, it will be considered a shape request and convert to a mx.random.normal with the shape matching the tuple
    def assertCmpNumpy(
        self,
        args: List[Union[Tuple[int], Any]],
        mx_fn: Callable[..., mx.array],
        np_fn: Callable[..., np.array],
        atol=1e-2,
        rtol=1e-2,
        dtype=mx.float32,
        **kwargs,
    ):
        assert dtype != mx.bfloat16, "numpy does not support bfloat16"
        args = [
            mx.random.normal(s, dtype=dtype) if isinstance(s, Tuple) else s
            for s in args
        ]
        mx_res = mx_fn(*args, **kwargs)
        np_res = np_fn(
            *[np.array(a) if isinstance(a, mx.array) else a for a in args], **kwargs
        )
        return self.assertEqualArray(mx_res, mx.array(np_res), atol=atol, rtol=rtol)

    def assertEqualArray(
        self,
        mx_res: mx.array,
        expected: mx.array,
        atol=1e-2,
        rtol=1e-2,
    ):
        self.assertEqual(
            tuple(mx_res.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={mx_res.shape}",
        )
        self.assertEqual(
            mx_res.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}",
        )
        if not isinstance(mx_res, mx.array) and not isinstance(expected, mx.array):
            np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
            return
        elif not isinstance(mx_res, mx.array):
            mx_res = mx.array(mx_res)
        elif not isinstance(expected, mx.array):
            expected = mx.array(expected)
        self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
