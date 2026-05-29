"""ForwardGenContext reentrancy guard.

R14 converted the module-level ``_forward_gen_context`` singleton in
``generate/_context.py`` to a ``threading.local()`` store so two
concurrent ``generate_pytorch_module`` calls (or any other parallel
use of forward generation) cannot stomp on each other's accumulator
state. These tests pin the new contract.
"""

__docformat__ = "restructuredtext"

import threading
from concurrent.futures import ThreadPoolExecutor

from torchonnx.generate._context import (
    ForwardGenContext,
    get_forward_gen_context,
    set_forward_gen_context,
)


class TestThreadLocalIsolation:
    """``_forward_gen_context`` is per-thread."""

    def test_new_thread_starts_with_no_context(self):
        """A freshly-spawned thread must not see the main thread's context."""
        main_ctx = ForwardGenContext()
        set_forward_gen_context(main_ctx)
        try:

            def worker():
                # The worker thread must see None even though the main
                # thread has a context set.
                return get_forward_gen_context()

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(worker)
                assert future.result() is None
        finally:
            set_forward_gen_context(None)

    def test_workers_do_not_see_each_others_contexts(self):
        """Two worker threads each have their own slot in the thread-local store."""
        captured: dict[int, ForwardGenContext | None] = {}
        barrier = threading.Barrier(parties=2)

        def worker(my_id: int) -> None:
            ctx = ForwardGenContext()
            ctx.first_input_name = f"x_thread_{my_id}"
            set_forward_gen_context(ctx)
            # Wait so both threads have written before either reads back.
            barrier.wait()
            captured[my_id] = get_forward_gen_context()

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert captured[1] is not None
        assert captured[2] is not None
        assert captured[1] is not captured[2]
        assert captured[1].first_input_name == "x_thread_1"
        assert captured[2].first_input_name == "x_thread_2"

    def test_clearing_does_not_leak_across_threads(self):
        """``set_forward_gen_context(None)`` on one thread leaves others alone."""
        worker_ctx_box: list[ForwardGenContext | None] = [None]
        ready = threading.Event()
        proceed = threading.Event()

        def worker() -> None:
            worker_ctx_box[0] = ForwardGenContext()
            set_forward_gen_context(worker_ctx_box[0])
            ready.set()
            proceed.wait(timeout=5.0)
            # After main thread cleared its own context, worker still sees its own.
            assert get_forward_gen_context() is worker_ctx_box[0]

        main_ctx = ForwardGenContext()
        set_forward_gen_context(main_ctx)
        t = threading.Thread(target=worker)
        t.start()
        ready.wait(timeout=5.0)
        set_forward_gen_context(None)
        assert get_forward_gen_context() is None
        proceed.set()
        t.join(timeout=5.0)
        assert not t.is_alive()
