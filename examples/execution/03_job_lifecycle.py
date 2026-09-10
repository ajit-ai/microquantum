"""Execution example 3: job lifecycle.

:func:`microquantum.submit` returns a :class:`Job` — a lifecycle envelope
with timestamps, backend/target identity and cancellation semantics. Local
simulators return already-completed jobs, but the metadata shape is the same
one an asynchronous hardware provider would use.
"""

from microquantum import JobStatus, QuantumCircuit, submit, submit_batch


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    job = submit(qc, shots=256, seed=5)

    print("=== Job ===")
    print(f"status   : {job.status.value}")
    print(f"terminal : {job.status.final}")
    print(f"created  : {job.created_at}")
    print(f"finished : {job.finished_at}")
    print(f"backend  : {job.backend_name}")
    print("metadata :")
    for key, value in job.metadata().items():
        print(f"  {key}: {value}")

    print("\n=== Submitting a batch of jobs ===")
    jobs = submit_batch([qc, qc, qc], shots=128, seed=1)
    statuses = {j.status for j in jobs}
    print(f"all completed: {statuses == {JobStatus.COMPLETED}}")


if __name__ == "__main__":
    main()