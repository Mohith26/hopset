"""Fixed priority preemptive scheduling, the part that actually has to hold.

A hopping radio is a hard real time system before it is anything else. The
synthesiser has to be retuned and settled inside every dwell, and a dwell at
111 hops per second is 9 milliseconds long. Miss it and you are not degraded,
you are off the net, because the receiver is listening on a channel you are
not transmitting on. So "it usually finishes in time" is not a specification.

This module has three pieces that check each other:

  * `response_time_analysis`, the standard fixed point recurrence that gives a
    worst case response time per task,
  * `simulate`, a tick accurate preemptive scheduler, and
  * `Resource`, a shared mutex with optional priority inheritance, so the
    classic unbounded priority inversion can be produced on demand and then
    fixed.

The analysis and the simulator are written independently and are supposed to
disagree in exactly one direction: no observed response time may ever exceed
the analytic bound. If simulation ever beats the bound, the bound is wrong. If
the bound is wildly above what is ever observed, the analysis is being
pessimistic and it is worth knowing by how much.
"""

import math
from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    period: int          # ticks
    wcet: int            # ticks of compute per job
    deadline: int = None  # defaults to the period
    cs_len: int = 0       # ticks of that compute spent holding the resource
    cs_start: int = 0     # offset into the job where the critical section begins
    priority: int = None  # lower value is higher priority
    offset: int = 0       # first release time, for demonstrating specific interleavings

    def __post_init__(self):
        if self.deadline is None:
            self.deadline = self.period
        if self.cs_len and self.cs_start + self.cs_len > self.wcet:
            raise ValueError("critical section does not fit inside the job")

    @property
    def utilization(self):
        return self.wcet / float(self.period)


def assign_rate_monotonic(tasks):
    """Shortest period gets the highest priority.

    Optimal among fixed priority assignments for implicit deadline task sets,
    which is why it is what you find in a radio's RTOS configuration rather
    than something hand tuned.
    """
    ordered = sorted(tasks, key=lambda t: (t.period, t.name))
    for i, t in enumerate(ordered):
        t.priority = i
    return ordered


def total_utilization(tasks):
    return sum(t.utilization for t in tasks)


def liu_layland_bound(n):
    """n * (2 ** (1/n) - 1). Sufficient, never necessary.

    Converges to ln 2, about 0.693. A task set above this is not necessarily
    unschedulable, it just is not settled by the utilization test alone, which
    is exactly the situation response time analysis exists for.
    """
    return n * (2.0 ** (1.0 / n) - 1.0)


def blocking_time(tasks, target):
    """Worst case blocking under the priority inheritance protocol.

    My first version of this only counted direct blocking: a task is delayed
    only if it touches the resource itself. The simulator disagreed with the
    analysis by a factor of nearly three on a task that has no critical
    section at all, and the simulator was right.

    The mechanism is push-through blocking. When a low priority task holds a
    resource that a high priority task is waiting on, inheritance raises the
    holder to the waiter's priority. It now outranks every task in between,
    including tasks that never touch the resource and, under the naive model,
    should not be affected at all. Those tasks get pushed through by a critical
    section they have no relationship to.

    So the correct condition is about the resource, not the task: a lower
    priority task's critical section can block task i if the resource it guards
    is used by any task whose priority is at least as high as i. With one
    shared resource that is the longest such critical section.
    """
    users = [t for t in tasks if t.cs_len > 0]
    if not any(u.priority <= target.priority for u in users):
        return 0
    lower = [t for t in users if t.priority > target.priority]
    return max([t.cs_len for t in lower], default=0)


def response_time_analysis(tasks, use_blocking=True):
    """R_i = C_i + B_i + sum over higher priority j of ceil(R_i / T_j) * C_j.

    Solved by fixed point iteration from R = C + B, which is monotonically
    increasing, so it either converges or crosses the deadline and the task set
    is rejected.
    """
    results = {}
    for t in tasks:
        higher = [h for h in tasks if h.priority < t.priority]
        blocking = blocking_time(tasks, t) if use_blocking else 0
        r = t.wcet + blocking
        for _ in range(10000):
            interference = sum(
                math.ceil(r / float(h.period)) * h.wcet for h in higher
            )
            nxt = t.wcet + blocking + interference
            if nxt == r:
                break
            r = nxt
            if r > t.deadline:
                break
        results[t.name] = {
            "response_time": r,
            "deadline": t.deadline,
            "blocking": blocking,
            "schedulable": r <= t.deadline,
        }
    return results


def rta_schedulable(tasks, use_blocking=True):
    return all(v["schedulable"] for v in response_time_analysis(tasks, use_blocking).values())


@dataclass
class _Job:
    task: Task
    released: int
    done: int = 0
    holding: bool = False
    finished_at: int = None


@dataclass
class SimResult:
    horizon: int
    worst_response: dict = field(default_factory=dict)
    deadline_misses: dict = field(default_factory=dict)
    completions: dict = field(default_factory=dict)
    worst_blocking: dict = field(default_factory=dict)
    idle_ticks: int = 0

    @property
    def any_miss(self):
        return any(v > 0 for v in self.deadline_misses.values())


def simulate(tasks, horizon, priority_inheritance=True):
    """Tick accurate fixed priority preemptive simulation. With every offset at
    zero this starts from a critical instant, meaning all tasks released
    together, which is the worst case release pattern for fixed priority
    scheduling. Non zero offsets exist to reproduce a specific interleaving.

    One shared resource, entered at `cs_start` ticks into a job and released
    `cs_len` ticks later.
    """
    for t in tasks:
        if t.priority is None:
            raise ValueError("assign priorities first")
    jobs = []
    worst = {t.name: 0 for t in tasks}
    misses = {t.name: 0 for t in tasks}
    completions = {t.name: 0 for t in tasks}
    worst_block = {t.name: 0 for t in tasks}
    blocked_since = {}
    idle = 0

    for tick in range(horizon):
        for t in tasks:
            if tick >= t.offset and (tick - t.offset) % t.period == 0:
                jobs.append(_Job(task=t, released=tick))

        ready = [j for j in jobs if j.finished_at is None and j.released <= tick]
        if not ready:
            idle += 1
            continue

        holder = next((j for j in ready if j.holding), None)

        def wants_resource(job):
            return (
                job.task.cs_len > 0
                and job.task.cs_start <= job.done < job.task.cs_start + job.task.cs_len
            )

        blocked = []
        if holder is not None:
            blocked = [j for j in ready if j is not holder and wants_resource(j)]

        effective = {}
        for j in ready:
            effective[id(j)] = j.task.priority
        if holder is not None and priority_inheritance and blocked:
            effective[id(holder)] = min(
                [holder.task.priority] + [b.task.priority for b in blocked]
            )

        runnable = [j for j in ready if j not in blocked]
        if not runnable:
            idle += 1
            continue

        chosen = min(runnable, key=lambda j: (effective[id(j)], j.released, j.task.name))

        for b in blocked:
            blocked_since.setdefault(id(b), tick)
        for j in ready:
            if j is chosen and id(j) in blocked_since:
                span = tick - blocked_since.pop(id(j))
                worst_block[j.task.name] = max(worst_block[j.task.name], span)

        if wants_resource(chosen) and not chosen.holding:
            chosen.holding = True
        chosen.done += 1
        if chosen.holding and chosen.done >= chosen.task.cs_start + chosen.task.cs_len:
            chosen.holding = False
        if chosen.done >= chosen.task.wcet:
            chosen.finished_at = tick + 1
            response = chosen.finished_at - chosen.released
            worst[chosen.task.name] = max(worst[chosen.task.name], response)
            completions[chosen.task.name] += 1
            if response > chosen.task.deadline:
                misses[chosen.task.name] += 1

        jobs = [j for j in jobs if j.finished_at is None]

    return SimResult(
        horizon=horizon,
        worst_response=worst,
        deadline_misses=misses,
        completions=completions,
        worst_blocking=worst_block,
        idle_ticks=idle,
    )


def hyperperiod(tasks):
    h = 1
    for t in tasks:
        h = h * t.period // math.gcd(h, t.period)
    return h


def radio_task_set(tick_us=100):
    """A plausible task set for a hopping tactical radio, in ticks of 100 us.

    Numbers are illustrative rather than taken from any real product. The
    shape is what matters: a fast hop timer and keystream advance that must
    finish inside a dwell, a slower vocoder and modem frame that dominates
    utilisation, periodic built in test, and a slow operator interface. Built
    in test holds the same synthesiser control resource as the hop timer, which
    is what creates the inversion.
    """
    return [
        Task("hop_timer", period=90, wcet=4, cs_len=2, cs_start=1),
        Task("keystream", period=90, wcet=6),
        Task("modem_frame", period=200, wcet=100),
        Task("operator_ui", period=500, wcet=20),
        Task("built_in_test", period=1000, wcet=30, cs_len=24, cs_start=3),
    ]
