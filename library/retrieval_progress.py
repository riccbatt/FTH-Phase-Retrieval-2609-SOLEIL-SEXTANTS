"""Compact notebook progress displays for the phase retrieval drivers."""

from collections import deque
from contextlib import contextmanager, redirect_stdout
import re

from tqdm.auto import tqdm


class _ProgressStream:
    def __init__(self, bar, kind):
        self.bar = bar
        self.kind = kind
        self.buffer = ""
        self.recent = deque(maxlen=8)
        self.active_observation = False
        self.finished = False

    def write(self, text):
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line.strip())
        return len(text)

    def flush(self):
        pass

    def _complete_observation(self):
        if self.active_observation:
            self.bar.update(1)
            self.active_observation = False

    def _line(self, line):
        if not line:
            return
        self.recent.append(line)
        if self.kind == "unified":
            match = re.match(r"Step (\d+): helicity=(.*?), mode=", line)
            if match:
                self.bar.set_postfix_str(
                    f"{match.group(2)} · stage {int(match.group(1)) + 1}", refresh=False
                )
                self.bar.update(1)
            return
        if "Warmup observation " in line or "Warmup energy " in line:
            self._complete_observation()
            self.bar.set_postfix_str("warmup", refresh=False)
            self.active_observation = True
        elif "warmup complete" in line:
            self._complete_observation()
        elif "Coherent refresh observation " in line:
            self._complete_observation()
            self.bar.set_postfix_str("coherent refresh", refresh=False)
            self.active_observation = True
        elif "Updating observation " in line or "Updating energy " in line:
            self._complete_observation()
            self.bar.set_postfix_str("joint retrieval", refresh=False)
            self.active_observation = True
        elif ("Applying joint physical projection" in line
              or "Applying joint energy projection" in line):
            self._complete_observation()
            self.bar.set_postfix_str("physical projection", refresh=False)
        elif ("applying final physical projection" in line
              or "applying final energy projection" in line):
            self._complete_observation()
            self.bar.set_postfix_str("final projection", refresh=False)
        elif "Universal phase retrieval: complete" in line:
            self._complete_observation()
            if self.bar.n < self.bar.total:
                self.bar.update(self.bar.total - self.bar.n)
            self.bar.set_postfix_str("complete", refresh=False)
            self.finished = True


@contextmanager
def retrieval_progress(recipe, n_observations=None, *, kind, leave=True,
                       description="Phase retrieval"):
    """Replace repetitive driver stdout with an updating notebook bar.

    ``kind`` is ``"unified"`` for stage-based retrieval or ``"universal"``
    for a joint physical or pure-energy driver. Exceptions remain visible, and the last
    driver messages are shown on failure to aid debugging.
    """
    if kind == "unified":
        total = len(recipe["helicity"])
        unit = "stage"
    elif kind == "universal":
        if n_observations is None or n_observations < 1:
            raise ValueError("n_observations must be positive")
        warmup = bool(recipe.get("warmup_mode")) and any(
            n > 0 for n in recipe.get("warmup_Nit", [])
        )
        warmup_passes = int(warmup)
        if (warmup and recipe.get("preserve_warmup_masked_intensity")
                and recipe.get("partial_coherence")):
            warmup_passes = 2
        total = n_observations * (int(recipe["outer_iterations"]) + warmup_passes + len(recipe.get("coherent_refresh_rounds", []))) + 1
        unit = "step"
    else:
        raise ValueError("kind must be 'unified' or 'universal'")
    with tqdm(total=total, desc=description, unit=unit,
              mininterval=0.2, leave=leave) as bar:
        stream = _ProgressStream(bar, kind)
        try:
            with redirect_stdout(stream):
                yield bar
        except Exception:
            if stream.buffer.strip():
                stream._line(stream.buffer.strip())
            tqdm.write("Recent phase retrieval messages:")
            for line in stream.recent:
                tqdm.write(f"  {line}")
            raise
