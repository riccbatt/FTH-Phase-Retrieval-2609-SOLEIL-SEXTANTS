"""Optional notebook observer; it never changes reconstruction fields or recipes."""
import time
import numpy as np


class LiveReconstruction:
    """Update one display at completed physical fits and on final output.

    Select observation indices, not modal indices. Each row shows its state's
    fitted magnetization and the amplitude/phase of every exit-wave mode.
    Throttling limits plotting cost; only selected observations are transformed.
    Callback arrays are borrowed read-only: no full reconstruction history is kept.
    """
    def __init__(self, observations=(0,), every=1, min_seconds=2.0):
        if not isinstance(every, int) or every < 1 or min_seconds < 0:
            raise ValueError("every must be positive and min_seconds nonnegative")
        self.observations = tuple(dict.fromkeys(observations))
        if not self.observations:
            raise ValueError("Select at least one observation")
        self.every, self.min_seconds = every, min_seconds
        self.count, self.last = 0, -float('inf')
        self.figure = self.axes = self.handle = None

    def __call__(self, event):
        from matplotlib import pyplot as plt
        from IPython.display import display
        from .phase_retrieval_geometry import support_bounding_roi
        from .phase_retrieval_universal import _projection_focus_transform
        self.count += 1
        final = event['stage'] == 'final'
        if not final and ((self.count - 1) % self.every or
                          time.monotonic() - self.last < self.min_seconds):
            return
        fields, components, recipe = event['fields'], event['components'], event['recipe']
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(fields)
               for i in self.observations):
            raise ValueError("Live observation index outside field stack")
        nmodes = fields.shape[1] if fields.ndim == 4 else 1
        roi = support_bounding_roi(event['supportmask'])
        if self.figure is None:
            self.figure, self.axes = plt.subplots(len(self.observations), 1 + 2*nmodes,
                figsize=(4*(1+2*nmodes), 3.5*len(self.observations)), squeeze=False)
            # Use a display handle rather than clearing the cell (which would
            # also erase its progress bar and printed recipe).
            plt.close(self.figure)
        for row, index in enumerate(self.observations):
            axes = self.axes[row]
            for ax in axes:
                ax.clear(); ax.set_axis_off()
            state = event['state_labels'][index]
            names = list(components.get('state_names', []))
            magnetization = components.get('magnetization')
            if magnetization is not None and state in names:
                magnetic = np.fft.ifftshift(magnetization[names.index(state)])[roi]
                axes[0].imshow(magnetic, cmap='RdBu_r', vmin=-1, vmax=1)
            else:
                axes[0].text(.5,.5,'No fitted magnetization',ha='center')
            axes[0].set_title(f'{state}: magnetization [-1, 1]')
            modes = fields[index] if fields.ndim == 4 else fields[index][None]
            for mode, field in enumerate(modes):
                # The shared focus helper expects at least two observations.
                # Duplicate only this selected plane, then discard the second.
                focused = _projection_focus_transform(np.stack([field, field]),
                    [event['energy_labels'][index]] * 2,
                    recipe['projection_focus_prop_um'], recipe['projection_focus_phase_rad'],
                    recipe['projection_focus_setup'],
                    integer_wavelength=recipe['projection_focus_integer_wavelength'])[0]
                obj = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(focused)))[roi]
                axes[1+2*mode].imshow(abs(obj), cmap='gray', vmin=0)
                axes[1+2*mode].set_title(f'Observation {index}, mode {mode+1}: amplitude (auto scale)')
                axes[2+2*mode].imshow(np.angle(obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
                axes[2+2*mode].set_title(f'Mode {mode+1}: phase [-π, π]')
        self.figure.suptitle(f"{event['stage']} · outer round {event['outer_round']}\n"
                            'Magnetization: latest physical fit; exit waves: current fields')
        self.figure.tight_layout()
        if self.handle is None:
            self.handle = display(self.figure, display_id=True)
        else:
            self.handle.update(self.figure)
        self.last = time.monotonic()
